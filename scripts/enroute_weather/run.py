"""Prepare IEM archives and publish a paired en-route weather ablation."""

from datetime import date, timedelta
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from scripts.bts import train as trainer
from scripts.bts.recent_delays import load_airports
from scripts.origin_weather.download import project_path, sha256, write_json
from scripts.progress import stage
from scripts.results import update_results

from . import clean_sigmets, clean_taf
from .build_features import IDENTITY, attach_schedule, effective_airport, enrich
from .download import download


def validate_cohort(original, enriched, *, context):
    """Check unique IDs and exact identity equality in both directions."""
    for frame in (original, enriched):
        stats = frame.agg(F.count("*").alias("rows"), F.countDistinct("flight_id").alias("ids")).first()
        if not stats.rows or stats.rows != stats.ids:
            raise ValueError(f"{context} requires nonempty cohorts with nonnull, unique flight IDs.")
    original, enriched = original.select(*IDENTITY), enriched.select(*IDENTITY)
    if original.exceptAll(enriched).limit(1).count() or enriched.exceptAll(original).limit(1).count():
        raise ValueError(f"{context} enrichment changed the cohort.")


def comparison_metrics(frames, thresholds):
    """Measure paired validation deltas at selected and common thresholds."""
    comparison = {}
    for point in ("selected", "common_0_5"):
        arms = {}
        for arm, frame in frames.items():
            threshold = thresholds[arm] if point == "selected" else 0.5
            metrics = trainer.evaluate_predictions(frame.withColumn(
                "prediction", (F.col("p_delayed") >= threshold).cast("int")
            ))
            fp, tn = (metrics["confusion_matrix"][key] for key in ("fp", "tn"))
            arms[arm] = {
                **metrics, "threshold": threshold, "false_positives": fp,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            }
        baseline, enhanced = arms["tier4"], arms["enroute"]
        comparison[point] = {
            **arms,
            "pr_auc_delta": (enhanced["pr_auc"] - baseline["pr_auc"]
                             if baseline["pr_auc"] is not None else None),
            **{key + "_delta": enhanced[key] - baseline[key] for key in ("f1", "precision", "recall")},
            "false_positive_reduction": baseline["false_positives"] - enhanced["false_positives"],
            "false_positive_rate_reduction": (baseline["false_positive_rate"] - enhanced["false_positive_rate"]
                                              if baseline["false_positive_rate"] is not None else None),
        }
    return comparison


def prepare(spark: SparkSession, root: Path, config: dict, weather_gold: Path, run_id: str) -> dict:
    """Download configured UTC coverage and materialize typed deduplicated Silver tables."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    cfg = config["datasets"]
    references = project_path(root, config["destination_weather"]["reference_directory"])
    airports = load_airports(references)
    first = date.fromisoformat(config["splits"]["train"]["start_date"])
    last = date.fromisoformat(config["splits"]["test"]["end_date"])
    training = spark.read.parquet(str(weather_gold / "train"))
    counts = training.groupBy("Dest").count().orderBy(F.desc("count"), "Dest").collect()
    limit = cfg["taf"]["station_limit"]
    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("datasets.taf.station_limit must be between 1 and 50.")
    selected = [(site, row["count"]) for row in counts[:limit]
                if (site := effective_airport(airports, row.Dest, first).get("icao"))]
    stations = sorted({site for site, _ in selected})
    with stage("Stage: Download and validate public IEM aviation archives"):
        taf_inputs, sigmet_inputs = download(root, first - timedelta(days=2), last + timedelta(days=2),
                                            stations, taf=cfg["taf"], sigmets=cfg["sigmets"])
    inputs = {**taf_inputs, **sigmet_inputs, references / "airports.csv": sha256(references / "airports.csv")}
    tables = {}
    with stage("Stage: Build TAF segments and convective SIGMET Silver"):
        for kind, files, cleaner in (("taf", taf_inputs, clean_taf), ("sigmets", sigmet_inputs, clean_sigmets)):
            table = project_path(root, f"{cfg[kind]['silver_directory']}/{run_id}")
            raw = spark.sparkContext.parallelize([str(path) for path in files], len(files))
            spark.createDataFrame(raw.flatMap(cleaner.read_archive), cleaner.SCHEMA).distinct().write.parquet(str(table))
            tables[kind] = table
        if not spark.read.parquet(str(tables["taf"])).take(1):
            raise ValueError("No usable TAFs were found for the configured training destinations.")
    return {
        "airports": airports, "stations": stations, "inputs": inputs, "tables": tables,
        "selected_training_destination_share": sum(count for _, count in selected) / sum(row["count"] for row in counts),
    }


def fit(spark: SparkSession, root: Path, rotation: Path, weather_gold: Path, prepared: dict, baseline: dict) -> Path:
    """Fit the new tier and publish validation deltas against its rotation parent."""
    feature_set = trainer.ENROUTE_FEATURE_SET
    run_id = rotation.name
    gold = project_path(root, f"data/gold/enroute/{run_id}")
    output = project_path(root, f"models/enroute/{run_id}")
    taf = spark.read.parquet(str(prepared["tables"]["taf"])).where(F.col("station").isin(prepared["stations"]))
    sigmets = spark.read.parquet(str(prepared["tables"]["sigmets"]))
    coverage = {}
    with stage("Stage: Join cutoff-safe TAF segments and convective SIGMETs"):
        for split in ("train", "val", "test"):
            parent = spark.read.parquet(str((weather_gold if split == "test" else rotation / "gold") / split))
            schedule = spark.read.parquet(str(weather_gold / split))
            clocks = attach_schedule(parent, schedule, prepared["airports"]).persist(StorageLevel.DISK_ONLY)
            try:
                enrich(parent, clocks, taf, sigmets, prepared["stations"]).write.parquet(str(gold / split))
            finally:
                clocks.unpersist(blocking=True)
            stored = spark.read.parquet(str(gold / split))
            validate_cohort(parent, stored, context="En-route weather")
            coverage[split] = stored.agg(
                F.count("*").alias("flights"), F.sum("taf_sigmet_missing").alias("missing"),
                F.count("dest_taf_is_ifr").alias("taf_observed"),
                F.count(F.when(F.col("taf_fallback") & F.col("dest_taf_is_ifr").isNotNull(), 1)).alias("taf_fallback"),
                F.count("enroute_sigmet_convective_count").alias("sigmet_observed"),
                F.sum("enroute_sigmet_corridor_blocked").alias("sigmet_overlap"),
            ).first().asDict()
    if not coverage["train"]["taf_observed"]:
        raise ValueError("No point-in-time destination forecasts matched the training cohort.")
    with stage("Training en-route weather tier and comparing against rotation"):
        summary = trainer.train(spark, gold, output, feature_set=feature_set, save_predictions=True,
                                publish_results=False, model_name="model")
        frames = {
            "tier4": spark.read.parquet(str(rotation / "rot" / "validation_predictions")),
            "enroute": spark.read.parquet(str(output / "validation_predictions")),
        }
        validation = spark.read.parquet(str(gold / "val"))
        for frame in frames.values():
            validate_cohort(validation, frame, context="En-route validation")
        comparison = comparison_metrics(frames, {
            "tier4": baseline["selected_model"]["best_threshold"],
            "enroute": summary["selected_model"]["best_threshold"],
        })
    if any(sha256(path) != fingerprint for path, fingerprint in prepared["inputs"].items()):
        raise ValueError("An en-route input changed during the run.")
    report = output / "validation_comparison.json"
    write_json(report, {
        "run_id": run_id, "parent_run_id": run_id, "validation": comparison, "coverage": coverage,
        "scope": "Same-cohort validation. Selected thresholds were tuned on this validation cohort.",
        "stations": prepared["stations"], "gold": str(gold),
        "selected_training_destination_share": prepared["selected_training_destination_share"],
        "input_sha256": {str(path.relative_to(root)): value for path, value in prepared["inputs"].items()},
    })
    marker = output / "_SUCCESS"
    marker.touch(exist_ok=False)
    try:
        update_results({feature_set: {"training": summary, "validation": comparison, "coverage": coverage,
                                      "run_id": run_id, "parent_run_id": run_id}}, root / "results")
    except Exception:
        marker.unlink()
        raise
    return report
