"""Prepare aviation archives and build en-route weather features."""

import json
from datetime import date, timedelta
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.io import sha256, write_json
from scripts.common.progress import pipeline_stage, stage
from scripts.common.references import load_airports
from scripts.common.stage_cache import consumed, logic_hash, resolve_stage
from scripts.common.time_utils import local_cutoff

from . import clean_sigmets, clean_taf
from .build_features import IDENTITY, attach_schedule, effective_airport, enrich
from .download import download


def validate_cohort(original, enriched, *, context):
    """Check unique IDs and exact identity equality in both directions."""
    for frame in (original, enriched):
        stats = frame.agg(F.count("*").alias("rows"), F.countDistinct("flight_id").alias("ids")).first()
        if not stats.rows or stats.rows != stats.ids:
            raise ValueError(f"{context} requires nonempty cohorts with nonnull, unique flight IDs.")
    # Unique IDs allow a single-pass merge-join instead of two full-table exceptAll shuffles.
    before_rows = original.select("flight_id", F.struct(*IDENTITY).alias("_before"))
    after_rows = enriched.select("flight_id", F.struct(*IDENTITY).alias("_after"))
    if (
        before_rows.hint("merge")
        .join(after_rows, "flight_id", "full")
        .filter(~F.col("_before").eqNullSafe(F.col("_after")))
        .limit(1)
        .count()
    ):
        raise ValueError(f"{context} enrichment changed the cohort.")


def prepare(spark: SparkSession, root: Path, config: dict, weather_gold: Path) -> dict:
    """Download configured UTC coverage and materialize typed deduplicated Silver tables."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    cfg = config["datasets"]
    references = project_path(root, config["destination_weather"]["reference_directory"])
    airport_digest = sha256(references / "airports.csv")
    airports = load_airports(references)
    first = date.fromisoformat(config["splits"]["train"]["start_date"])
    last = date.fromisoformat(config["splits"]["test"]["end_date"])
    limit = cfg["taf"]["station_limit"]
    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("datasets.taf.station_limit must be between 1 and 50.")

    def select(directory):
        training = spark.read.parquet(str(weather_gold / "train"))
        counts = training.groupBy("Dest").count().orderBy(F.desc("count"), "Dest").collect()
        selected = [(site, row["count"]) for row in counts[:limit]
                    if (site := effective_airport(airports, row.Dest, first).get("icao"))]
        write_json(directory / "stations.json", {
            "stations": sorted({site for site, _ in selected}),
            "share": sum(count for _, count in selected) / sum(row["count"] for row in counts),
        })
        return {"destination_train": str(weather_gold / "train")}

    def selection_inputs():
        return {"destination": consumed(weather_gold, ("train",)), "limit": limit, "date": first.isoformat(),
                "reference": airport_digest,
                "logic": logic_hash(select, effective_airport, load_airports)}

    selection = resolve_stage(root / "data/silver/taf_stations", selection_inputs, (), ("stations.json",), select)
    selected = json.loads((selection / "stations.json").read_text())
    stations = selected["stations"]
    with stage("Download and validate public IEM aviation archives"):
        taf_inputs, sigmet_inputs = download(root, first - timedelta(days=2), last + timedelta(days=2),
                                            stations, taf=cfg["taf"], sigmets=cfg["sigmets"])
    inputs = {**taf_inputs, **sigmet_inputs, references / "airports.csv": airport_digest}
    tables = {}
    with stage("Build TAF segments and convective SIGMET Silver"):
        for kind, files, cleaner in (("taf", taf_inputs, clean_taf), ("sigmets", sigmet_inputs, clean_sigmets)):
            def silver_inputs():
                return {"archives": {str(path.relative_to(root)): digest for path, digest in files.items()},
                        "stations": stations if kind == "taf" else [],
                        "coverage": [first.isoformat(), last.isoformat()], "parser": sha256(cleaner.__file__),
                        "schema": cleaner.SCHEMA, "writer": logic_hash(build)}

            def build(directory):
                raw = spark.sparkContext.parallelize([str(path) for path in files], len(files))
                table = directory / "table"
                spark.createDataFrame(raw.flatMap(cleaner.read_archive), cleaner.SCHEMA).distinct().write.parquet(str(table))
                if kind == "taf" and not spark.read.parquet(str(table)).take(1):
                    raise ValueError("No usable TAFs were found for the configured training destinations.")

            tables[kind] = resolve_stage(project_path(root, cfg[kind]["silver_directory"]), silver_inputs,
                                         ("table",), (), build) / "table"
    return {
        "airports": airports, "stations": stations, "inputs": inputs, "tables": tables,
        "selected_training_destination_share": selected["share"], "reference": references / "airports.csv",
    }


def prepare_gold(spark, root, rotation_gold, weather_gold, prepared):
    splits = ("train", "val", "test")

    def inputs():
        return {"rotation": consumed(rotation_gold.parent, tuple(f"gold/{split}" for split in splits)),
                "destination": consumed(weather_gold, splits),
                "aviation": {kind: consumed(table.parent, ("table",)) for kind, table in prepared["tables"].items()},
                "stations": prepared["stations"], "reference": sha256(prepared["reference"]),
                "code": {name: sha256(root / "scripts" / name) for name in
                         ("enroute_weather/build_features.py", "aircraft_rotation/build_features.py", "bts/recent_delays.py")},
                "logic": logic_hash(build, validate_cohort, local_cutoff)}

    def build(gold):
        taf = spark.read.parquet(str(prepared["tables"]["taf"])).where(F.col("station").isin(prepared["stations"]))
        sigmets = spark.read.parquet(str(prepared["tables"]["sigmets"]))
        coverage = {}
        with stage("Join cutoff-safe TAF segments and convective SIGMETs"):
            for split in splits:
                print(f"    Enriching {split} split with destination TAFs and convective SIGMETs...", flush=True)
                parent = spark.read.parquet(str(rotation_gold / split))
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
        write_json(gold / "coverage.json", coverage)
        return {"destination": str(weather_gold), "rotation": str(rotation_gold),
                "aviation": {kind: str(table) for kind, table in prepared["tables"].items()}}

    gold = resolve_stage(root / "data/gold/enroute", inputs, splits, ("coverage.json",), build)
    return gold, json.loads((gold / "coverage.json").read_text())


def run(ctx: PipelineContext) -> None:
    """Stage 5 entry point: prepare TAF/SIGMET tables, build en-route Gold, and update `ctx`."""
    with pipeline_stage(5, 6, "En-Route Weather (TAF & SIGMET)", timings=ctx.timings):
        weather_gold = ctx.destination_gold_dir or ctx.gold_dir
        rotation_gold = ctx.rotation_gold_dir or ctx.gold_dir
        aviation = prepare(ctx.spark, ctx.root, ctx.config, weather_gold)
        gold, aviation_coverage = prepare_gold(ctx.spark, ctx.root, rotation_gold, weather_gold, aviation)
        ctx.gold_dir = gold
        ctx.coverage["enroute"] = aviation_coverage
        ctx.sources["sha256"].update(
            {path.relative_to(ctx.root).as_posix(): digest for path, digest in aviation["inputs"].items()}
        )
        ctx.sources["taf_stations"] = aviation["stations"]
        ctx.sources["selected_training_destination_share"] = aviation["selected_training_destination_share"]
