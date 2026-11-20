"""Compare combined weather with and without aircraft rotation on validation."""

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

from scripts.bts import train as bts_train
from scripts.bts.build_features import get_spark_session
from scripts.bts.download import build_download_url, iter_months
from scripts.bts.cache import features_current
from scripts.bts.recent_delays import load_airports
from scripts.origin_weather.build_features import local_cutoff
from scripts.origin_weather.clean import file_fingerprints
from scripts.origin_weather.download import ROOT, load_config, project_path, sha256, write_json
from scripts.progress import stage
from scripts.weather_cache import consumed, find_gold, logic_hash, resolve_stage

from .build_features import (
    MAX_GAP_HOURS, OUTCOME_LAG_MINUTES, ROTATION_FLAGS, ROTATION_NUMERIC,
    build_operations, build_rotation_links,
)

FEATURE_SET = "bts_origin_destination_weather"
IDENTITY = ["flight_id", "FlightDate", "ArrDel15"]
AUDIT = [
    "scheduled_departure_utc", "rotation_reason", "prior_flight_id", "prior_scheduled_departure_utc",
    "prior_scheduled_arrival_utc", "prior_actual_arrival_utc", "prior_arrival_available_utc",
    "prior_departure_gap_minutes",
]
METRICS = ("pr_auc", "roc_auc", "f1", "precision", "recall")


def resolve_inputs(
    root: Path,
) -> tuple[dict[str, Any], Path, Path, list[Path]]:
    """Verify prerequisites and return config, references dir, parent Gold dir, and raw files."""
    settings, policy, config = load_config(root)
    if not features_current(root, config):
        raise ValueError("Phase 3 requires current BTS Gold.")
    origin = find_gold(root, settings, policy, config)
    parent = find_gold(root, settings, policy, config, origin) if origin is not None else None
    if parent is None:
        raise ValueError("Phase 3 requires compatible completed origin and destination weather Gold.")
    return config, settings["reference_directory"], parent, rotation_raw_files(root, config)


def rotation_raw_files(root: Path, config: dict[str, Any]) -> list[Path]:
    """Locate all BTS Bronze monthly CSV files covering the study period through validation."""
    bts = config["datasets"]["bts"]
    bronze = Path(bts["output_directory"])
    end_month = config["splits"]["val"]["end_date"][:7]
    if not bts["start_month"] <= end_month <= bts["end_month"]:
        raise ValueError("The configured BTS archive must cover the validation months.")
    raw_files = []
    for year, month in iter_months(bts["start_month"], end_month):
        archive = build_download_url(year, month).rsplit("/", 1)[-1]
        path = project_path(root, bronze / str(year) / f"{month:02d}" / Path(archive).with_suffix(".csv"))
        if not path.is_file():
            raise ValueError(f"Missing BTS rotation input: {path}")
        raw_files.append(path)
    return raw_files


def source_identity(
    root: Path,
    references: Path,
    parent: Path,
    raw_files: list[Path],
) -> dict[str, Any]:
    """Compute SHA-256 hashes of code/config and fingerprints of parent and raw data."""
    small_files = [
        root / "config.json", references / "airports.csv", references / "sources.json",
        parent / "_metadata.json", parent / "_reuse.json",
        *[root / "scripts" / path for path in (
            "aircraft_rotation/build_features.py", "aircraft_rotation/run.py",
            "bts/recent_delays.py", "bts/train.py", "origin_weather/build_features.py",
        )],
    ]
    return {
        "sha256": {str(path.relative_to(root)): sha256(path) for path in small_files},
        "raw_files": file_fingerprints(raw_files),
        "parent_files": file_fingerprints(
            path for split in ("train", "val") for path in (parent / split).rglob("*.parquet")
        ),
    }


def prepare_experiment(
    parent: DataFrame,
    operations: DataFrame,
    run_dir: Path,
    first_reliable_date: date,
    validation_bound: datetime,
) -> dict[str, Any]:
    """Link parent cohort to operations, filter training availability, and write isolated Gold splits."""
    totals = parent.agg(
        F.count("*").alias("rows"), F.countDistinct("flight_id").alias("unique_ids"),
        F.count(F.when(F.col("split") == "train", 1)).alias("train"),
        F.count(F.when(F.col("split") == "val", 1)).alias("val"),
    ).first()
    if (totals.rows != totals.unique_ids or totals.rows != totals.train + totals.val
            or not totals.train or not totals.val):
        raise ValueError("Rotation parent requires distinct nonnull IDs and nonempty train/val splits.")
    operations.persist(StorageLevel.DISK_ONLY)
    linked = None
    try:
        if operations.where("tail is not null and FlightDate is null").limit(1).count():
            raise ValueError("A known-tail operation lacks a service date for bounding its possible history.")
        bound = F.to_timestamp(F.lit(validation_bound.isoformat()))
        linked = (
            build_rotation_links(parent, operations, first_reliable_date)
            .withColumn(
                "training_label_status",
                F.when(F.col("split") == "val", "validation")
                .when(F.col("label_available_utc").isNull()
                      | ~F.coalesce(F.col("label_matches"), F.lit(False)), "unresolved")
                .when(F.col("label_available_utc") >= bound, "after_bound").otherwise("eligible"),
            )
            .persist(StorageLevel.DISK_ONLY)
        )
        coverage = linked.groupBy(
            "split", F.trunc("FlightDate", "month").cast("string").alias("month"),
            "rotation_reason", "training_label_status",
        ).count()
        coverage = [row.asDict() for row in coverage.orderBy(
            "split", "month", "rotation_reason", "training_label_status"
        ).collect()]
        exclusions = {}
        for split in ("train", "val"):
            if sum(row["count"] for row in coverage if row["split"] == split) != totals[split]:
                raise ValueError(f"Rotation join changed the {split} cohort.")
        for status in ("unresolved", "after_bound"):
            exclusions[status] = sum(row["count"] for row in coverage if row["training_label_status"] == status)
        if sum(exclusions.values()) == totals.train:
            raise ValueError("No training labels meet the rotation experiment's availability guard.")
        linked.select("flight_id", "FlightDate", "split", *AUDIT, "training_label_status").write.parquet(
            str(run_dir / "rotation_links")
        )
        numeric, flags = bts_train.weather_columns(FEATURE_SET)
        splits = linked.select(
            *IDENTITY, "split", "training_label_status", *bts_train.INPUT_FEATURES, *numeric, *flags,
            *ROTATION_NUMERIC, *ROTATION_FLAGS,
        )
        for split, status in (("train", "eligible"), ("val", "validation")):
            splits.where(F.col("training_label_status") == status).drop("training_label_status").write.parquet(
                str(run_dir / "gold" / split)
            )
        return {"coverage": coverage, "training_exclusions": exclusions}
    finally:
        if linked is not None:
            linked.unpersist(blocking=True)
        operations.unpersist(blocking=True)


def validation_predictions(
    spark: SparkSession,
    run_dir: Path,
    arm: str,
    expected: DataFrame,
) -> DataFrame:
    """Read and validate scored validation predictions for a comparison arm."""
    predictions = spark.read.parquet(str(run_dir / arm / "validation_predictions"))
    p = F.col("p_delayed")
    actual = predictions.select(
        *IDENTITY, F.coalesce(p.between(0, 1), F.lit(False)).alias("valid_probability"),
        F.lit(1).alias("delta"),
    )
    reference = expected.select(*IDENTITY, F.lit(True).alias("valid_probability"), F.lit(-1).alias("delta"))
    mismatches = actual.unionByName(reference).groupBy(*IDENTITY, "valid_probability").agg(
        F.sum("delta").alias("balance")
    ).where("balance != 0")
    if mismatches.limit(1).count():
        raise ValueError(f"{arm} predictions have invalid probabilities or differ from the validation cohort.")
    return predictions


def run_comparison(spark: SparkSession, run_dir: Path, gold: Optional[Path] = None) -> dict[str, Any]:
    """Train and evaluate W2 (weather baseline) and H1 (weather + rotation) models on validation."""
    gold = gold or run_dir / "gold"
    timings, arms = {}, {}
    for arm, variant in (("w2", FEATURE_SET), ("h1", bts_train.ROTATION_FEATURE_SET)):
        with stage(f"Phase 3 fit and validate {arm.upper()}", timings=timings, key=arm):
            arms[arm] = bts_train.train(
                spark, gold, run_dir / arm, feature_set=variant,
                save_predictions=True, publish_results=False,
            )
    with stage("Phase 3 verify prediction cohorts and compare scores", timings=timings, key="compare"):
        validation = spark.read.parquet(str(gold / "val"))
        expected = validation.select(*IDENTITY)
        covered = validation.where(F.col(ROTATION_FLAGS[1]) == 0).select("flight_id")
        covered_rows = covered.count()
        for arm, summary in arms.items():
            predictions = validation_predictions(spark, run_dir, arm, expected)
            summary["completed_predecessor_validation"] = (
                bts_train.evaluate_predictions(predictions.join(covered, "flight_id")) if covered_rows else None
            )
        w2, h1 = [
            arms[arm]["selected_model"]["validation_metrics_at_selected_threshold"] for arm in ("w2", "h1")
        ]
        deltas = {
            key: h1[key] - w2[key] if h1[key] is not None and w2[key] is not None else None for key in METRICS
        }
    return {
        "arms": arms, "h1_minus_w2": deltas, "completed_predecessor_validation_rows": covered_rows,
        "timing_seconds": timings,
    }


def publish_run(run_dir: Path, manifest: dict[str, Any], current_inputs: Any) -> None:
    """Verify input stability, write run.json manifest, and mark run as successful."""
    if current_inputs() != manifest["inputs"]:
        raise ValueError("Rotation inputs changed during the run.")
    write_json(run_dir / "run.json", manifest)
    (run_dir / "_SUCCESS").touch(exist_ok=False)


def prepare_rotation(
    spark: SparkSession,
    config: dict[str, Any],
    references: Path,
    parent: Path,
    raw_files: list[Path],
    run_dir: Path,
) -> dict[str, Any]:
    """Prepare rotation operations, compute predecessor links, and stage comparison Gold data."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    first_reliable_date = date.fromisoformat(config["datasets"]["bts"]["start_month"] + "-01") + timedelta(days=2)
    validation_bound = datetime.fromisoformat(config["splits"]["val"]["start_date"] + "T00:00:00+14:00")
    validation_bound = validation_bound.astimezone(timezone.utc)
    numeric, flags = bts_train.weather_columns(FEATURE_SET)
    projection = [*IDENTITY, "split", "scheduled_departure_utc", *bts_train.INPUT_FEATURES, *numeric, *flags]

    def inputs():
        # Rotation only reads development splits. A new destination test
        # partition can change en-route schedules without changing these links.
        return {"destination": consumed(parent, ("train", "val")), "raw": file_fingerprints(raw_files),
                "projection": projection, "first_reliable": first_reliable_date.isoformat(),
                "validation_bound": validation_bound.isoformat(),
                "linking": logic_hash(build, prepare_experiment, local_cutoff),
                "audit": AUDIT, "identity": IDENTITY,
                "code": {name: sha256(ROOT / "scripts" / name) for name in
                         ("aircraft_rotation/build_features.py", "bts/recent_delays.py")},
                "reference": {name: sha256(references / name) for name in ("airports.csv", "sources.json")}}

    def build(directory):
        raw = spark.read.option("header", True).option("enforceSchema", False).csv([str(path) for path in raw_files])
        operations = build_operations(raw, load_airports(references))
        weather = spark.read.parquet(str(parent / "train"), str(parent / "val")).select(*projection)
        report = prepare_experiment(weather, operations, directory, first_reliable_date, validation_bound)
        write_json(directory / "coverage.json", report)
        return {"destination": str(parent)}

    timings = {}
    with stage("Phase 3 link aircraft and prepare shared splits", timings=timings, key="prepare"):
        directory = resolve_stage(ROOT / "data/gold/rotation", inputs, ("gold/train", "gold/val", "rotation_links"),
                                  ("coverage.json",), build)
        preparation = json.loads((directory / "coverage.json").read_text())
    return {
        "scope": "Conditional retrospective aircraft rotation using final BTS tail assignments. Validation only.",
        "max_gap_hours": MAX_GAP_HOURS, "assumed_outcome_lag_minutes": OUTCOME_LAG_MINUTES,
        "first_reliable_service_date": first_reliable_date.isoformat(),
        "validation_lower_bound_utc": validation_bound.isoformat(),
        "training_gold": str(directory / "gold"), "timing_seconds": timings, **preparation,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Run W2 vs H1 aircraft rotation comparison experiment and record results."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    root = ROOT
    config, references, parent, raw_files = resolve_inputs(root)
    inputs = source_identity(root, references, parent, raw_files)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    run_dir = project_path(root, f"models/aircraft_rotation_comparison/{run_id}")
    run_dir.mkdir(parents=True, exist_ok=False)
    spark = get_spark_session("aircraft_rotation_comparison")
    try:
        preparation = prepare_rotation(spark, config, references, parent, raw_files, run_dir)
        comparison = run_comparison(spark, run_dir, Path(preparation["training_gold"]))
        comparison["timing_seconds"].update(preparation["timing_seconds"])
        manifest = {
            "run_id": run_id, "parent": str(parent), "splits": config["splits"], "inputs": inputs,
            **preparation, **comparison,
        }
        with stage("Phase 3 publish completed comparison"):
            publish_run(run_dir, manifest, lambda: source_identity(root, references, parent, raw_files))
        print(f"Aircraft rotation comparison saved to {run_dir / 'run.json'}", flush=True)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
