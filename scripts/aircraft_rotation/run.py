"""Prepare aircraft rotation features for the flight-delay pipeline."""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, functions as F

from scripts.bts.build_features import BTS_FEATURE_COLUMNS
from scripts.bts.download import build_download_url, iter_months
from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.io import file_fingerprints, sha256, write_json
from scripts.common.progress import pipeline_stage, stage
from scripts.common.references import load_airports
from scripts.common.stage_cache import consumed, logic_hash, resolve_stage
from scripts.common.time_utils import local_cutoff
from scripts.destination_weather.build_features import DESTINATION_WEATHER_FLAGS, DESTINATION_WEATHER_NUMERIC
from scripts.origin_weather.build_features import WEATHER_FLAGS, WEATHER_NUMERIC

from .build_features import (
    MAX_GAP_HOURS, RECOVERY_FEATURES, ROTATION_FLAGS, ROTATION_NUMERIC,
    add_recovery_shortfall, add_turnaround_shortfall, build_operations, build_rotation_links,
)

SPLITS = ("train", "val", "test")
IDENTITY = ["flight_id", "FlightDate", "ArrDel15"]
SURFACE_WEATHER_COLUMNS = [*WEATHER_NUMERIC, *DESTINATION_WEATHER_NUMERIC, *WEATHER_FLAGS, *DESTINATION_WEATHER_FLAGS]
AUDIT = [
    "scheduled_departure_utc", "rotation_reason", "prior_flight_id", "prior_scheduled_departure_utc",
    "prior_scheduled_arrival_utc", "prior_actual_arrival_utc", "prior_arrival_available_utc",
    "prior_departure_gap_minutes",
]


def rotation_raw_files(root: Path, config: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """Resolve the original development and extended scoring archive ranges."""
    bts = config["datasets"]["bts"]
    bronze = Path(bts["output_directory"])
    validation_month = config["splits"]["val"]["end_date"][:7]
    test_month = config["splits"]["test"]["end_date"][:7]
    if not bts["start_month"] <= validation_month <= test_month <= bts["end_month"]:
        raise ValueError("The configured BTS archive must cover validation and test.")
    development, extended = [], []
    for year, month in iter_months(bts["start_month"], test_month):
        archive = build_download_url(year, month).rsplit("/", 1)[-1]
        period = f"{year}-{month:02d}"
        path = project_path(root, bronze / str(year) / f"{month:02d}" / Path(archive).with_suffix(".csv"))
        if not path.is_file():
            raise ValueError(f"Missing BTS rotation input: {path}")
        extended.append(path)
        if period <= validation_month:
            development.append(path)
    return development, extended


def build_rotation_gold(
    parent: DataFrame,
    development_operations: DataFrame,
    test_operations: DataFrame,
    directory: Path,
    first_reliable_date: date,
    validation_bound: datetime,
) -> dict[str, Any]:
    """Link parent cohort to operations, filter training availability, and write isolated Gold splits."""
    totals = parent.agg(
        F.count("*").alias("rows"), F.countDistinct("flight_id").alias("unique_ids"),
        F.count(F.when(F.col("split") == "train", 1)).alias("train"),
        F.count(F.when(F.col("split") == "val", 1)).alias("val"),
        F.count(F.when(F.col("split") == "test", 1)).alias("test"),
    ).first()
    if (totals.rows != totals.unique_ids or totals.rows != sum(totals[split] for split in SPLITS)
            or any(not totals[split] for split in SPLITS)):
        raise ValueError("Rotation parent requires distinct nonnull IDs and nonempty train, validation, and test splits.")
    development_operations.persist(StorageLevel.DISK_ONLY)
    test_operations.persist(StorageLevel.DISK_ONLY)
    linked = None
    try:
        if test_operations.where("tail is not null and FlightDate is null").limit(1).count():
            raise ValueError("A known-tail operation lacks a service date for bounding its possible history.")
        bound = F.to_timestamp(F.lit(validation_bound.isoformat()))
        with stage("Link flights to prior same-aircraft legs and check label cutoffs"):
            # Keep retrospective rows from later months out of development linkage.
            development = build_rotation_links(parent.where("split != 'test'"), development_operations, first_reliable_date)
            test = build_rotation_links(parent.where("split = 'test'"), test_operations, first_reliable_date)
            linked = (
                development.unionByName(test)
                .withColumn(
                    "training_label_status",
                    F.when(F.col("split") != "train", "scoring")
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
            for split in SPLITS:
                if sum(row["count"] for row in coverage if row["split"] == split) != totals[split]:
                    raise ValueError(f"Rotation join changed the {split} cohort.")
            for status in ("unresolved", "after_bound"):
                exclusions[status] = sum(row["count"] for row in coverage if row["training_label_status"] == status)
            if sum(exclusions.values()) == totals.train:
                raise ValueError("No training labels meet the rotation availability guard.")
            linked.select("flight_id", "FlightDate", "split", *AUDIT, "training_label_status").write.parquet(
                str(directory / "rotation_links")
            )
        with stage("Compute turnaround shortfall and net turnaround shortfall, then write rotation Gold splits"):
            splits = add_recovery_shortfall(add_turnaround_shortfall(linked, validation_bound)).select(
                *IDENTITY, "split", "training_label_status", *BTS_FEATURE_COLUMNS, *SURFACE_WEATHER_COLUMNS,
                *ROTATION_NUMERIC, *ROTATION_FLAGS, *RECOVERY_FEATURES,
            )
            # The label guard changes only training. Scoring cohorts must match exactly.
            identity = [*IDENTITY, "split"]
            before = parent.where("split != 'train'").select(*identity)
            after = splits.where("split != 'train'").select(*identity)
            after_stats = after.agg(
                F.count("*").alias("rows"), F.countDistinct("flight_id").alias("ids")
            ).first()
            if after_stats.rows != after_stats.ids:
                raise ValueError("Rotation changed validation or test flight identity.")
            before_rows = before.select("flight_id", F.struct(*identity).alias("_before"))
            after_rows = after.select("flight_id", F.struct(*identity).alias("_after"))
            if (
                before_rows.hint("merge")
                .join(after_rows, "flight_id", "full")
                .filter(~F.col("_before").eqNullSafe(F.col("_after")))
                .limit(1)
                .count()
            ):
                raise ValueError("Rotation changed validation or test flight identity.")
            recovery_coverage = {}
            for split, status in (("train", "eligible"), ("val", "scoring"), ("test", "scoring")):
                path = directory / "gold" / split
                (splits.where((F.col("split") == split) & (F.col("training_label_status") == status))
                 .drop("training_label_status").write.parquet(str(path)))
                recovery_coverage[split] = parent.sparkSession.read.parquet(str(path)).agg(
                    F.count("*").alias("rows"), F.count(RECOVERY_FEATURES[0]).alias("observed"),
                ).first().asDict()
        return {"coverage": coverage, "training_exclusions": exclusions, "recovery_coverage": recovery_coverage}
    finally:
        if linked is not None:
            linked.unpersist(blocking=True)
        development_operations.unpersist(blocking=True)
        test_operations.unpersist(blocking=True)


def prepare_rotation(
    spark: SparkSession,
    root: Path,
    config: dict[str, Any],
    references: Path,
    parent: Path,
    raw_files: tuple[list[Path], list[Path]],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Build or reuse rotation Gold and return its source fingerprints."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    first_reliable_date = date.fromisoformat(config["datasets"]["bts"]["start_month"] + "-01") + timedelta(days=2)
    validation_bound = datetime.fromisoformat(config["splits"]["val"]["start_date"] + "T00:00:00+14:00")
    validation_bound = validation_bound.astimezone(timezone.utc)
    projection = [*IDENTITY, "split", "scheduled_departure_utc", *BTS_FEATURE_COLUMNS, *SURFACE_WEATHER_COLUMNS]
    development, extended = raw_files
    references_used = [references / name for name in ("airports.csv", "sources.json")]

    def inputs():
        return {
            "destination": consumed(parent, SPLITS),
            "raw": file_fingerprints(extended),
            "development_raw": [path.relative_to(root).as_posix() for path in development],
            "projection": projection, "first_reliable": first_reliable_date.isoformat(),
            "validation_bound": validation_bound.isoformat(),
            "linking": logic_hash(build, build_rotation_gold, local_cutoff),
            "audit": AUDIT, "identity": IDENTITY,
            "code": {name: sha256(root / "scripts" / name) for name in
                     ("aircraft_rotation/build_features.py", "bts/recent_delays.py")},
            "reference": {path.relative_to(root).as_posix(): sha256(path) for path in references_used},
        }

    def build(directory):
        airports = load_airports(references)
        development_names = sorted({path.name for path in development})
        # Read all 13 extended CSVs once and isolate the 11-month development subset by filename.
        extended_tagged = (
            spark.read.option("header", True)
            .csv([str(path) for path in extended])
            .withColumn("_source_name", F.element_at(F.split(F.input_file_name(), "/"), -1))
            .persist(StorageLevel.DISK_ONLY)
        )
        try:
            development_raw = extended_tagged.filter(F.col("_source_name").isin(development_names)).drop("_source_name")
            extended_raw = extended_tagged.drop("_source_name")
            weather = spark.read.parquet(*(str(parent / split) for split in SPLITS)).select(*projection)
            report = build_rotation_gold(
                weather, build_operations(development_raw, airports), build_operations(extended_raw, airports),
                directory, first_reliable_date, validation_bound,
            )
            write_json(directory / "coverage.json", report)
            return {"destination": str(parent)}
        finally:
            extended_tagged.unpersist(blocking=True)

    partitions = (*(f"gold/{split}" for split in SPLITS), "rotation_links")
    with stage("Resolve or build rotation Gold"):
        directory = resolve_stage(root / "data/gold/rotation", inputs, partitions, ("coverage.json",), build)
    manifest = json.loads((directory / "_stage.json").read_text(encoding="utf-8"))
    preparation = json.loads((directory / "coverage.json").read_text(encoding="utf-8"))
    return directory / "gold", {
        "scope": "Retrospective aircraft rotation using final BTS tail assignments.",
        "max_gap_hours": MAX_GAP_HOURS,
        "first_reliable_service_date": first_reliable_date.isoformat(),
        "validation_lower_bound_utc": validation_bound.isoformat(),
        **preparation,
    }, manifest["contract"]["inputs"]


def run(ctx: PipelineContext) -> None:
    """Stage 4 entry point: link aircraft rotations and update `ctx`."""
    with pipeline_stage(4, 6, "Aircraft Rotation & Turnaround", timings=ctx.timings):
        raw_files = rotation_raw_files(ctx.root, ctx.config)
        parent = ctx.destination_gold_dir or ctx.gold_dir
        rotation_gold, rotation_coverage, rotation_inputs = prepare_rotation(
            ctx.spark,
            ctx.root,
            ctx.config,
            ctx.settings["reference_directory"],
            parent,
            raw_files,
        )
        ctx.rotation_gold_dir = rotation_gold
        ctx.gold_dir = rotation_gold
        ctx.coverage["rotation"] = rotation_coverage
        ctx.sources["rotation"] = rotation_inputs
        ctx.sources["sha256"].update(rotation_inputs["reference"])
