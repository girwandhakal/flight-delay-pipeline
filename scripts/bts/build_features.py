"""Build schedule-based features and assign chronological splits for BTS modeling."""

from datetime import date, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

from scripts.common.config import ROOT, project_path
from scripts.common.io import sha256, write_json
from scripts.common.progress import stage
from scripts.common.references import load_airports
from scripts.common.results import records, update_results
from scripts.common.spark import configure_spark_env, get_spark_session

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from .block_time import (
    BLOCK_TIME_SIDECAR, POLICY as BLOCK_TIME_POLICY, SLACK_FEATURES, add_outbound_schedule_slack,
    read_block_history, resolve_block_history,
)
from .download import iter_months
from .recent_delays import (
    FEATURES as RECENT_CALENDAR_FEATURES,
    TAXI_BASELINE_FALLBACK_MINUTES, TAXI_BASELINE_FILE, TAXI_MEAN, TAXI_SURGE,
    add_cutoffs, add_holidays, add_recent_delays, build_events, compute_unimpeded_taxi_baselines,
    feature_manifest, feature_months,
)

CATEGORICAL_FEATURES = ["Reporting_Airline", "Origin", "Dest", "DayOfWeek", "DepHour", "ArrHour"]
NUMERIC_FEATURES = ["CRSElapsedTime", "Distance"]
BTS_FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURES[:3], "Month", *CATEGORICAL_FEATURES[3:],
    *NUMERIC_FEATURES, *RECENT_CALENDAR_FEATURES, *SLACK_FEATURES,
]


def build_features(
    spark: SparkSession,
    silver_dir: Optional[Union[str, Path]] = None,
    gold_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    publish_results: bool = True,
    bronze_dir: Optional[Union[str, Path]] = None,
    *,
    block_history: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Construct schedule, rolling delay, taxi-out and holiday features for chronological splits.

    Generates:
    - Month, DayOfWeek (Monday=1..Sunday=7), DepHour (0-23), ArrHour (0-23).
    - 2-hour rolling delay features at scheduled departure cutoff for origin departure and destination arrival.
    - Federal holiday indicators (day-of, day-before, day-after).
    - Fixed chronological split assignment (train, val, test).

    Args:
        spark: Active SparkSession.
        silver_dir: Directory containing cleaned BTS Silver Parquet partitions.
        gold_dir: Output directory for BTS Gold Parquet splits.
        results_dir: Results directory for results.json.
        config_path: Path to config.json.
        publish_results: Whether to record split counts in results.json.
        bronze_dir: Directory containing raw BTS CSVs for event extraction.

    Returns:
        Pandas DataFrame summarizing split counts, date ranges, and delay rates.
    """
    if all(path is None for path in (bronze_dir, silver_dir, results_dir, gold_dir, config_path)):
        from .cache import prepare
        return prepare(spark, ROOT, "gold", publish_results)
    silver_dir = Path(silver_dir or ROOT / "data" / "silver" / "bts")
    gold_dir = Path(gold_dir or ROOT / "data" / "gold" / "bts")
    results_dir = Path(results_dir or ROOT / "results")

    # Read config once for both splits and dataset settings
    config_file = Path(config_path or ROOT / "config.json")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    splits = config["splits"]
    manifest = feature_manifest(ROOT, config, bronze_dir, silver_dir, block_history)
    block_history = manifest["block_history"]

    bronze = Path(bronze_dir or project_path(ROOT, config["datasets"]["bts"]["output_directory"]))
    csv_files = sorted(bronze.glob("*/*/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No BTS CSV files found in {bronze}")
    settings = config["datasets"]["bts"]
    core_months = feature_months(settings)
    source_months = {path: date(int(path.parent.parent.name), int(path.parent.name), 1) for path in csv_files}
    core_files = [str(path) for path in csv_files if source_months[path] in core_months]
    if not core_files:
        raise FileNotFoundError(f"No configured BTS month CSVs found in {bronze}")
    airports = load_airports(project_path(ROOT, config["datasets"]["weather"]["reference_directory"]))

    silver_months = [silver_dir / f"{month.year:04d}" / f"{month.month:02d}" for month in sorted(core_months)]
    silver_paths = [str(path) for path in silver_months if path.is_dir()]
    if not silver_paths:
        raise FileNotFoundError(f"No configured BTS Silver partitions found in {silver_dir}")
    silver_df = spark.read.parquet(*silver_paths)
    input_count = silver_df.count()
    if input_count == 0:
        raise ValueError(f"No records found in {silver_dir}")

    # Map Spark Sunday=1..Saturday=7 to BTS standard Monday=1..Sunday=7
    bts_day_of_week = (F.dayofweek(F.col("FlightDate")) + 5) % 7 + 1

    # Extract 0-23 hour from HHMM string; 2400 represents midnight (hour 0)
    dep_hour = (F.col("CRSDepTime").cast("int") / 100).cast("int") % 24
    arr_hour = (F.col("CRSArrTime").cast("int") / 100).cast("int") % 24

    # Build chronological split condition using inclusive date boundaries
    split_col = (
        F.when(
            (F.col("FlightDate") >= splits["train"]["start_date"])
            & (F.col("FlightDate") <= splits["train"]["end_date"]),
            "train",
        )
        .when(
            (F.col("FlightDate") >= splits["val"]["start_date"])
            & (F.col("FlightDate") <= splits["val"]["end_date"]),
            "val",
        )
        .when(
            (F.col("FlightDate") >= splits["test"]["start_date"])
            & (F.col("FlightDate") <= splits["test"]["end_date"]),
            "test",
        )
        .otherwise(None)
    )

    feature_df = silver_df.select(
        F.col("flight_id"),
        F.col("FlightDate"),
        F.col("Flight_Number_Reporting_Airline"),
        F.col("CRSDepTime"),
        F.col("CRSArrTime"),
        F.col("Reporting_Airline"),
        F.col("Origin"),
        F.col("Dest"),
        F.month(F.col("FlightDate")).alias("Month"),
        bts_day_of_week.alias("DayOfWeek"),
        dep_hour.alias("DepHour"),
        arr_hour.alias("ArrHour"),
        F.col("CRSElapsedTime").cast("double").alias("CRSElapsedTime"),
        F.col("Distance").cast("double").alias("Distance"),
        F.col("ArrDel15").cast("int").alias("ArrDel15"),
        split_col.alias("split"),
    )

    # Ensure every retained flight receives an assigned split
    unassigned = feature_df.filter(F.col("split").isNull()).count()
    if unassigned > 0:
        raise ValueError(
            f"Found {unassigned} flights outside configured split boundaries"
        )

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    raw = spark.read.csv(core_files, header=True)
    events = None
    history = None
    try:
        with stage("Extract completed flight events and taxi-out baselines"):
            events = build_events(raw, airports).persist(StorageLevel.DISK_ONLY)
            baselines = compute_unimpeded_taxi_baselines(events, feature_df, splits, airports)
            # Materialize the small lookup once for both the join and the published reference.
            baseline_rows = baselines.orderBy("Origin", "taxi_snapshot").collect()
            baseline_lookup = spark.createDataFrame(baseline_rows, baselines.schema)
            identity = ["flight_id", "FlightDate", "ArrDel15", "split"]
            expected = feature_df.select(*identity)
            cutoffs = add_cutoffs(feature_df, airports)
            history_months = {
                f"{year:04d}-{month:02d}" for year, month in iter_months(
                    splits["train"]["start_date"][:7], splits["train"]["end_date"][:7],
                )
            }
            releases = [source for source in block_history["months"] if source["source_month"] in history_months]
            source_coverage = []
            if releases:
                print(
                    f"  Building schedule buffer baseline from {len(releases)} historical training months "
                    f"(source: {block_history['status']}).",
                    flush=True,
                )
                history, rejected = read_block_history(spark, ROOT, releases, airports, splits["train"])
                history = history.persist(StorageLevel.DISK_ONLY)
                counts = history.groupBy("_source_month", "reason").count().unionByName(rejected)
                source_coverage = [row.asDict() for row in counts.orderBy("_source_month", "reason").collect()]
            else:
                print(
                    "  No released BTS training history found; schedule buffer features will be marked missing.",
                    flush=True,
                )

        with stage("Compute schedule buffer, 2h rolling airport delays, taxi surge, and holidays"):
            feature_df = add_outbound_schedule_slack(cutoffs, history, splits, releases)
            feature_df = add_recent_delays(feature_df, events, manifest["rolling_window_minutes"])
            feature_df = (feature_df.withColumn(
                "taxi_snapshot",
                F.when(F.col("split") == "train", F.date_format("FlightDate", "yyyy-MM")).otherwise("evaluation"),
            ).join(F.broadcast(baseline_lookup), ["Origin", "taxi_snapshot"], "left")
              .withColumn(TAXI_SURGE, F.col(TAXI_MEAN) - F.coalesce("unimpeded_taxi_minutes",
                                                                   F.lit(TAXI_BASELINE_FALLBACK_MINUTES)))
              .drop("taxi_snapshot", "unimpeded_taxi_minutes"))
            feature_df = add_holidays(feature_df, splits["train"]["start_date"], splits["test"]["end_date"])
            feature_df = feature_df.persist(StorageLevel.DISK_ONLY)

            actual = feature_df.select(*identity)
            for frame in (expected, actual):
                id_stats = frame.agg(
                    F.count("*").alias("rows"),
                    F.countDistinct("flight_id").alias("ids"),
                ).first()
                if id_stats.rows != id_stats.ids:
                    raise ValueError("BTS enrichment changed flight identity, label or split.")
            expected_rows = expected.select("flight_id", F.struct(*identity).alias("_expected"))
            actual_rows = actual.select("flight_id", F.struct(*identity).alias("_actual"))
            if (
                expected_rows.hint("merge")
                .join(actual_rows, "flight_id", "full")
                .filter(~F.col("_expected").eqNullSafe(F.col("_actual")))
                .limit(1)
                .count()
            ):
                raise ValueError("BTS enrichment changed flight identity, label or split.")
        coverage = [
            row.asDict() for row in feature_df.groupBy(
                "split", F.date_format("_snapshot", "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("snapshot"), "_block_reason",
            ).agg(
                F.count("*").alias("count"),
                F.countDistinct(F.when(F.col("_block_samples").isNotNull(), F.struct(
                    F.when(F.col("_block_reason") == "carrier_route", F.col("Reporting_Airline")), "Origin", "Dest",
                ))).alias("reference_keys"),
                F.min("_block_samples").alias("min_reference_samples"),
                F.max("_block_samples").alias("max_reference_samples"),
            ).orderBy("split", "snapshot", "_block_reason").collect()
        ]
        snapshots = sorted({row["snapshot"] for row in coverage if row["snapshot"] is not None})
        usable_months = {row["_source_month"] for row in source_coverage if row["reason"] == "usable"}
        available = sorted(datetime.fromisoformat(source["available_utc"]) for source in releases
                           if source["source_month"] in usable_months)
        bounds = {}
        for snapshot in snapshots:
            latest = next((instant for instant in reversed(available) if instant < datetime.fromisoformat(snapshot)), None)
            bounds[snapshot] = latest.isoformat().replace("+00:00", "Z") if latest else None
        (gold_dir / "_features.json").unlink(missing_ok=True)
        summary_records = []
        with stage("Write BTS Gold train/val/test Parquet splits"):
            # Compute all three split summaries in a single groupBy pass over feature_df.
            aggregates = [
                F.count("*").alias("total_flights"),
                F.min("FlightDate").alias("min_date"),
                F.max("FlightDate").alias("max_date"),
            ]
            if publish_results:
                aggregates.append(F.count(F.when(F.col("ArrDel15") == 1, 1)).alias("delayed_flights"))
            stats_by_split = {
                row["split"]: row
                for row in feature_df.groupBy("split").agg(*aggregates).collect()
            }

            # Save partitioned Parquet datasets directly for each chronological split
            for split_name in ["train", "val", "test"]:
                split_df = feature_df.filter(F.col("split") == split_name)
                dest_split_dir = gold_dir / split_name
                dest_split_dir.mkdir(parents=True, exist_ok=True)
                split_df.drop("_snapshot", "_block_reason", "_block_samples").write.mode("overwrite").parquet(str(dest_split_dir))

                stats = stats_by_split.get(split_name)
                total_flights = stats["total_flights"] if stats else 0
                record = {
                    "split": split_name,
                    "start_date": str(stats["min_date"]) if stats else "None",
                    "end_date": str(stats["max_date"]) if stats else "None",
                    "total_flights": total_flights,
                }
                if publish_results:
                    delayed_flights = stats["delayed_flights"] if stats else 0
                    record.update({
                        "on_time_flights": total_flights - delayed_flights,
                        "delayed_flights": delayed_flights,
                        "delay_rate": round(delayed_flights / total_flights, 4) if total_flights else 0.0,
                    })
                summary_records.append(record)

        if sum(record["total_flights"] for record in summary_records) != input_count:
            raise ValueError("BTS features changed the Silver cohort size")
        baseline_path = gold_dir / TAXI_BASELINE_FILE
        write_json(baseline_path, [row.asDict() for row in baseline_rows])
        block_path = gold_dir / BLOCK_TIME_SIDECAR
        write_json(block_path, {"policy": BLOCK_TIME_POLICY, "history": block_history,
                                "missing_source_months": sorted(history_months - {s["source_month"] for s in releases}),
                                "sources": source_coverage, "snapshot_release_upper_bounds": bounds,
                                "coverage": coverage})
        write_json(gold_dir / "_features.json", {
            **manifest, "taxi_baselines_sha256": sha256(baseline_path),
            "block_time_sha256": sha256(block_path),
        })
        summary_pd = pd.DataFrame(summary_records)
        if publish_results:
            update_results({"bts": {"splits": records(summary_pd)}}, results_dir)

        return summary_pd
    finally:
        feature_df.unpersist(blocking=True)
        if history is not None:
            history.unpersist(blocking=True)
        if events is not None:
            events.unpersist(blocking=True)


def main(publish_results: bool = True) -> int:
    """Execute BTS feature engineering and chronological splitting workflow."""
    spark = get_spark_session()
    try:
        summary = build_features(spark, publish_results=publish_results)
        total_flights = summary["total_flights"].sum()
        print("-" * 80)
        print(
            f"Chronological Splitting Summary (Total Processed Flights: {total_flights:,}):"
        )
        print("-" * 80)
        print(summary.to_string(index=False))
        print("-" * 80)
        return 0
    finally:
        spark.stop()



if __name__ == "__main__":
    main()
