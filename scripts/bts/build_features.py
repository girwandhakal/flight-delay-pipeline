"""Build schedule-based features and assign chronological splits for BTS modeling."""

from datetime import date
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

from scripts.results import records, update_results

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

from .recent_delays import (
    TAXI_BASELINE_FALLBACK_MINUTES, TAXI_BASELINE_FILE, TAXI_MEAN, TAXI_SURGE, UPSTREAM_ROLES,
    add_cutoffs, add_holidays, add_recent_delays, build_events, compute_unimpeded_taxi_baselines,
    feature_manifest, feature_months, load_airports,
)
from .upstream import add_upstream_pressure, build_inbound_schedule, uncovered_month_edges
from scripts.origin_weather.download import project_path, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]


def configure_spark_env() -> None:
    """Configure Hadoop binaries on Windows and ensure PySpark workers use the active interpreter."""
    if os.name == "nt" and "HADOOP_HOME" not in os.environ:
        hadoop_dir = ROOT / "envs" / "hadoop"
        if (hadoop_dir / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(hadoop_dir)
            os.environ["PATH"] = (
                str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")
            )
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# Apply runtime environment configuration on module load
configure_spark_env()


def get_spark_session(app_name: str = "bts_features") -> SparkSession:
    """Create or retrieve a tuned local Spark session bounded for laptop execution.

    Configures loopback networking, local[2] parallelism, 6GB driver memory,
    64 shuffle partitions, and adaptive execution parameters.
    """
    python_path = os.environ.get("PYTHONPATH", "")
    if str(ROOT) not in python_path.split(os.pathsep):
        # Workers need the project package on PYTHONPATH when driver starts elsewhere
        os.environ["PYTHONPATH"] = str(ROOT) + (os.pathsep + python_path if python_path else "")
    return (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.files.maxPartitionBytes", "32m")
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "16m")
        .config("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
        .getOrCreate()
    )


def load_split_config(config_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load chronological split boundaries from config.json."""
    config_file = Path(config_path or ROOT / "config.json")
    with config_file.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["splits"]


def build_features(
    spark: SparkSession,
    silver_dir: Optional[Union[str, Path]] = None,
    gold_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    publish_results: bool = True,
    bronze_dir: Optional[Union[str, Path]] = None,
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
    manifest = feature_manifest(ROOT, config, bronze_dir, silver_dir)

    bronze = Path(bronze_dir or project_path(ROOT, config["datasets"]["bts"]["output_directory"]))
    csv_files = sorted(bronze.glob("*/*/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No BTS CSV files found in {bronze}")
    settings = config["datasets"]["bts"]
    core_months, adjacent_months = feature_months(settings)
    source_months = {path: date(int(path.parent.parent.name), int(path.parent.name), 1) for path in csv_files}
    core_files = [str(path) for path in csv_files if source_months[path] in core_months]
    adjacent_files = [str(path) for path in csv_files if source_months[path] in adjacent_months]
    if not core_files:
        raise FileNotFoundError(f"No configured BTS month CSVs found in {bronze}")
    edges = uncovered_month_edges(core_months, set(source_months.values()))
    airports = load_airports(project_path(ROOT, config["datasets"]["weather"]["reference_directory"]))

    # Adjacent Bronze months can populate schedule windows but cannot enter the model cohort.
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
    adjacent = spark.read.csv(adjacent_files, header=True) if adjacent_files else None
    upstream_raw = raw.unionByName(adjacent) if adjacent is not None else raw
    events = build_events(raw, airports).persist(StorageLevel.DISK_ONLY)
    schedule = None
    try:
        baselines = compute_unimpeded_taxi_baselines(events, feature_df, splits, airports)
        # Materialize the small lookup once for both the join and the published reference.
        baseline_rows = baselines.orderBy("Origin", "taxi_snapshot").collect()
        baseline_lookup = spark.createDataFrame(baseline_rows, baselines.schema)
        feature_df = add_recent_delays(
            add_cutoffs(feature_df, airports),
            events,
            manifest["rolling_window_minutes"],
        )
        feature_df = (feature_df.withColumn(
            "taxi_snapshot",
            F.when(F.col("split") == "train", F.date_format("FlightDate", "yyyy-MM")).otherwise("evaluation"),
        ).join(F.broadcast(baseline_lookup), ["Origin", "taxi_snapshot"], "left")
          .withColumn(TAXI_SURGE, F.col(TAXI_MEAN) - F.coalesce("unimpeded_taxi_minutes",
                                                               F.lit(TAXI_BASELINE_FALLBACK_MINUTES)))
          .drop("taxi_snapshot", "unimpeded_taxi_minutes"))
        schedule = build_inbound_schedule(upstream_raw, airports).persist(StorageLevel.DISK_ONLY)
        schedule_coverage = {row.status: row["count"] for row in schedule.groupBy("status").count().collect()}
        schedule_rejections = [
            row.asDict() for row in schedule.where("status != 'usable'")
            .groupBy("source_month", "Dest", "status").count().orderBy("source_month", "Dest", "status").collect()
        ]
        upstream_events = build_events(upstream_raw, airports) if adjacent is not None else events
        feature_df = add_upstream_pressure(feature_df, schedule, upstream_events, airports, edges)
        feature_df = add_holidays(feature_df, splits["train"]["start_date"], splits["test"]["end_date"])
        feature_df = feature_df.persist(StorageLevel.DISK_ONLY)

        (gold_dir / "_features.json").unlink(missing_ok=True)
        summary_records = []
        upstream_coverage = {
            "schedule": schedule_coverage, "schedule_rejections": schedule_rejections,
            "uncovered_edges_utc_seconds": edges, "splits": {},
        }
        # Save partitioned Parquet datasets directly for each chronological split
        for split_name in ["train", "val", "test"]:
            split_df = feature_df.filter(F.col("split") == split_name)
            dest_split_dir = gold_dir / split_name
            dest_split_dir.mkdir(parents=True, exist_ok=True)
            split_df.write.mode("overwrite").parquet(str(dest_split_dir))

            # Calculate split summary in one pass
            aggregates = [
                F.count("*").alias("total_flights"),
                F.min("FlightDate").alias("min_date"),
                F.max("FlightDate").alias("max_date"),
            ]
            if publish_results:
                aggregates.append(F.count(F.when(F.col("ArrDel15") == 1, 1)).alias("delayed_flights"))

            for role in UPSTREAM_ROLES:
                aggregates.extend([
                    F.count(f"{role}_upstream_scheduled_arrivals").alias(f"{role}_window_count"),
                    F.count(f"{role}_upstream_delay_pressure").alias(f"{role}_pressure_count"),
                    F.avg(f"{role}_upstream_observed_share").alias(f"{role}_mean_observed_share"),
                ])
            stats = split_df.agg(*aggregates).first()
            upstream_coverage["splits"][split_name] = {
                key: value for key, value in stats.asDict().items()
                if key.startswith(("origin_", "dest_"))
            }
            total_flights = stats["total_flights"]
            record = {
                "split": split_name,
                "start_date": str(stats["min_date"]),
                "end_date": str(stats["max_date"]),
                "total_flights": total_flights,
            }
            if publish_results:
                delayed_flights = stats["delayed_flights"]
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
        write_json(gold_dir / "_features.json", {
            **manifest, "taxi_baselines_sha256": sha256(baseline_path), "upstream_coverage": upstream_coverage,
        })
        summary_pd = pd.DataFrame(summary_records)
        if publish_results:
            update_results({"bts": {"splits": records(summary_pd)}}, results_dir)

        return summary_pd
    finally:
        feature_df.unpersist(blocking=True)
        if schedule is not None:
            schedule.unpersist(blocking=True)
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
