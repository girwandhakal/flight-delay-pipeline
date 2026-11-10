"""Build schedule-based features and assign chronological splits for BTS modeling."""

import json
import os
from pathlib import Path
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]

# Configure Hadoop environment for Windows PySpark runtime if needed.
if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    hadoop_dir = ROOT / "envs" / "hadoop"
    if (hadoop_dir / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

# Ensure PySpark workers use the active Python interpreter.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark_session(app_name="bts_features"):
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_split_config(config_path=None):
    config_path = Path(config_path or ROOT / "config.json")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["splits"]


def build_features(spark, silver_dir=None, gold_dir=None, results_dir=None, config_path=None):
    silver_dir = Path(silver_dir or ROOT / "data" / "silver" / "bts")
    gold_dir = Path(gold_dir or ROOT / "data" / "gold" / "bts")
    results_dir = Path(results_dir or ROOT / "results" / "features_bts")
    results_dir.mkdir(parents=True, exist_ok=True)
    splits = load_split_config(config_path)

    # Read all cleaned Silver Parquet partitions.
    silver_df = spark.read.parquet(str(silver_dir / "*/*"))
    input_count = silver_df.count()
    if input_count == 0:
        raise ValueError(f"No records found in {silver_dir}")

    # Map Spark Sunday=1..Saturday=7 to BTS standard Monday=1..Sunday=7.
    bts_day_of_week = (F.dayofweek(F.col("FlightDate")) + 5) % 7 + 1

    # Extract 0-23 hour from HHMM string; 2400 represents midnight (hour 0).
    dep_hour = (F.col("CRSDepTime").cast("int") / 100).cast("int") % 24
    arr_hour = (F.col("CRSArrTime").cast("int") / 100).cast("int") % 24

    # Build chronological split condition using inclusive date boundaries.
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

    # Ensure every retained flight receives an assigned split.
    unassigned = feature_df.filter(F.col("split").isNull()).count()
    if unassigned > 0:
        raise ValueError(f"Found {unassigned} flights outside configured split boundaries")

    summary_records = []
    # Save partitioned Parquet datasets directly for each chronological split.
    for split_name in ["train", "val", "test"]:
        split_df = feature_df.filter(F.col("split") == split_name)
        dest_split_dir = gold_dir / split_name
        dest_split_dir.mkdir(parents=True, exist_ok=True)
        split_df.write.mode("overwrite").parquet(str(dest_split_dir))

        # Calculate the split summary in one pass.
        stats = split_df.agg(
            F.count("*").alias("total_flights"),
            F.count(F.when(F.col("ArrDel15") == 1, 1)).alias("delayed_flights"),
            F.min("FlightDate").alias("min_date"),
            F.max("FlightDate").alias("max_date"),
        ).first()
        total_flights = stats["total_flights"]
        delayed_flights = stats["delayed_flights"]
        ontime_flights = total_flights - delayed_flights
        delay_rate = delayed_flights / total_flights if total_flights else 0.0

        summary_records.append({
            "split": split_name,
            "start_date": str(stats["min_date"]),
            "end_date": str(stats["max_date"]),
            "total_flights": total_flights,
            "on_time_flights": ontime_flights,
            "delayed_flights": delayed_flights,
            "delay_rate": round(delay_rate, 4),
        })

    summary_pd = pd.DataFrame(summary_records)
    summary_path = results_dir / "split_summary.csv"
    summary_pd.to_csv(summary_path, index=False)

    return summary_pd


def main():
    spark = get_spark_session()
    try:
        summary = build_features(spark)
        total_flights = summary["total_flights"].sum()
        print("-" * 80)
        print(f"Chronological Splitting Summary (Total Processed Flights: {total_flights:,}):")
        print("-" * 80)
        print(summary.to_string(index=False))
        print("-" * 80)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
