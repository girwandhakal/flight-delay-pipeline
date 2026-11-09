"""Clean raw BTS CSV files into Silver Parquet matching Bronze folder structure."""

import os
from pathlib import Path
import shutil
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


def get_spark_session(app_name="bts_cleaning"):
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def clean_bts_data(spark=None, bronze_dir=None, silver_dir=None, results_dir=None):
    bronze_dir = Path(bronze_dir or ROOT / "data" / "bronze" / "bts")
    silver_dir = Path(silver_dir or ROOT / "data" / "silver" / "bts")
    quarantine_dir = Path(ROOT / "data" / "quarantine" / "bts")
    results_dir = Path(results_dir or ROOT / "results" / "cleaning_bts")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(bronze_dir.glob("*/*/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No BTS CSV files found in {bronze_dir}")

    # Remove legacy Hive-style partition directories if present.
    for legacy_dir in silver_dir.glob("Year=*"):
        shutil.rmtree(legacy_dir, ignore_errors=True)

    summary_records = []

    for csv_path in csv_files:
        year_str = csv_path.parent.parent.name
        month_str = csv_path.parent.name
        month_key = f"{year_str}-{month_str}"

        raw_df = spark.read.csv(str(csv_path), header=True)
        raw_count = raw_df.count()

        # Clean string identifiers and pad schedule times to 4-character HHMM.
        crs_dep = F.lpad(F.trim(F.col("CRSDepTime")), 4, "0")
        crs_arr = F.lpad(F.trim(F.col("CRSArrTime")), 4, "0")
        flight_num = F.trim(F.col("Flight_Number_Reporting_Airline"))
        airline = F.trim(F.col("Reporting_Airline"))
        origin = F.trim(F.col("Origin"))
        dest = F.trim(F.col("Dest"))
        flight_date = F.to_date(F.col("FlightDate"), "yyyy-MM-dd")

        # Deterministic flight ID constructed from complete scheduled key.
        flight_id = F.concat_ws("_", F.col("FlightDate"), airline, flight_num, origin, dest, crs_dep)

        cancelled = F.coalesce(F.col("Cancelled").cast("double"), F.lit(1.0))
        diverted = F.coalesce(F.col("Diverted").cast("double"), F.lit(1.0))
        arr_del = F.col("ArrDel15").cast("double")
        distance = F.col("Distance").cast("double")
        elapsed_time = F.col("CRSElapsedTime").cast("double")

        # Exclusions according to agreed cleaning rules.
        canc_count = raw_df.filter(cancelled == 1.0).count()
        div_count = raw_df.filter((cancelled == 0.0) & (diverted == 1.0)).count()

        is_eligible = (cancelled == 0.0) & (diverted == 0.0) & arr_del.isin([0.0, 1.0])
        eligible_df = raw_df.filter(is_eligible)

        # Exclude missing flight numbers from otherwise eligible flights.
        missing_fn_count = eligible_df.filter(flight_num.isNull() | (flight_num == "")).count()
        valid_df = eligible_df.filter(flight_num.isNotNull() & (flight_num != "") & (distance > 0.0) & (elapsed_time > 0.0))

        # Quarantine records with non-positive distance or duration among eligible flights.
        quarantine_df = eligible_df.filter(flight_num.isNotNull() & (flight_num != "") & ((distance <= 0.0) | (elapsed_time <= 0.0)))
        if quarantine_df.count() > 0:
            dest_quarantine = quarantine_dir / year_str / month_str
            dest_quarantine.mkdir(parents=True, exist_ok=True)
            quarantine_df.write.mode("overwrite").parquet(str(dest_quarantine))

        # Project typed silver schema and deduplicate identical flight IDs.
        silver_df = (
            valid_df.select(
                flight_id.alias("flight_id"),
                flight_date.alias("FlightDate"),
                airline.alias("Reporting_Airline"),
                flight_num.alias("Flight_Number_Reporting_Airline"),
                origin.alias("Origin"),
                dest.alias("Dest"),
                crs_dep.alias("CRSDepTime"),
                crs_arr.alias("CRSArrTime"),
                elapsed_time.alias("CRSElapsedTime"),
                distance.alias("Distance"),
                arr_del.cast("int").alias("ArrDel15"),
                F.lit(0).alias("Cancelled"),
                F.lit(0).alias("Diverted"),
                F.lit(csv_path.name).alias("source_file"),
                F.lit(int(year_str)).alias("Year"),
                F.lit(int(month_str)).alias("Month"),
            )
            .dropDuplicates(["flight_id"])  # Deduplicate exact complete keys if any repeats occur.
        )

        dest_month_dir = silver_dir / year_str / month_str
        dest_month_dir.mkdir(parents=True, exist_ok=True)
        silver_df.write.mode("overwrite").parquet(str(dest_month_dir))

        retained_count = silver_df.count()

        summary_records.append({
            "month": month_key,
            "raw_records": raw_count,
            "excluded_cancelled": canc_count,
            "excluded_diverted": div_count,
            "missing_target_total": canc_count + div_count,
            "excluded_missing_flight_num": missing_fn_count,
            "retained_records": retained_count,
        })

    summary_pd = pd.DataFrame(summary_records)
    summary_path = results_dir / "cleaning_summary.csv"
    summary_pd.to_csv(summary_path, index=False)

    return summary_pd


def main():
    spark = get_spark_session()
    try:
        summary = clean_bts_data(spark)
        print("BTS cleaning completed successfully:")
        print(summary.to_string(index=False))
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
