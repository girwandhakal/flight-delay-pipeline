"""Clean raw BTS CSV files into Silver Parquet matching Bronze folder structure."""

"""Clean raw BTS CSV files into Silver Parquet matching Bronze folder structure."""

from pathlib import Path
import sys
from typing import Optional, Union

from scripts.results import records, update_results

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from .build_features import get_spark_session as _get_spark_session

ROOT = Path(__file__).resolve().parents[2]


def get_spark_session(app_name: str = "bts_cleaning") -> SparkSession:
    """Create or retrieve a Spark session configured for BTS cleaning."""
    return _get_spark_session(app_name=app_name)


def clean_bts_data(
    spark: Optional[SparkSession] = None,
    bronze_dir: Optional[Union[str, Path]] = None,
    silver_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    quarantine_dir: Optional[Union[str, Path]] = None,
    publish_results: bool = True,
    selected_months: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Clean and filter monthly BTS raw CSV files into typed Silver Parquet datasets.

    Enforces domain validity rules:
    - Retains only non-cancelled, non-diverted flights with confirmed binary arrival delay.
    - Excludes flights with missing airline flight numbers.
    - Quarantines flights with non-positive distance or duration.
    - Generates a deterministic unique composite `flight_id`.
    - Deduplicates exact duplicate flight keys.

    Args:
        spark: Active SparkSession instance.
        bronze_dir: Directory containing raw BTS CSVs. Defaults to <ROOT>/data/bronze/bts.
        silver_dir: Explicit output directory for a standalone build. The default pipeline uses generations.
        results_dir: Directory for publishing results.json. Defaults to <ROOT>/results.
        quarantine_dir: Directory for quarantined records. Cached builds keep them with the Silver generation.
        publish_results: Whether to publish cleaning summary to results.json.

    Returns:
        Pandas DataFrame summarizing raw, excluded, and retained records by month.
    """
    if all(path is None for path in (bronze_dir, silver_dir, results_dir, quarantine_dir)):
        from .cache import prepare
        return prepare(spark, ROOT, "silver", publish_results)
    bronze_dir = Path(bronze_dir or ROOT / "data" / "bronze" / "bts")
    silver_dir = Path(silver_dir or ROOT / "data" / "silver" / "bts")
    quarantine_dir = Path(quarantine_dir or ROOT / "data" / "quarantine" / "bts")
    results_dir = Path(results_dir or ROOT / "results")

    csv_files = sorted(bronze_dir.glob("*/*/*.csv"))
    if selected_months is not None:
        csv_files = [path for path in csv_files if path.parent.relative_to(bronze_dir).as_posix() in selected_months]
    if not csv_files:
        raise FileNotFoundError(f"No BTS CSV files found in {bronze_dir}")

    summary_records = []

    for csv_path in csv_files:
        year_str = csv_path.parent.parent.name
        month_str = csv_path.parent.name
        month_key = f"{year_str}-{month_str}"

        raw_df = spark.read.csv(str(csv_path), header=True)

        # Standardize string identifiers and pad scheduled times to 4-character HHMM
        crs_dep = F.lpad(F.trim(F.col("CRSDepTime")), 4, "0")
        crs_arr = F.lpad(F.trim(F.col("CRSArrTime")), 4, "0")
        flight_num = F.trim(F.col("Flight_Number_Reporting_Airline"))
        airline = F.trim(F.col("Reporting_Airline"))
        origin = F.trim(F.col("Origin"))
        dest = F.trim(F.col("Dest"))
        flight_date = F.to_date(F.col("FlightDate"), "yyyy-MM-dd")

        # Deterministic unique flight ID constructed from complete scheduled key
        flight_id = F.concat_ws(
            "_", F.col("FlightDate"), airline, flight_num, origin, dest, crs_dep
        )

        # Default missing Cancelled/Diverted to 1.0 so unconfirmed flights are safely excluded
        cancelled = F.coalesce(F.col("Cancelled").cast("double"), F.lit(1.0))
        diverted = F.coalesce(F.col("Diverted").cast("double"), F.lit(1.0))
        arr_del = F.col("ArrDel15").cast("double")
        distance = F.col("Distance").cast("double")
        elapsed_time = F.col("CRSElapsedTime").cast("double")

        # Retain only non-cancelled, non-diverted flights with confirmed binary arrival delay label
        is_eligible = (cancelled == 0.0) & (diverted == 0.0) & arr_del.isin([0.0, 1.0])
        has_flight_num = flight_num.isNotNull() & (flight_num != "")
        has_positive_dimensions = (distance > 0.0) & (elapsed_time > 0.0)

        # Compute summary metrics and counts in a single pass over raw data
        stats = raw_df.agg(
            F.count("*").alias("raw_count"),
            F.count(F.when(cancelled == 1.0, 1)).alias("canc_count"),
            F.count(F.when((cancelled == 0.0) & (diverted == 1.0), 1)).alias("div_count"),
            F.count(F.when(is_eligible & ~has_flight_num, 1)).alias("missing_fn_count"),
            F.count(F.when(is_eligible & has_flight_num & ~has_positive_dimensions, 1)).alias("quarantine_count"),
        ).first()

        raw_count = stats["raw_count"]
        canc_count = stats["canc_count"]
        div_count = stats["div_count"]
        missing_fn_count = stats["missing_fn_count"]
        quarantine_count = stats["quarantine_count"]

        eligible_df = raw_df.filter(is_eligible)
        valid_df = eligible_df.filter(has_flight_num & has_positive_dimensions)

        # Quarantine records with non-positive distance or duration among eligible flights
        if quarantine_count > 0:
            quarantine_df = eligible_df.filter(has_flight_num & ~has_positive_dimensions)
            dest_quarantine = quarantine_dir / year_str / month_str
            dest_quarantine.mkdir(parents=True, exist_ok=True)
            quarantine_df.write.mode("overwrite").parquet(str(dest_quarantine))

        # Project typed silver schema and deduplicate identical flight IDs
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
            ).dropDuplicates(["flight_id"])
        )

        dest_month_dir = silver_dir / year_str / month_str
        dest_month_dir.mkdir(parents=True, exist_ok=True)
        silver_df.write.mode("overwrite").parquet(str(dest_month_dir))

        retained_count = silver_df.count()

        summary_records.append(
            {
                "month": month_key,
                "raw_records": raw_count,
                "excluded_cancelled": canc_count,
                "excluded_diverted": div_count,
                "missing_target_total": canc_count + div_count,
                "excluded_missing_flight_num": missing_fn_count,
                "retained_records": retained_count,
            }
        )

    summary_pd = pd.DataFrame(summary_records)
    if publish_results:
        update_results({"bts": {"cleaning": records(summary_pd)}}, results_dir)

    return summary_pd


def main(publish_results: bool = True) -> int:
    """Clean BTS raw CSV files and display tabular summary."""
    spark = get_spark_session()
    try:
        summary = clean_bts_data(spark, publish_results=publish_results)
        total_retained = summary["retained_records"].sum()
        print("-" * 80)
        print(
            f"BTS Cleaning Summary Table (Total Retained Flights: {total_retained:,}):"
        )
        print("-" * 80)
        print(summary.to_string(index=False))
        print("-" * 80)
        return 0
    finally:
        spark.stop()



if __name__ == "__main__":
    sys.exit(main())
