"""Normalize verified LCDv2 reports into UTC observations and retain rejection counts."""

from datetime import datetime, timezone
from pathlib import Path
import json

from scripts.results import records, update_results

from pyspark.sql import functions as F, types as T, Window

from .download import (
    ROOT,
    REPORTS,
    load_config,
    standard_offset,
    write_json,
)
from .measurements import (
    SOURCE_FIELDS, NUMERIC_FEATURES, MISSING_FLAGS, INDICATOR_FLAGS,
    FEATURE_COLUMNS, UNITS, decode_weather,
)


def clean_weather(spark, bronze_dir, silver_dir, quarantine_dir, manifest, report_dir):
    if (
        not manifest.get("complete")
        or manifest.get("timestamp_basis") not in {"local_standard", "metar_verified_utc_or_local_standard"}
    ):
        raise ValueError("Weather acquisition/time verification is incomplete.")
    entries = [
        entry for entry in manifest["files"].values() if entry["status"] == "verified"
    ]
    if not entries:
        raise ValueError("No verified weather files available.")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    paths = [
        str(
            Path(bronze_dir)
            / f"{entry['year']}/LCD_{entry['station_id']}_{entry['year']}.csv"
        )
        for entry in entries
    ]
    offsets = spark.createDataFrame(
        [
            (e["station_id"], e["year"], standard_offset(e["time_zone"], e["year"]))
            for e in entries
        ],
        "station_id string, source_year int, standard_offset int",
    ).dropDuplicates()
    if (
        offsets.groupBy("station_id", "source_year")
        .count()
        .filter("count > 1")
        .limit(1)
        .count()
    ):
        raise ValueError("Conflicting station time zones.")
    raw = (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .csv(paths)
    )
    audit_columns = [name for name in ("LATITUDE", "LONGITUDE", "ELEVATION") if name in raw.columns]
    missing = set(SOURCE_FIELDS) - set(raw.columns)
    if missing:
        raise ValueError(f"Weather source lacks required LCDv2 fields: {sorted(missing)}")
    raw = (
        raw.select(
            "STATION",
            "DATE",
            "REPORT_TYPE",
            "REM",
            *SOURCE_FIELDS,
            *audit_columns,
            F.input_file_name().alias("source_file"),
        )
        .withColumnRenamed("STATION", "station_id")
        .withColumn("source_year", F.substring("DATE", 1, 4).cast("int"))
    )
    csv_clock = F.to_timestamp("DATE", "yyyy-MM-dd'T'HH:mm:ss")
    metar_clock = F.regexp_extract("REM", r"(?:METAR|SPECI)\s+\w+\s+(\d{6})Z", 1)
    # Some station files use UTC; accept it only when the embedded UTC clock agrees.
    parsed = raw.join(
        F.broadcast(offsets), ["station_id", "source_year"], "left"
    ).withColumn(
        "observation_utc",
        F.when(metar_clock == F.date_format(csv_clock, "ddHHmm"), csv_clock).otherwise(
            F.from_unixtime(csv_clock.cast("long") - F.col("standard_offset")).cast("timestamp")
        ),
    )
    schema = T.StructType(
        [T.StructField(name, T.DoubleType(), True) for name in NUMERIC_FEATURES]
        + [T.StructField(name, T.IntegerType(), False) for name in [*MISSING_FLAGS, *INDICATOR_FLAGS]]
    )
    decode = F.udf(lambda *tokens: decode_weather(dict(zip(SOURCE_FIELDS, tokens))), schema)
    parsed = parsed.withColumn("decoded_weather", decode(*SOURCE_FIELDS)).select(
        "*", "decoded_weather.*"
    ).drop("decoded_weather").cache()
    try:
        inconsistent = F.col("REPORT_TYPE").isin(REPORTS) & (
            metar_clock != F.date_format("observation_utc", "ddHHmm")
        )
        invalid = (
            F.col("station_id").isNull()
            | F.col("observation_utc").isNull()
            | F.coalesce(inconsistent, F.lit(True))
        )
        stats = (
            parsed.agg(
                F.count("*").alias("input_rows"),
                F.sum(invalid.cast("int")).alias("invalid_rows"),
                F.sum(
                    (
                        ~F.col("REPORT_TYPE").isin(REPORTS)
                        | F.col("REPORT_TYPE").isNull()
                    ).cast("int")
                ).alias("excluded_reports"),
            )
            .first()
            .asDict()
        )
        parsed.filter(invalid).withColumn(
            "reason", F.lit("invalid_station_or_time")
        ).write.parquet(str(quarantine_dir))
        observations = parsed.filter(~invalid & F.col("REPORT_TYPE").isin(REPORTS))
        # Prefer routine METAR to a conflicting special report, then use a stable tie break.
        priority = Window.partitionBy("station_id", "observation_utc").orderBy(
            F.when(F.col("REPORT_TYPE") == "FM-15", 0).otherwise(1),
            F.col("source_file"),
            F.sha2(
                F.concat_ws(
                    "|",
                    *[F.coalesce(F.col(c), F.lit("")) for c in ["REM", *SOURCE_FIELDS]],
                ),
                256,
            ),
        )
        observations = (
            observations.withColumn("rank", F.row_number().over(priority))
            .filter("rank = 1")
            .select(
                "station_id",
                "observation_utc",
                "source_file",
                "REPORT_TYPE",
                "REM",
                *SOURCE_FIELDS,
                *FEATURE_COLUMNS,
                *audit_columns,
            )
            .withColumn("year", F.year("observation_utc"))
            .withColumn("month", F.month("observation_utc"))
        )
        observations.write.partitionBy("year", "month").parquet(str(silver_dir))
        clean = spark.read.parquet(str(silver_dir))
        stats["retained_observations"] = clean.count()
        eligible = parsed.filter(~invalid & F.col("REPORT_TYPE").isin(REPORTS)).count()
        stats["duplicate_observations"] = eligible - stats["retained_observations"]
        stats["excluded_reports"] = parsed.filter(
            ~invalid
            & (~F.col("REPORT_TYPE").isin(REPORTS) | F.col("REPORT_TYPE").isNull())
        ).count()
        if stats["input_rows"] != sum(
            stats[key]
            for key in (
                "invalid_rows",
                "excluded_reports",
                "duplicate_observations",
                "retained_observations",
            )
        ):
            raise ValueError("Weather cleaning counts do not reconcile.")
        cadence = Window.partitionBy("station_id").orderBy("observation_utc")
        station_quality = (
            clean.withColumn(
                "gap_minutes",
                (
                    F.col("observation_utc").cast("long")
                    - F.lag(F.col("observation_utc").cast("long")).over(cadence)
                )
                / 60,
            )
            .groupBy("station_id")
            .agg(
                F.count("*").alias("reports"),
                F.avg("gap_minutes").alias("mean_gap_minutes"),
                F.max("gap_minutes").alias("max_gap_minutes"),
                *[
                    F.sum(F.col(name).isNull().cast("int")).alias(name + "_missing")
                    for name in NUMERIC_FEATURES
                ],
            )
            .toPandas()
        )
        write_json(
            Path(report_dir) / "_metadata.json",
            {
                "quality": stats,
                "station_quality": records(station_quality),
                "source_contract": {
                    "timestamp_basis": "metar_verified_utc_or_local_standard",
                    "report_types": REPORTS,
                    "fields": SOURCE_FIELDS,
                    "numeric_features": NUMERIC_FEATURES,
                    "missingness_flags": MISSING_FLAGS,
                    "indicator_flags": INDICATOR_FLAGS,
                    "units": UNITS,
                    "time_evidence": "CSV clocks are interpreted as UTC or local standard time only when the embedded METAR/SPECI UTC clock agrees. Disagreements and missing embedded clocks are quarantined, including observed DST-transition source anomalies.",
                    "quality_policy": "Strict finite numeric tokens; flagged/suffixed tokens become null. Negative values are allowed only for temperature/dew point; humidity must be 0-100 and pressure positive. Trace precipitation is encoded as lower-bound 0 mm plus a trace flag. Direction uses sine/cosine plus calm/variable flags. Weather contractions are multi-hot; blank/unsupported codes remain unknown rather than clear. Cloud oktas and lowest BKN/OVC/VV ceiling are decoded; CSV heights are multiplied by 100 to meters. No ceiling is distinguished from missing ceiling. All features use one report; no forward fill. Raw fields and REM are retained.",
                    "duplicate_priority": "FM-15, FM-16, source filename, raw-value SHA-256",
                },
            },
        )
        return clean
    finally:
        parsed.unpersist()


def main():
    from scripts.bts.build_features import get_spark_session
    import json

    settings, _, _ = load_config()
    manifest = json.loads((settings["output_directory"] / "manifest.json").read_text())
    build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    spark = get_spark_session("weather_clean")
    try:
        clean_weather(
            spark,
            settings["output_directory"],
            ROOT / "data/silver/weather" / build_id,
            ROOT / "data/quarantine/weather" / build_id,
            manifest,
            ROOT / "data/silver/weather" / build_id,
        )
        metadata = ROOT / "data/silver/weather" / build_id / "_metadata.json"
        update_results({"bts_origin_weather": {"cleaning": json.loads(metadata.read_text())}})
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
