"""Normalize downloaded LCDv2 reports into UTC observations and retain rejection counts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Union

from pyspark.sql import DataFrame, SparkSession, Window, functions as F, types as T

from scripts.common.config import ROOT, load_config
from scripts.common.io import file_fingerprints, sha256, write_json
from scripts.common.results import records, update_results
from scripts.common.spark import get_spark_session
from .download import REPORTS, standard_offset
from .measurements import (
    SOURCE_FIELDS, NUMERIC_FEATURES, MISSING_FLAGS, INDICATOR_FLAGS,
    FEATURE_COLUMNS, UNITS, decode_weather,
)


def clean_weather(
    spark: SparkSession,
    bronze_dir: Union[str, Path],
    silver_dir: Union[str, Path],
    quarantine_dir: Union[str, Path],
    manifest: dict[str, Any],
    report_dir: Union[str, Path],
) -> DataFrame:
    """Clean and decode raw Bronze LCDv2 CSVs into partitioned Silver observations."""
    if not manifest.get("complete"):
        raise ValueError("Weather acquisition is incomplete.")
    entries = [
        entry for entry in manifest["files"].values()
        if entry["status"] in {"downloaded", "verified"}
    ]
    if not entries:
        raise ValueError("No downloaded weather files available.")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    silver_dir = Path(silver_dir)
    cache_inputs = {
        "files": file_fingerprints([
            Path(bronze_dir) / f"{e['year']}/LCD_{e['station_id']}_{e['year']}.csv"
            for e in entries
        ]),
        "stations": sorted([e["station_id"], e["year"], e["time_zone"]] for e in entries),
        "code": {name: sha256(Path(__file__).with_name(name))
                 for name in ("clean.py", "measurements.py", "download.py")},
        "spark": spark.version,
    }
    for candidate in sorted(silver_dir.parent.glob("*/_metadata.json"), reverse=True):
        try:
            metadata = json.loads(candidate.read_text())
            if not isinstance(metadata, dict) or not all(
                name in metadata for name in ("quality", "station_quality", "source_contract")
            ):
                continue
            cached = metadata.get("cleaning_cache", {})
            if not isinstance(cached, dict):
                continue
            inputs_match = cached.get("inputs") == cache_inputs
            has_success_marker = (candidate.parent / "_SUCCESS").is_file()
            cached_outputs = cached.get("outputs")
            outputs_match = bool(cached_outputs) and cached_outputs == file_fingerprints(
                candidate.parent.rglob("*.parquet")
            )
            if not (inputs_match and has_success_marker and outputs_match):
                continue
        except (OSError, ValueError):
            continue
        clean = spark.read.parquet(str(candidate.parent), pathGlobFilter="*.parquet")
        write_json(Path(report_dir) / "_metadata.json", metadata)
        print(
            f"  Reusing cached weather Silver ({candidate.parent.name}) because inputs and cleaning code match.",
            flush=True,
        )
        return clean
    print("  Cleaning raw NOAA weather CSVs (checking per-station-year Silver cache).", flush=True)
    station_cache_root = silver_dir.parent / "_by_station"
    cached_dirs: list[Path] = []
    cached_qualities: list[dict[str, int]] = []
    missing_entries: list[tuple[dict[str, Any], dict[str, Any], Path]] = []

    for entry in sorted(entries, key=lambda item: (item["station_id"], int(item["year"]))):
        station_id = entry["station_id"]
        year = int(entry["year"])
        station_dir = station_cache_root / f"{station_id}_{year}"
        entry_inputs = {
            "file": file_fingerprints([Path(bronze_dir) / f"{year}/LCD_{station_id}_{year}.csv"]),
            "station": [station_id, year, entry["time_zone"]],
            "code": cache_inputs["code"],
            "spark": spark.version,
        }
        meta_path = station_dir / "_metadata.json"
        if meta_path.is_file() and (station_dir / "_SUCCESS").is_file():
            try:
                station_meta = json.loads(meta_path.read_text())
                station_cached = station_meta.get("cleaning_cache", {})
                if (
                    isinstance(station_cached, dict)
                    and station_cached.get("inputs") == entry_inputs
                    and station_cached.get("outputs")
                    and station_cached.get("outputs") == file_fingerprints(station_dir.rglob("*.parquet"))
                ):
                    cached_dirs.append(station_dir)
                    cached_qualities.append(station_meta["quality"])
                    continue
            except (OSError, ValueError, KeyError):
                pass
        missing_entries.append((entry, entry_inputs, station_dir))

    if missing_entries:
        print(
            f"  Weather Silver cache: {len(cached_dirs)} station-year(s) reused, "
            f"{len(missing_entries)} to clean.",
            flush=True,
        )
        missing_raw_entries = [item[0] for item in missing_entries]
        paths = [
            str(
                Path(bronze_dir)
                / f"{entry['year']}/LCD_{entry['station_id']}_{entry['year']}.csv"
            )
            for entry in missing_raw_entries
        ]
        offsets = spark.createDataFrame(
            [
                (e["station_id"], e["year"], standard_offset(e["time_zone"], e["year"]))
                for e in missing_raw_entries
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
        missing_cols = set(SOURCE_FIELDS) - set(raw.columns)
        if missing_cols:
            raise ValueError(f"Weather source lacks required LCDv2 fields: {sorted(missing_cols)}")
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
            .withColumn(
                "_cache_key",
                F.concat_ws(
                    "_",
                    F.col("station_id"),
                    F.regexp_extract("source_file", r"LCD_[^/\\]+_(\d{4})\.csv$", 1),
                ),
            )
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
            is_report_excluded = (
                ~F.col("REPORT_TYPE").isin(REPORTS) | F.col("REPORT_TYPE").isNull()
            )
            # Compute per-station-year quality counts in one pass over decoded observations.
            parsed_stats = {
                row["_cache_key"]: {
                    "input_rows": int(row["input_rows"]),
                    "invalid_rows": int(row["invalid_rows"]),
                    "excluded_reports": int(row["excluded_reports"]),
                }
                for row in parsed.groupBy("_cache_key")
                .agg(
                    F.count("*").alias("input_rows"),
                    F.sum(invalid.cast("int")).alias("invalid_rows"),
                    F.sum((~invalid & is_report_excluded).cast("int")).alias("excluded_reports"),
                )
                .collect()
            }
            parsed.filter(invalid).drop("_cache_key").withColumn(
                "reason", F.lit("invalid_station_or_time")
            ).write.parquet(str(quarantine_dir))
            eligible_df = parsed.filter(~invalid & F.col("REPORT_TYPE").isin(REPORTS))
            eligible_counts = {
                row["_cache_key"]: int(row["eligible"])
                for row in eligible_df.groupBy("_cache_key").agg(F.count("*").alias("eligible")).collect()
            }
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
                eligible_df.withColumn("rank", F.row_number().over(priority))
                .filter("rank = 1")
                .select(
                    "_cache_key",
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
            retained_counts = {
                row["_cache_key"]: int(row["retained"])
                for row in observations.groupBy("_cache_key").agg(F.count("*").alias("retained")).collect()
            }
            staging_dir = station_cache_root / f"_staging_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            try:
                observations.write.partitionBy("_cache_key").parquet(str(staging_dir))
                for entry, entry_inputs, station_dir in missing_entries:
                    key = f"{entry['station_id']}_{int(entry['year'])}"
                    shutil.rmtree(station_dir, ignore_errors=True)
                    part_dir = staging_dir / f"_cache_key={key}"
                    if part_dir.is_dir():
                        station_dir.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(part_dir), str(station_dir))
                    else:
                        observations.drop("_cache_key").limit(0).write.parquet(str(station_dir))
                    p_stat = parsed_stats.get(key, {"input_rows": 0, "invalid_rows": 0, "excluded_reports": 0})
                    retained_n = retained_counts.get(key, 0)
                    eligible_n = eligible_counts.get(key, 0)
                    station_quality_stats = {
                        "input_rows": p_stat["input_rows"],
                        "invalid_rows": p_stat["invalid_rows"],
                        "excluded_reports": p_stat["excluded_reports"],
                        "duplicate_observations": eligible_n - retained_n,
                        "retained_observations": retained_n,
                    }
                    if station_quality_stats["input_rows"] != sum(
                        station_quality_stats[k]
                        for k in ("invalid_rows", "excluded_reports", "duplicate_observations", "retained_observations")
                    ):
                        raise ValueError(f"Weather cleaning counts do not reconcile for {key}.")
                    write_json(
                        station_dir / "_metadata.json",
                        {
                            "cleaning_cache": {
                                "inputs": entry_inputs,
                                "outputs": file_fingerprints(station_dir.rglob("*.parquet")),
                            },
                            "quality": station_quality_stats,
                        },
                    )
                    (station_dir / "_SUCCESS").write_text("", encoding="utf-8")
                    cached_dirs.append(station_dir)
                    cached_qualities.append(station_quality_stats)
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
        finally:
            parsed.unpersist(blocking=True)
    else:
        print(
            f"  Reusing all {len(cached_dirs)} cached per-station-year weather Silver partitions.",
            flush=True,
        )

    clean = spark.read.parquet(*[str(d) for d in cached_dirs], pathGlobFilter="*.parquet")
    stats = {
        key: sum(item[key] for item in cached_qualities)
        for key in ("input_rows", "invalid_rows", "excluded_reports", "duplicate_observations", "retained_observations")
    }
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
            "cleaning_cache": {
                "inputs": cache_inputs,
                "outputs": file_fingerprints(
                    p for d in cached_dirs for p in d.rglob("*.parquet")
                ),
            },
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


def main() -> int:
    """Clean and validate raw Bronze weather data into Silver observation tables."""
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
