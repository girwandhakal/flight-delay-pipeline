"""Attach the latest eligible origin observation without changing the BTS cohort."""

import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo

from pyspark.sql import DataFrame, SparkSession, Window, functions as F, types as T

from scripts.bts.cache import active
from scripts.common.config import ROOT, load_config
from scripts.common.io import write_json
from scripts.common.progress import stage as progress_stage
from scripts.common.results import records, update_results
from scripts.common.spark import get_spark_session
from scripts.common.stage_cache import find_gold, input_identity, save_gold, scoped_manifest
from scripts.common.time_utils import local_cutoff
from .download import standard_offset
from .measurements import NUMERIC_FEATURES, MISSING_FLAGS, INDICATOR_FLAGS, FEATURE_COLUMNS

WEATHER_NUMERIC = [*NUMERIC_FEATURES, "origin_weather_age_minutes"]
WEATHER_FLAGS = ["origin_weather_missing", *MISSING_FLAGS, *INDICATOR_FLAGS]


def load_references(
    directory: Union[str, Path],
    prefix: str = "origin",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load airports.csv and airport_station.csv mappings and validate interval validity."""
    validate_prefix(prefix)
    directory = Path(directory)
    with (directory / "airports.csv").open(encoding="utf-8", newline="") as stream:
        airports = list(csv.DictReader(stream))
    with (directory / "airport_station.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        mappings = list(csv.DictReader(stream))
    for rows in (airports, mappings):
        for row in rows:
            if date.fromisoformat(row["valid_from"]) >= date.fromisoformat(
                row["valid_to"]
            ):
                raise ValueError("Reference interval must be nonempty.")
        ordered = sorted(rows, key=lambda r: (r["Origin"], r["valid_from"]))
        for previous, current in zip(ordered, ordered[1:]):
            if (
                prefix == "origin"
                and previous["Origin"] == current["Origin"]
                and previous["valid_to"] > current["valid_from"]
            ):
                raise ValueError(
                    f"Overlapping reference intervals: {current['Origin']}"
                )
    if prefix == "origin":
        for row in airports:
            ZoneInfo(row["time_zone"])
    for row in mappings:
        if prefix == "origin" and (not row["station_id"] or not any(
            a["Origin"] == row["Origin"]
            and a["valid_from"] <= row["valid_from"]
            and a["valid_to"] >= row["valid_to"]
            for a in airports
        )):
            raise ValueError("Station mapping lacks a valid airport interval.")
    return airports, mappings


def build_flight_cutoffs(
    spark: SparkSession,
    flights: DataFrame,
    references: tuple[list[dict[str, str]], list[dict[str, str]]],
) -> DataFrame:
    """Attach airport time zone, mapped station ID, and UTC departure timestamp to flights."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    airports, mappings = references
    airport_df = spark.createDataFrame(
        [
            (r["Origin"], r["time_zone"], r["valid_from"], r["valid_to"])
            for r in airports
        ],
        "airport_code string, origin_time_zone string, airport_from string, airport_to string",
    )
    mapping_df = spark.createDataFrame(
        [
            (r["Origin"], r["station_id"], r["valid_from"], r["valid_to"], r["method"])
            for r in mappings
        ],
        "mapping_code string, origin_station_id string, mapping_from string, mapping_to string, origin_mapping_method string",
    )
    joined = flights.join(
        F.broadcast(airport_df),
        (flights.Origin == airport_df.airport_code)
        & (F.col("FlightDate") >= F.to_date("airport_from"))
        & (F.col("FlightDate") < F.to_date("airport_to")),
        "left",
    )
    joined = joined.join(
        F.broadcast(mapping_df),
        (joined.Origin == mapping_df.mapping_code)
        & (F.col("FlightDate") >= F.to_date("mapping_from"))
        & (F.col("FlightDate") < F.to_date("mapping_to")),
        "left",
    )

    def convert_clock(flight_date, clock, zone):
        utc, status = local_cutoff(flight_date, clock, zone)
        return utc.isoformat() + "Z" if utc else None, status

    convert = F.udf(convert_clock, "utc_clock string, origin_cutoff_status string")
    # Deduplicate (FlightDate, CRSDepTime, origin_time_zone) keys before Python UDF evaluation.
    cutoff_keys = (
        joined.select("FlightDate", "CRSDepTime", "origin_time_zone")
        .distinct()
        .withColumn("cutoff", convert("FlightDate", "CRSDepTime", "origin_time_zone"))
        .select(
            F.col("FlightDate").alias("_k_date"),
            F.col("CRSDepTime").alias("_k_clock"),
            F.col("origin_time_zone").alias("_k_zone"),
            "cutoff.*",
        )
    )
    return (
        joined.join(
            cutoff_keys,
            joined.FlightDate.eqNullSafe(cutoff_keys._k_date)
            & joined.CRSDepTime.eqNullSafe(cutoff_keys._k_clock)
            & joined.origin_time_zone.eqNullSafe(cutoff_keys._k_zone),
            "left",
        )
        .select(
            *flights.columns,
            "origin_time_zone",
            "origin_station_id",
            "origin_mapping_method",
            "utc_clock",
            "origin_cutoff_status",
        )
        .withColumn("scheduled_departure_utc", F.to_timestamp("utc_clock"))
        .drop("utc_clock")
    )


def required_station_years(
    cutoffs: DataFrame,
    max_age_minutes: int,
    prefix: str = "origin",
) -> list[tuple[str, int, str]]:
    """Determine all (station_id, year, timezone) tuples needed to cover the flight cutoffs."""
    validate_prefix(prefix)
    cutoffs = cutoffs.select(
        "scheduled_departure_utc",
        F.col(f"{prefix}_station_id").alias("origin_station_id"),
        F.col(f"{prefix}_time_zone").alias("origin_time_zone"),
    )
    bounds = (
        cutoffs.filter(
            "scheduled_departure_utc is not null and origin_station_id is not null"
        )
        .groupBy("origin_station_id", "origin_time_zone")
        .agg(
            F.date_format(
                F.min("scheduled_departure_utc"), "yyyy-MM-dd'T'HH:mm:ss"
            ).alias("start"),
            F.date_format(
                F.max("scheduled_departure_utc"), "yyyy-MM-dd'T'HH:mm:ss"
            ).alias("end"),
        )
        .collect()
    )
    requests = []
    for row in bounds:
        start, end = datetime.fromisoformat(row.start), datetime.fromisoformat(row.end)
        for year in range(start.year - 1, end.year + 2):
            offset = timedelta(seconds=standard_offset(row.origin_time_zone, year))
            if (
                (min(start, start + offset) - timedelta(minutes=max_age_minutes)).year
                <= year
                <= max(end, end + offset).year
            ):
                requests.append((row.origin_station_id, year, row.origin_time_zone))
    if not requests and prefix == "origin":
        raise ValueError("No mapped origin cutoffs; cannot train a weather model.")
    return requests


def validate_prefix(prefix: str) -> None:
    """Validate that weather role prefix is either 'origin' or 'destination'."""
    if prefix not in ("origin", "destination"):
        raise ValueError("Weather role must be origin or destination.")


def bts_fingerprints() -> list[dict[str, Any]]:
    """Return provenance fingerprints (relative path, size, mtime) for all BTS Gold parquet files."""
    gold = active(ROOT, "gold")
    return [
        {
            "path": str(path.relative_to(gold)),
            "bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(gold.rglob("*.parquet"))
    ]


def build_origin_features(
    cutoffs: DataFrame,
    observations: DataFrame,
    manifest: dict[str, Any],
    max_age_minutes: int = 120,
) -> DataFrame:
    """Enrich flight cutoffs with origin weather observations matching the configured time policy."""
    return build_weather_features(cutoffs, observations, manifest, max_age_minutes)


def build_weather_features(
    cutoffs: DataFrame,
    observations: DataFrame,
    manifest: dict[str, Any],
    max_age_minutes: int = 120,
    prefix: str = "origin",
) -> DataFrame:
    """Match the latest eligible weather observation within the maximum age to each flight."""
    validate_prefix(prefix)
    observation = "observation_utc" if prefix == "origin" else "destination_observation_utc"
    source = "source_file" if prefix == "origin" else "destination_source_file"
    if max_age_minutes <= 0:
        raise ValueError("Invalid maximum weather age.")
    missing = set(FEATURE_COLUMNS) - set(observations.columns)
    if missing:
        raise ValueError("Weather features are outdated; rebuild origin-weather Silver with all 11 fields.")
    weather = observations.withColumn(
        "weather_hour", F.floor(F.col("observation_utc").cast("long") / 3600)
    ).alias("weather")
    # Integer hour buckets (floor of epoch seconds / 3600) bound candidate expansion
    # before exact timestamp checks, enabling a scalable merge join.
    flights = cutoffs.select(
        "flight_id", f"{prefix}_station_id", "scheduled_departure_utc"
    ).withColumn(
        "match_hour",
        F.explode_outer(
            F.sequence(
                F.floor(
                    (
                        F.col("scheduled_departure_utc").cast("long")
                        - max_age_minutes * 60
                    )
                    / 3600
                ).cast("long"),
                F.floor(F.col("scheduled_departure_utc").cast("long") / 3600).cast("long"),
            )
        ),
    ).alias("flights")
    condition = (
        (F.col(f"flights.{prefix}_station_id") == F.col("weather.station_id"))
        & (F.col("match_hour") == F.col("weather_hour"))
        & (
            F.col("weather.observation_utc")
            >= F.col("flights.scheduled_departure_utc")
            - F.expr(f"INTERVAL {max_age_minutes} MINUTES")
        )
        & (
            F.col("weather.observation_utc")
            <= F.col("flights.scheduled_departure_utc")
        )
    )
    # Keep cohort-scale matches partitioned rather than broadcasting them on the driver.
    selected = (
        flights.hint("merge").join(weather, condition, "inner")
        .select(
            "flights.flight_id",
            F.col("weather.observation_utc").alias(observation),
            F.col("weather.source_file").alias(source),
            *[F.col("weather." + name).alias(name.replace("origin_", prefix + "_", 1)) for name in FEATURE_COLUMNS],
            *[F.col("weather." + name).alias(prefix + "_" + name)
              for name in observations.columns
              if prefix == "destination" and name not in {*FEATURE_COLUMNS, "station_id", "observation_utc", "source_file", "year", "month"}],
            *[
                F.col("weather." + name).alias(output.replace("origin_", prefix + "_", 1))
                for name, output in [
                    ("LATITUDE", "origin_station_latitude"),
                    ("LONGITUDE", "origin_station_longitude"),
                    ("ELEVATION", "origin_station_elevation_raw"),
                ] if name in observations.columns
            ],
        )
        .withColumn(
            "rank",
            F.row_number().over(
                Window.partitionBy("flight_id").orderBy(
                    F.col(observation).desc_nulls_last(),
                    F.col(source).asc_nulls_last(),
                )
            ),
        )
    )
    selected = selected.filter("rank = 1").drop("rank")
    # Attach the wide parent only after ranking actual matches, avoiding repeated payloads.
    selected = cutoffs.hint("merge").join(selected, "flight_id", "left").select(
        *cutoffs.columns, *[name for name in selected.columns if name != "flight_id"]
    )
    selected = (
        selected.withColumn(
            f"{prefix}_weather_age_minutes",
            (
                F.col("scheduled_departure_utc").cast("long")
                - F.col(observation).cast("long")
            )
            / 60,
        )
        .withColumn(
            f"{prefix}_available_utc",
            F.col(observation),
        )
        .withColumn(
            f"{prefix}_weather_missing", F.col(observation).isNull().cast("int")
        )
    )
    for flag in MISSING_FLAGS:
        flag = flag.replace("origin_", prefix + "_", 1)
        selected = selected.withColumn(flag, F.coalesce(F.col(flag), F.lit(1)))
    for flag in INDICATOR_FLAGS:
        flag = flag.replace("origin_", prefix + "_", 1)
        selected = selected.withColumn(flag, F.coalesce(F.col(flag), F.lit(0)))
    unavailable = {
        (entry["station_id"], entry["year"])
        for entry in manifest["files"].values()
        if entry["status"] == "unavailable"
    }

    if unavailable:
        def source_missing(station, cutoff, zone):
            if not station or not cutoff or not zone:
                return False
            cutoff = datetime.fromisoformat(cutoff)
            offset = timedelta(seconds=standard_offset(zone, cutoff.year))
            start = (min(cutoff, cutoff + offset) - timedelta(minutes=max_age_minutes)).year
            end = max(cutoff, cutoff + offset).year
            return any((station, year) in unavailable for year in range(start, end + 1))

        has_missing_source = F.udf(source_missing, T.BooleanType())
        station_col = f"{prefix}_station_id"
        zone_col = f"{prefix}_time_zone"
        selected = selected.withColumn(
            "_cutoff_str", F.date_format("scheduled_departure_utc", "yyyy-MM-dd'T'HH:mm:ss")
        )
        missing_keys = (
            selected.filter(
                F.col(observation).isNull()
                & F.col("scheduled_departure_utc").isNotNull()
                & F.col(station_col).isNotNull()
            )
            .select(station_col, "_cutoff_str", zone_col)
            .distinct()
            .withColumn("_source_missing", has_missing_source(station_col, "_cutoff_str", zone_col))
            .select(
                F.col(station_col).alias("_m_station"),
                F.col("_cutoff_str").alias("_m_cutoff"),
                F.col(zone_col).alias("_m_zone"),
                "_source_missing",
            )
        )
        selected = selected.join(
            F.broadcast(missing_keys),
            selected[station_col].eqNullSafe(missing_keys._m_station)
            & selected["_cutoff_str"].eqNullSafe(missing_keys._m_cutoff)
            & selected[zone_col].eqNullSafe(missing_keys._m_zone),
            "left",
        )
        missing_source_col = F.coalesce(F.col("_source_missing"), F.lit(False))
    else:
        missing_source_col = F.lit(False)

    selected = selected.withColumn(
        f"{prefix}_weather_match_status",
        F.when(F.col(observation).isNotNull(), "matched")
        .when(F.col("scheduled_departure_utc").isNull(), "invalid_cutoff")
        .when(F.col(f"{prefix}_station_id").isNull(), "unmapped_airport")
        .when(missing_source_col, "source_unavailable")
        .otherwise("no_eligible_report"),
    )
    if unavailable:
        selected = selected.drop("_cutoff_str", "_m_station", "_m_cutoff", "_m_zone", "_source_missing")
    return selected


def validate_cohort(
    original: DataFrame,
    enriched: DataFrame,
    max_age_minutes: int = 120,
) -> None:
    """Verify that weather enrichment preserved flight identities, count, and predictor values."""
    for frame in (original, enriched):
        stats = frame.agg(
            F.count("*").alias("rows"), F.countDistinct("flight_id").alias("ids")
        ).first()
        if stats.rows != stats.ids:
            raise ValueError("Flight IDs must be nonnull and unique.")

    columns = original.columns
    # Unique IDs allow a single-pass merge-join instead of two full-table exceptAll shuffles.
    parent_rows = original.select("flight_id", F.struct(*columns).alias("_parent"))
    enriched_rows = enriched.select("flight_id", F.struct(*columns).alias("_enriched"))
    if (
        parent_rows.hint("merge")
        .join(enriched_rows, "flight_id", "full")
        .filter(~F.col("_parent").eqNullSafe(F.col("_enriched")))
        .limit(1)
        .count()
    ):
        raise ValueError("Weather enrichment changed the BTS cohort or predictors.")

    # Guard against future observations leaking into features before departure.
    if (
        enriched.filter("origin_available_utc > scheduled_departure_utc")
        .limit(1)
        .count()
    ):
        raise ValueError("Future weather selected.")

    # Confirm weather observation age falls strictly within configured policy bounds.
    if (
        enriched.filter(
            (F.col("origin_weather_age_minutes") > max_age_minutes)
            | (F.col("origin_weather_age_minutes") < 0)
        )
        .limit(1)
        .count()
    ):
        raise ValueError("Selected weather violates age bounds.")


def main(
    publish_results: bool = True,
    spark: Optional[SparkSession] = None,
    bts_files: Optional[dict[str, Any]] = None,
) -> tuple[Path, Path]:
    """Execute end-to-end origin weather enrichment pipeline on BTS Gold cohorts."""
    from .clean import clean_weather
    from .download import download_weather, prepare_references

    settings, policy, config = load_config()
    cached = find_gold(ROOT, settings, policy, config, bts_files=bts_files)
    if cached is not None:
        if publish_results:
            update_results({"bts_origin_weather": {"data": json.loads((cached / "_metadata.json").read_text())}})
        return cached, cached
    owns_spark = spark is None
    if owns_spark:
        spark = get_spark_session("origin_weather_features")
    try:
        flights = spark.read.parquet(
            *[str(active(ROOT, "gold") / name) for name in config["splits"]]
        )
        with progress_stage("Origin weather: prepare airport references and acquire files"):
            origins = [row.Origin for row in flights.select("Origin").distinct().collect()]
            prepare_references(
                origins,
                settings["reference_directory"],
                config["splits"]["train"]["start_date"],
                config["splits"]["test"]["end_date"],
            )
            refs = load_references(settings["reference_directory"])
            cutoffs = build_flight_cutoffs(spark, flights, refs).cache()
            requests = required_station_years(cutoffs, policy["max_age_minutes"])
            manifest = download_weather(requests, settings["output_directory"])
            scope = sorted({f"{year}/LCD_{station}_{year}.csv" for station, year, _ in requests})
            manifest = scoped_manifest(manifest, scope)
        cache_inputs = input_identity(ROOT, settings, policy, config, scope=scope, bts_files=bts_files)
        build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        gold = ROOT / "data/gold/bts_origin_weather" / build_id
        report = ROOT / "data/silver/weather" / build_id
        with progress_stage("Origin weather: clean observations and check source clocks"):
            observations = clean_weather(
                spark,
                settings["output_directory"],
                ROOT / "data/silver/weather" / build_id,
                ROOT / "data/quarantine/weather" / build_id,
                manifest,
                report,
            )
        with progress_stage("Origin weather: match observations and validate the flight cohort"):
            enriched = build_origin_features(
                cutoffs, observations, manifest, **policy
            ).cache()
            validate_cohort(flights, enriched, **policy)
        with progress_stage("Origin weather: write splits, coverage and metadata"):
            coverage = enriched.groupBy(
                "split",
                "Origin",
                F.trunc("FlightDate", "month").alias("month"),
                "origin_weather_match_status",
            ).agg(
                F.count("*").alias("flights"),
                F.avg("origin_weather_age_minutes").alias("mean_age_minutes"),
                *[F.sum(name).alias(name) for name in WEATHER_FLAGS],
            )
            for split in config["splits"]:
                enriched.filter(F.col("split") == split).write.parquet(str(gold / split))
            write_json(
                gold / "_metadata.json",
                {
                    **json.loads((report / "_metadata.json").read_text()),
                    "coverage": records(coverage.toPandas()),
                    "dataset_id": build_id,
                    "gold_dir": str(gold),
                    "policy": policy,
                    "splits": config["splits"],
                    "weather_columns": WEATHER_NUMERIC + WEATHER_FLAGS,
                    "reference_sources": json.loads(
                        (settings["reference_directory"] / "sources.json").read_text()
                    ),
                    "source_manifest": manifest,
                    "bts_input_files": bts_fingerprints(),
                },
            )
            save_gold(gold, cache_inputs)
            if publish_results:
                update_results({"bts_origin_weather": {"data": json.loads((gold / "_metadata.json").read_text())}})
        return gold, gold
    finally:
        spark.catalog.clearCache()
        if owns_spark:
            spark.stop()


if __name__ == "__main__":
    gold, report = main()
    print(f"Enriched flights: {gold}\nCoverage: {report}")
