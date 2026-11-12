"""Attach the latest eligible origin observation without changing the BTS cohort."""

import csv
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from pyspark.sql import functions as F, types as T, Window

from scripts.results import records, update_results

from .download import ROOT, load_config, standard_offset, write_json
from .measurements import NUMERIC_FEATURES, MISSING_FLAGS, INDICATOR_FLAGS, FEATURE_COLUMNS

WEATHER_NUMERIC = [*NUMERIC_FEATURES, "origin_weather_age_minutes"]
WEATHER_FLAGS = ["origin_weather_missing", *MISSING_FLAGS, *INDICATOR_FLAGS]


def load_references(directory):
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
                previous["Origin"] == current["Origin"]
                and previous["valid_to"] > current["valid_from"]
            ):
                raise ValueError(
                    f"Overlapping reference intervals: {current['Origin']}"
                )
    for row in airports:
        ZoneInfo(row["time_zone"])
    for row in mappings:
        if not row["station_id"] or not any(
            a["Origin"] == row["Origin"]
            and a["valid_from"] <= row["valid_from"]
            and a["valid_to"] >= row["valid_to"]
            for a in airports
        ):
            raise ValueError("Station mapping lacks a valid airport interval.")
    return airports, mappings


def local_cutoff(flight_date, clock, zone):
    if flight_date is None or clock is None or not zone:
        return None, "invalid_cutoff"
    spelling = str(clock).strip()
    if not spelling.isdigit() or len(spelling) > 4:
        return None, "invalid_cutoff"
    hours, minutes = divmod(int(spelling), 100)
    if not (0 <= hours < 24 and 0 <= minutes < 60) and (hours, minutes) != (24, 0):
        return None, "invalid_cutoff"
    local = datetime.combine(flight_date, datetime.min.time()) + timedelta(
        hours=hours, minutes=minutes
    )
    zone = ZoneInfo(zone)
    candidates = set()
    for fold in (0, 1):
        utc = local.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if utc.astimezone(zone).replace(tzinfo=None) == local:
            candidates.add(utc)
    # A DST gap or repeated clock has no uniquely defensible departure instant.
    if len(candidates) != 1:
        return None, "invalid_cutoff"
    return candidates.pop().replace(tzinfo=None), "valid"


def build_flight_cutoffs(spark, flights, references):
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
    return (
        joined.withColumn(
            "cutoff", convert("FlightDate", "CRSDepTime", "origin_time_zone")
        )
        .select(
            *flights.columns,
            "origin_time_zone",
            "origin_station_id",
            "origin_mapping_method",
            "cutoff.*",
        )
        .withColumn("scheduled_departure_utc", F.to_timestamp("utc_clock"))
        .drop("utc_clock")
    )


def required_station_years(cutoffs, max_age_minutes):
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
    if not requests:
        raise ValueError("No mapped origin cutoffs; cannot train a weather model.")
    return requests


def build_origin_features(
    cutoffs, observations, manifest, max_age_minutes=120, publication_lag_minutes=0
):
    if not 0 <= publication_lag_minutes <= max_age_minutes or max_age_minutes <= 0:
        raise ValueError("Invalid weather age/lag.")
    missing = set(FEATURE_COLUMNS) - set(observations.columns)
    if missing:
        raise ValueError("Weather features are outdated; rebuild origin-weather Silver with all 11 fields.")
    weather = observations.withColumn(
        "weather_hour", F.floor(F.col("observation_utc").cast("long") / 3600)
    ).alias("weather")
    # Hour buckets bound candidate expansion before the exact timestamp checks.
    flights = cutoffs.withColumn(
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
                F.floor(
                    (
                        F.col("scheduled_departure_utc").cast("long")
                        - publication_lag_minutes * 60
                    )
                    / 3600
                ).cast("long"),
            )
        ),
    ).alias("flights")
    condition = (
        (F.col("flights.origin_station_id") == F.col("weather.station_id"))
        & (F.col("match_hour") == F.col("weather_hour"))
        & (
            F.col("weather.observation_utc")
            >= F.col("flights.scheduled_departure_utc")
            - F.expr(f"INTERVAL {max_age_minutes} MINUTES")
        )
        & (
            F.col("weather.observation_utc")
            <= F.col("flights.scheduled_departure_utc")
            - F.expr(f"INTERVAL {publication_lag_minutes} MINUTES")
        )
    )
    selected = (
        flights.join(weather, condition, "left")
        .select(
            "flights.*",
            "weather.observation_utc",
            "weather.source_file",
            *["weather." + name for name in FEATURE_COLUMNS],
            *[
                F.col("weather." + name).alias(output)
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
                    F.col("observation_utc").desc_nulls_last(),
                    F.col("source_file").asc_nulls_last(),
                )
            ),
        )
    )
    selected = selected.filter("rank = 1").drop("rank", "match_hour")
    selected = (
        selected.withColumn(
            "origin_weather_age_minutes",
            (
                F.col("scheduled_departure_utc").cast("long")
                - F.col("observation_utc").cast("long")
            )
            / 60,
        )
        .withColumn(
            "origin_available_utc",
            F.col("observation_utc")
            + F.expr(f"INTERVAL {publication_lag_minutes} MINUTES"),
        )
        .withColumn(
            "origin_weather_missing", F.col("observation_utc").isNull().cast("int")
        )
    )
    for flag in MISSING_FLAGS:
        selected = selected.withColumn(flag, F.coalesce(F.col(flag), F.lit(1)))
    for flag in INDICATOR_FLAGS:
        selected = selected.withColumn(flag, F.coalesce(F.col(flag), F.lit(0)))
    unavailable = {
        (entry["station_id"], entry["year"])
        for entry in manifest["files"].values()
        if entry["status"] == "unavailable"
    }

    def source_missing(station, cutoff, zone):
        if not station or not cutoff or not zone:
            return False
        cutoff = datetime.fromisoformat(cutoff)
        offset = timedelta(seconds=standard_offset(zone, cutoff.year))
        start = (min(cutoff, cutoff + offset) - timedelta(minutes=max_age_minutes)).year
        end = max(cutoff, cutoff + offset).year
        return any((station, year) in unavailable for year in range(start, end + 1))

    has_missing_source = F.udf(source_missing, T.BooleanType())
    selected = selected.withColumn(
        "origin_weather_match_status",
        F.when(F.col("observation_utc").isNotNull(), "matched")
        .when(F.col("scheduled_departure_utc").isNull(), "invalid_cutoff")
        .when(F.col("origin_station_id").isNull(), "unmapped_airport")
        .when(
            has_missing_source(
                "origin_station_id",
                F.date_format("scheduled_departure_utc", "yyyy-MM-dd'T'HH:mm:ss"),
                "origin_time_zone",
            ),
            "source_unavailable",
        )
        .otherwise("no_eligible_report"),
    )
    return selected


def validate_cohort(
    original, enriched, max_age_minutes=120, publication_lag_minutes=0
):
    for frame in (original, enriched):
        stats = frame.agg(
            F.count("*").alias("rows"), F.countDistinct("flight_id").alias("ids")
        ).first()
        if stats.rows != stats.ids:
            raise ValueError("Flight IDs must be nonnull and unique.")
    columns = original.columns
    if (
        original.select(columns).exceptAll(enriched.select(columns)).limit(1).count()
        or enriched.select(columns).exceptAll(original.select(columns)).limit(1).count()
    ):
        raise ValueError("Weather enrichment changed the BTS cohort or predictors.")
    if (
        enriched.filter("origin_available_utc > scheduled_departure_utc")
        .limit(1)
        .count()
    ):
        raise ValueError("Future weather selected.")
    if (
        enriched.filter(
            (F.col("origin_weather_age_minutes") > max_age_minutes)
            | (F.col("origin_weather_age_minutes") < publication_lag_minutes)
        )
        .limit(1)
        .count()
    ):
        raise ValueError("Selected weather violates age/lag bounds.")


def main():
    from scripts.bts.build_features import get_spark_session
    from .clean import clean_weather
    from .download import download_weather, prepare_references

    settings, policy, config = load_config()
    spark = get_spark_session("origin_weather_features")
    try:
        flights = spark.read.parquet(
            *[str(ROOT / "data/gold/bts" / name) for name in config["splits"]]
        )
        origins = [row.Origin for row in flights.select("Origin").distinct().collect()]
        prepare_references(
            origins,
            settings["reference_directory"],
            config["splits"]["train"]["start_date"],
            config["splits"]["test"]["end_date"],
        )
        refs = load_references(settings["reference_directory"])
        cutoffs = build_flight_cutoffs(spark, flights, refs).cache()
        manifest = download_weather(
            required_station_years(cutoffs, policy["max_age_minutes"]),
            settings["output_directory"],
        )
        build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        gold = ROOT / "data/gold/bts_origin_weather" / build_id
        report = ROOT / "data/silver/weather" / build_id
        observations = clean_weather(
            spark,
            settings["output_directory"],
            ROOT / "data/silver/weather" / build_id,
            ROOT / "data/quarantine/weather" / build_id,
            manifest,
            report,
        )
        enriched = build_origin_features(
            cutoffs, observations, manifest, **policy
        ).cache()
        validate_cohort(flights, enriched, **policy)
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
                "reference_sources": json.loads(
                    (settings["reference_directory"] / "sources.json").read_text()
                ),
                "source_manifest": manifest,
                "bts_input_files": [
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "modified_ns": path.stat().st_mtime_ns,
                    }
                    for path in sorted((ROOT / "data/gold/bts").rglob("*.parquet"))
                ],
            },
        )
        update_results({"bts_origin_weather": {"data": json.loads((gold / "_metadata.json").read_text())}})
        return gold, gold
    finally:
        spark.catalog.clearCache()
        spark.stop()


if __name__ == "__main__":
    gold, report = main()
    print(f"Enriched flights: {gold}\nCoverage: {report}")
