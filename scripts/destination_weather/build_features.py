"""Destination observations at the existing origin-derived departure instant."""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pyspark import StorageLevel
from pyspark.sql import functions as F

from scripts.origin_weather import build_features as origin
from scripts.origin_weather.clean import clean_weather
from scripts.origin_weather.download import (
    ROOT, load_config, prepare_references, project_path, download_weather,
    reference_matches, sha256, write_json,
)
from scripts.results import records

DESTINATION_WEATHER_NUMERIC = [name.replace("origin_", "destination_", 1) for name in origin.WEATHER_NUMERIC]
DESTINATION_WEATHER_FLAGS = [name.replace("origin_", "destination_", 1) for name in origin.WEATHER_FLAGS]

bts_fingerprints = origin.bts_fingerprints


def validate_parent(directory, settings, policy, config):
    """Reject stale parent data without hashing raw weather or inspecting test labels."""
    directory = Path(directory).resolve()
    if not directory.is_relative_to((ROOT / "data/gold/bts_origin_weather").resolve()):
        raise ValueError("Parent Gold must be inside the origin-weather data directory.")
    metadata = json.loads((directory / "_metadata.json").read_text())
    if (metadata.get("policy") != policy or metadata.get("splits") != config["splits"]
            or metadata.get("weather_columns") != origin.WEATHER_NUMERIC + origin.WEATHER_FLAGS
            or metadata.get("bts_input_files") != bts_fingerprints()
            or metadata.get("source_contract", {}).get("timestamp_basis") != "metar_verified_utc_or_local_standard"
            or not all((directory / split / "_SUCCESS").exists() for split in config["splits"])):
        raise ValueError("Origin Gold schema, policy, inputs or timestamp contract changed.")
    sources = metadata["reference_sources"]
    current = json.loads((settings["reference_directory"] / "sources.json").read_text())
    if sources != current:
        raise ValueError("Origin reference identity changed.")
    for name, checksum in sources["files"].items():
        if not reference_matches(settings["reference_directory"] / name, checksum):
            raise ValueError(f"Origin reference changed: {name}")
    return metadata


def validate_parent_data(spark, directory, config):
    expected = {name: "double" for name in origin.WEATHER_NUMERIC}
    expected.update({name: "int" for name in origin.WEATHER_FLAGS})
    expected["scheduled_departure_utc"] = "timestamp"
    for split in config["splits"]:
        frame = spark.read.parquet(str(Path(directory) / split))
        actual = dict(frame.dtypes)
        if any(actual.get(name) != dtype for name, dtype in expected.items()):
            raise ValueError(f"Parent Gold has an outdated schema in {split}.")
        bts = spark.read.parquet(str(ROOT / "data/gold/bts" / split))
        if frame.select(bts.columns).schema != bts.schema:
            raise ValueError(f"Parent BTS column types changed in {split}.")
        origin.validate_cohort(bts, frame, **config["origin_weather"])


def build_destination_cutoffs(spark, flights, references):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    airports, mappings = references
    by_airport, by_mapping = {}, {}
    for row in airports:
        by_airport.setdefault(row["Origin"], []).append(row)
    for row in mappings:
        by_mapping.setdefault(row["Origin"], []).append(row)

    def resolve(code, instant):
        if instant is None:
            return None, None, None
        instant = datetime.fromisoformat(instant).replace(tzinfo=timezone.utc)
        candidates = []
        for airport in by_airport.get(code, []):
            try:
                local_date = instant.astimezone(ZoneInfo(airport["time_zone"])).date().isoformat()
            except (ZoneInfoNotFoundError, ValueError):
                continue
            if airport["valid_from"] <= local_date < airport["valid_to"]:
                candidates.append((airport["time_zone"], local_date))
        if len(candidates) != 1:
            return None, None, None
        zone, local_date = candidates[0]
        valid_mappings = [
            row for row in by_mapping.get(code, [])
            if row["station_id"] and row["valid_from"] <= local_date < row["valid_to"]
        ]
        if len(valid_mappings) != 1:
            return None, None, None
        return valid_mappings[0]["station_id"], zone, valid_mappings[0]["method"]

    lookup = F.udf(
        resolve,
        "destination_station_id string, destination_time_zone string, destination_mapping_method string",
    )
    scheduled_utc_formatted = F.date_format("scheduled_departure_utc", "yyyy-MM-dd'T'HH:mm:ss")
    return (
        flights.withColumn(
            "_destination_mapping", lookup("Dest", scheduled_utc_formatted)
        )
        .select(*flights.columns, "_destination_mapping.*")
    )


def build_destination_features(cutoffs, observations, manifest, **policy):
    return origin.build_weather_features(cutoffs, observations, manifest, prefix="destination", **policy)


def validate_destination_cohort(original, enriched, max_age_minutes=120, publication_lag_minutes=0):
    if original.schema != enriched.select(original.columns).schema:
        raise ValueError("Destination enrichment changed parent columns or types.")
    for frame in (original, enriched):
        stats = frame.agg(F.count("*").alias("rows"), F.countDistinct("flight_id").alias("ids")).first()
        if stats.rows != stats.ids:
            raise ValueError("Flight IDs must be nonnull and unique.")
    if (original.exceptAll(enriched.select(original.columns)).limit(1).count()
            or enriched.select(original.columns).exceptAll(original).limit(1).count()):
        raise ValueError("Destination enrichment changed parent columns or cohort.")
    invalid = F.lit(False)
    for name in DESTINATION_WEATHER_FLAGS:
        invalid = invalid | F.col(name).isNull() | ~F.col(name).isin(0, 1)
    for name in DESTINATION_WEATHER_NUMERIC:
        invalid = invalid | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    if enriched.filter(invalid).limit(1).count():
        raise ValueError("Invalid destination numeric values or flags.")
    observation = F.col("destination_observation_utc")
    cutoff = F.col("scheduled_departure_utc")
    age = (cutoff.cast("long") - observation.cast("long")) / 60
    available = observation + F.expr(f"INTERVAL {publication_lag_minutes} MINUTES")
    invalid_report = observation.isNotNull() & (
        cutoff.isNull() | F.col("destination_station_id").isNull()
        | (observation < cutoff - F.expr(f"INTERVAL {max_age_minutes} MINUTES"))
        | (observation > cutoff - F.expr(f"INTERVAL {publication_lag_minutes} MINUTES"))
        | ~F.col("destination_weather_age_minutes").eqNullSafe(age)
        | ~F.col("destination_available_utc").eqNullSafe(available)
        | (F.col("destination_weather_missing") != 0)
    )
    invalid_missing = F.col("destination_available_utc").isNotNull()
    for name in DESTINATION_WEATHER_NUMERIC:
        invalid_missing = invalid_missing | F.col(name).isNotNull()
    for name in DESTINATION_WEATHER_FLAGS:
        expected = 1 if name.endswith("_missing") else 0
        invalid_missing = invalid_missing | (F.col(name) != expected)
    if enriched.filter(invalid_report | (observation.isNull() & invalid_missing)).limit(1).count():
        raise ValueError("Destination observation or missing-report semantics violate the cutoff contract.")


def main(origin_gold_dir):
    from scripts.bts.build_features import get_spark_session

    settings, policy, config = load_config()
    parent = validate_parent(origin_gold_dir, settings, policy, config)
    reference = project_path(ROOT, config["destination_weather"]["reference_directory"])
    if reference == settings["reference_directory"]:
        raise ValueError("Destination references must have a separate scope.")
    spark = get_spark_session("destination_weather_features")
    try:
        flights = spark.read.parquet(*[str(Path(origin_gold_dir) / split) for split in config["splits"]])
        expected = set(origin.WEATHER_NUMERIC + origin.WEATHER_FLAGS + ["scheduled_departure_utc"])
        if not expected.issubset(flights.columns):
            raise ValueError("Parent Gold lacks the current origin feature schema.")
        bts = spark.read.parquet(*[str(ROOT / "data/gold/bts" / split) for split in config["splits"]])
        origin.validate_cohort(bts, flights, **policy)
        dates = flights.agg(
            F.date_format(F.min("scheduled_departure_utc"), "yyyy-MM-dd").alias("start"),
            F.date_format(F.max("scheduled_departure_utc"), "yyyy-MM-dd").alias("end"),
        ).first()
        # A one-day envelope includes every destination civil date, including date-line crossings.
        if dates.start is None or dates.end is None:
            raise ValueError("No valid departure instants in the parent cohort.")
        start = min(
            date.fromisoformat(config["splits"]["train"]["start_date"]),
            date.fromisoformat(dates.start) - timedelta(days=1),
        )
        end = max(
            date.fromisoformat(config["splits"]["test"]["end_date"]),
            date.fromisoformat(dates.end) + timedelta(days=1),
        )
        destinations = [
            row.Dest
            for row in flights.select("Dest").where(F.col("Dest").isNotNull()).distinct().collect()
        ]
        prepare_references(
            destinations, reference, start.isoformat(), end.isoformat(), prefix="destination"
        )
        cutoffs = build_destination_cutoffs(
            spark, flights, origin.load_references(reference, prefix="destination")
        )
        requests = origin.required_station_years(cutoffs, policy["max_age_minutes"], "destination")
        # Combine destination requests with origin requests to ensure unified cache coverage in single fetch pass.
        requests += origin.required_station_years(flights, policy["max_age_minutes"])
        manifest = download_weather(requests, settings["output_directory"])
        build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        gold = ROOT / "data/gold/bts_origin_destination_weather" / build_id
        silver = ROOT / "data/silver/weather" / build_id
        observations = clean_weather(
            spark,
            settings["output_directory"],
            silver,
            ROOT / "data/quarantine/weather" / build_id,
            manifest,
            silver,
        )
        enriched = build_destination_features(cutoffs, observations, manifest, **policy).persist(
            StorageLevel.DISK_ONLY
        )
        validate_destination_cohort(flights, enriched, **policy)
        coverage = {}
        for role, airport in (("origin", "Origin"), ("destination", "Dest")):
            flags = origin.WEATHER_FLAGS if role == "origin" else DESTINATION_WEATHER_FLAGS
            coverage_df = enriched.groupBy(
                "split",
                airport,
                F.trunc("FlightDate", "month").alias("month"),
                f"{role}_weather_match_status",
            ).agg(
                F.count("*").alias("flights"),
                F.avg(f"{role}_weather_age_minutes").alias("mean_age_minutes"),
                *[F.sum(name).alias(name) for name in flags],
            )
            coverage[role] = records(coverage_df.toPandas())
        for split in config["splits"]:
            count = enriched.filter(F.col("split") == split).count()
            if any(sum(row["flights"] for row in rows if row["split"] == split) != count for rows in coverage.values()):
                raise ValueError(f"Coverage does not reconcile: {split}")
            enriched.filter(F.col("split") == split).write.parquet(str(gold / split))
        metadata = {
            **json.loads((silver / "_metadata.json").read_text()),
            "dataset_id": build_id,
            "gold_dir": str(gold),
            "parent_gold_dir": str(origin_gold_dir),
            "parent_metadata_sha256": sha256(Path(origin_gold_dir) / "_metadata.json"),
            "policy": policy,
            "splits": config["splits"],
            "coverage": coverage,
            "origin_reference_sources": parent["reference_sources"],
            "destination_reference_sources": json.loads((reference / "sources.json").read_text()),
            "reference_legacy_keys": "Origin and sources.origins identify destination airports in the destination reference version.",
            "source_manifest": manifest,
            "bts_input_files": parent["bts_input_files"],
            "weather_columns": (
                origin.WEATHER_NUMERIC
                + origin.WEATHER_FLAGS
                + DESTINATION_WEATHER_NUMERIC
                + DESTINATION_WEATHER_FLAGS
            ),
            "schema": enriched.schema.jsonValue(),
            "availability_assumption": "Observation time plus lag; historical publication and metadata availability unverified. Conditions at departure, not an arrival forecast.",
            "raw_integrity": "Cached CSVs reused without content scans or checksums; cleaning checks records, not historical immutability.",
            "source_files": [
                {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size}
                for path in sorted(settings["output_directory"].rglob("*.csv"))
            ],
        }
        write_json(gold / "_metadata.json", metadata)
        return gold, gold
    finally:
        spark.catalog.clearCache()
        spark.stop()
