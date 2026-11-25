"""Synthetic destination contracts; real test-period labels are never read."""

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pyspark.sql import functions as F

from tests.conftest import get_test_spark_session
from scripts.common.io import sha256, write_json
from scripts.destination_weather import build_features as destination
from scripts.origin_weather import build_features as origin
from scripts.origin_weather.measurements import FEATURE_COLUMNS, NUMERIC_FEATURES, SOURCE_FIELDS, decode_weather


class DestinationTests(unittest.TestCase):

    def test_parent_reference_accepts_zip_line_endings_and_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "data/gold/bts_origin_weather/fixture"
            for split in ("train", "val", "test"):
                (parent / split).mkdir(parents=True)
                (parent / split / "_SUCCESS").touch()
            reference = root / "references"
            reference.mkdir()
            target = reference / "airports.csv"
            target.write_bytes(b"Origin\r\nORD\r\n")
            sources = {"files": {"airports.csv": sha256(target)}}
            write_json(reference / "sources.json", sources)
            metadata = {
                "policy": {}, "splits": dict.fromkeys(("train", "val", "test")),
                "weather_columns": origin.WEATHER_NUMERIC + origin.WEATHER_FLAGS,
                "bts_input_files": [], "reference_sources": sources,
                "source_contract": {"timestamp_basis": "metar_verified_utc_or_local_standard"},
            }
            write_json(parent / "_metadata.json", metadata)
            target.write_bytes(b"Origin\nORD\n")
            with patch.object(destination, "ROOT", root), patch.object(destination, "bts_fingerprints", return_value=[]):
                self.assertEqual(destination.validate_parent(parent, {"reference_directory": reference}, {}, metadata), metadata)
                target.write_bytes(b"Origin\nLAX\n")
                with self.assertRaisesRegex(ValueError, "Origin reference changed: airports.csv"):
                    destination.validate_parent(parent, {"reference_directory": reference}, {}, metadata)


class DestinationSparkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark_session("destination_contract")

    def references(self):
        airports = [{"Origin": "LAX", "time_zone": "America/Los_Angeles", "valid_from": "2023-01-01", "valid_to": "2026-01-01"}]
        mappings = [{"Origin": "LAX", "station_id": "USW00023174", "method": "exact", "valid_from": "2023-01-01", "valid_to": "2026-01-01"}]
        return airports, mappings

    def observations(self, instants):
        tokens = {name: "10" for name in SOURCE_FIELDS}
        tokens.update(HourlyPresentWeatherType="AU:RA SN", HourlySkyConditions="BKN:02 10", HourlyPrecipitation="T")
        decoded = decode_weather(tokens)
        schema = "station_id string, observation_utc timestamp, source_file string, REM string, " + ", ".join(f"{name} {'double' if name in NUMERIC_FEATURES else 'int'}" for name in FEATURE_COLUMNS)
        return self.spark.createDataFrame([("USW00023174", instant, source, "raw report", *[decoded[name] for name in FEATURE_COLUMNS]) for instant, source in instants], schema)

    def test_effective_local_dates_and_unresolved_mappings(self):
        flights = self.spark.createDataFrame([
            ("west", "LAX", date(2024, 1, 1), datetime(2024, 1, 1, 1, tzinfo=timezone.utc)),
            ("transition", "LAX", date(2024, 1, 1), datetime(2024, 1, 1, 8, 30, tzinfo=timezone.utc)),
            ("east", "TKO", date(2023, 12, 31), datetime(2023, 12, 31, 20, tzinfo=timezone.utc)),
            ("invalid", "LAX", date(2024, 1, 1), None),
            ("missing", "XXX", date(2024, 1, 1), datetime(2024, 1, 1, 1)),
            ("badzone", "BAD", date(2024, 1, 1), datetime(2024, 1, 1, 1)),
        ], "flight_id string, Dest string, FlightDate date, scheduled_departure_utc timestamp")
        airports, mappings = self.references()
        mappings[0]["valid_to"] = "2024-01-01"
        for code, zone in (("TKO", "Asia/Tokyo"), ("BAD", "Invalid/Zone")):
            airports.append({**airports[0], "Origin": code, "time_zone": zone})
            mappings.append({**mappings[0], "Origin": code, "valid_from": "2024-01-01", "valid_to": "2026-01-01"})
        rows = {row.flight_id: row for row in destination.build_destination_cutoffs(self.spark, flights, (airports, mappings)).collect()}
        self.assertEqual(rows["west"].destination_station_id, "USW00023174")
        self.assertEqual(rows["east"].destination_time_zone, "Asia/Tokyo")
        for key in ("invalid", "missing", "badzone", "transition"):
            self.assertIsNone(rows[key].destination_station_id)
        self.assertEqual(rows["west"].scheduled_departure_utc, flights.filter("flight_id = 'west'").first().scheduled_departure_utc)
        ambiguous = [*airports, {**airports[0], "time_zone": "Asia/Tokyo"}]
        row = destination.build_destination_cutoffs(self.spark, flights.filter("flight_id = 'west'"), (ambiguous, mappings)).first()
        self.assertIsNone(row.destination_station_id)

    def test_bounds_missing_flags_and_parent_equality(self):
        instant = datetime(2024, 1, 1, 18)
        ages = (0, 14, 15, 120, 121, -1)
        flight_rows = [
            (f"ok_{age}", "LAX", instant + timedelta(minutes=age), "keep origin", 123.0)
            for age in ages
        ] + [
            ("unmapped", "XXX", instant, "keep missing", 456.0),
            ("null_destination", None, instant, "keep null destination", 456.0),
            ("invalid", "LAX", None, "keep invalid", 789.0),
        ]
        flights = self.spark.createDataFrame(
            flight_rows,
            "flight_id string, Dest string, scheduled_departure_utc timestamp, source_file string, origin_temperature_c double"
        )
        cutoffs = destination.build_destination_cutoffs(self.spark, flights, self.references())
        observations = self.observations([(instant, "selected")])
        enriched = destination.build_destination_features(cutoffs, observations, {"files": {}}).cache()
        try:
            plan = enriched._jdf.queryExecution().executedPlan().toString()
            self.assertIn("SortMergeJoin", plan)
            self.assertNotIn("BroadcastHashJoin", plan)
            self.assertNotIn("BroadcastNestedLoopJoin", plan)
            rows = {row.flight_id: row for row in enriched.collect()}
            for age in ages:
                row = rows[f"ok_{age}"]
                matched = 0 <= age <= 120
                self.assertEqual(row.destination_weather_missing, int(not matched))
                self.assertEqual(row.source_file, "keep origin")
                self.assertEqual(row.origin_temperature_c, 123)
                if matched:
                    self.assertEqual(row.destination_weather_age_minutes, age)
                    self.assertLessEqual(row.destination_available_utc, instant + timedelta(minutes=age))
                else:
                    self.assertTrue(all(row[name] is None for name in destination.DESTINATION_WEATHER_NUMERIC))
                    self.assertTrue(all(row[name] == 1 for name in destination.DESTINATION_WEATHER_FLAGS[:13]))
                    self.assertTrue(all(row[name] == 0 for name in destination.DESTINATION_WEATHER_FLAGS[13:]))
            self.assertEqual(rows["invalid"].destination_weather_match_status, "invalid_cutoff")
            self.assertEqual(rows["unmapped"].destination_weather_match_status, "unmapped_airport")
            self.assertEqual(rows["null_destination"].destination_weather_match_status, "unmapped_airport")
            destination.validate_destination_cohort(flights, enriched)
            for changed in (
                enriched.filter("flight_id != 'invalid'"),
                enriched.withColumn("flight_id", F.concat(F.lit("other_"), F.col("flight_id"))),
                enriched.withColumn("source_file", F.lit(None).cast("string")),
            ):
                with self.subTest(change=changed.columns), self.assertRaisesRegex(ValueError, "parent columns"):
                    destination.validate_destination_cohort(flights, changed)
            with self.assertRaisesRegex(ValueError, "nonnull and unique"):
                destination.validate_destination_cohort(flights, enriched.unionByName(enriched))
            with self.assertRaisesRegex(ValueError, "parent columns"):
                destination.validate_destination_cohort(flights, enriched.withColumn("origin_temperature_c", F.lit(999.0)))
            with self.assertRaisesRegex(ValueError, "observation"):
                destination.validate_destination_cohort(flights, enriched.withColumn(
                    "destination_observation_utc", F.col("scheduled_departure_utc") + F.expr("INTERVAL 1 MINUTE")
                ))
        finally:
            enriched.unpersist()
