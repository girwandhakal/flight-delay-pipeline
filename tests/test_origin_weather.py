"""Exercise weather timing, acquisition, cohort preservation and shared training."""

import csv
from datetime import date, datetime, timezone
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from pyspark.sql import functions as F

from tests.conftest import get_test_spark_session

from scripts.common import io as io_utils
from scripts.common.config import project_path
from scripts.common.io import sha256, write_json
from scripts.origin_weather import build_features as features, clean, download
from scripts.origin_weather.measurements import SOURCE_FIELDS, FEATURE_COLUMNS, NUMERIC_FEATURES, decode_weather


HEADER = ["STATION", "DATE", "REPORT_TYPE", "REM", *SOURCE_FIELDS]


def raw_csv(rows):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(HEADER)
    writer.writerows(list(row) + [""] * (len(HEADER) - len(row)) for row in rows)
    return stream.getvalue().encode()


@pytest.mark.unit
class WeatherDownloadTests(unittest.TestCase):
    def test_reference_line_endings_preserve_checksums_but_content_changes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "airports.csv"
            for original, converted in ((b"Origin\r\nORD\r\n", b"Origin\nORD\n"),
                                        (b"Origin\nORD\n", b"Origin\r\nORD\r\n")):
                with self.subTest(original=original):
                    target.write_bytes(original)
                    checksum = sha256(target)
                    write_json(directory / "sources.json", {
                        "origins": ["ORD"], "scope": ["2024-01-01", "2025-01-31"],
                        "files": {"airports.csv": checksum},
                    })
                    with patch.object(download, "fetch") as network:
                        for payload in (original, converted):
                            target.write_bytes(payload)
                            download.prepare_references(["ORD"], directory, "2024-01-01", "2025-01-31")
                            self.assertEqual(target.read_bytes(), payload)
                        target.write_bytes(converted.replace(b"ORD", b"LAX"))
                        with self.assertRaisesRegex(ValueError, "Reference changed: airports.csv"):
                            download.prepare_references(["ORD"], directory, "2024-01-01", "2025-01-31")
                        network.assert_not_called()

    def test_dst_clocks_and_invalid_times(self):
        convert = features.local_cutoff
        self.assertEqual(
            convert(date(2024, 7, 1), "1200", "America/Chicago")[0],
            datetime(2024, 7, 1, 17),
        )
        self.assertEqual(
            convert(date(2024, 1, 1), "1200", "America/Chicago")[0],
            datetime(2024, 1, 1, 18),
        )
        self.assertEqual(
            convert(date(2024, 12, 31), "2400", "America/Chicago")[0],
            datetime(2025, 1, 1, 6),
        )
        self.assertEqual(
            convert(date(2024, 3, 10), "0230", "America/Chicago")[1], "invalid_cutoff"
        )
        self.assertEqual(
            convert(date(2024, 11, 3), "0130", "America/Chicago")[1], "invalid_cutoff"
        )
        self.assertEqual(
            convert(date(2024, 7, 1), "1200", "America/Phoenix")[0],
            datetime(2024, 7, 1, 19),
        )
        for clock in ("2460", "1260", "2401", "-1", "12.0", None):
            self.assertEqual(
                convert(date(2024, 1, 1), clock, "America/Chicago")[1], "invalid_cutoff"
            )

    def test_download_resume_without_content_or_checksum_scans(self):
        payload = b"CSV content is checked during cleaning"
        request = [("USW00094846", 2024, "America/Chicago")]
        with tempfile.TemporaryDirectory() as tmp:
            def response(*args, **kwargs):
                stream = io.BytesIO(payload)
                stream.headers = {"Content-Length": str(len(payload))}
                return stream

            with (
                patch.object(io_utils, "urlopen", side_effect=response) as network,
                patch.object(io_utils, "sha256", side_effect=AssertionError("Unexpected checksum scan")),
                patch.object(download.csv, "DictReader", side_effect=AssertionError("Unexpected CSV scan")),
                patch.object(download.tqdm, "write") as messages,
            ):
                first = download.download_weather(request, tmp)
                self.assertEqual(first["files"]["2024/LCD_USW00094846_2024.csv"]["status"], "downloaded")
                target = Path(tmp) / "2024/LCD_USW00094846_2024.csv"
                self.assertTrue(any(f"Downloading weather file {target}" in call.args[0]
                                    for call in messages.call_args_list))
                self.assertTrue(any(f"Downloaded weather file {target} in " in call.args[0]
                                    for call in messages.call_args_list))
                target.write_bytes(b"changed locally")
                second = download.download_weather(request, tmp)
                messages.assert_called_with(f"Reusing weather file {target} because it already exists.")
                self.assertTrue(first["complete"] and second["complete"])
                self.assertEqual(target.read_bytes(), b"changed locally")
                self.assertEqual(network.call_count, 1)
                # Existing files also work without an older provenance manifest.
                (Path(tmp) / "manifest.json").unlink()
                third = download.download_weather(request, tmp)
                self.assertTrue(third["complete"])
                self.assertEqual(network.call_count, 1)

    def test_unavailable_and_failed_downloads_are_distinct(self):
        request = [("USW00094846", 2024, "America/Chicago")]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(download, "fetch", side_effect=HTTPError("url", 404, "missing", {}, None)):
                result = download.download_weather(request, tmp)
            self.assertEqual(next(iter(result["files"].values()))["status"], "unavailable")
            with patch.object(download, "fetch", side_effect=OSError("transfer failed")):
                with self.assertRaisesRegex(OSError, "transfer failed"):
                    download.download_weather(request, tmp)
            self.assertFalse(json.loads((Path(tmp) / "manifest.json").read_text())["complete"])
            self.assertFalse((Path(tmp) / "2024/LCD_USW00094846_2024.csv").exists())
            with self.assertRaisesRegex(ValueError, "inside"):
                project_path(Path(tmp), "../escape")


class WeatherSparkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark_session("test_origin_weather")

    def references(self):
        return (
            [
                {
                    "Origin": "ORD",
                    "time_zone": "America/Chicago",
                    "valid_from": "2024-01-01",
                    "valid_to": "2025-02-01",
                }
            ],
            [
                {
                    "Origin": "ORD",
                    "station_id": "USW00094846",
                    "valid_from": "2024-01-01",
                    "valid_to": "2025-02-01",
                    "method": "exact_icao_wban",
                }
            ],
        )

    def test_left_join_boundaries_and_equal_cohort(self):
        flights = self.spark.createDataFrame(
            [
                ("a", date(2024, 1, 1), "ORD", "1200", "train", 0),
                ("b", date(2024, 1, 1), "ORD", "1200", "train", 1),
                ("c", date(2024, 1, 1), "XXX", "1200", "train", 0),
            ],
            "flight_id string, FlightDate date, Origin string, CRSDepTime string, split string, ArrDel15 int",
        )
        cutoffs = features.build_flight_cutoffs(self.spark, flights, self.references())
        cutoffs = cutoffs.withColumn("parent_payload", F.repeat(F.col("flight_id"), 4096))
        weather = self.spark.createDataFrame(
            [
                (
                    "USW00094846",
                    datetime(2024, 1, 1, 16, tzinfo=timezone.utc),
                    "old",
                    1.0,
                    2.0,
                    3.0,
                ),
                (
                    "USW00094846",
                    datetime(2024, 1, 1, 17, 45, tzinfo=timezone.utc),
                    "eligible",
                    4.0,
                    5.0,
                    None,
                ),
                (
                    "USW00094846",
                    datetime(2024, 1, 1, 17, 46, tzinfo=timezone.utc),
                    "within_15_minutes",
                    9.0,
                    9.0,
                    9.0,
                ),
                ("USW00094846", datetime(2024, 1, 1, 18, tzinfo=timezone.utc), "at_departure", 6.0, 7.0, None),
                ("USW00094846", datetime(2024, 1, 1, 18, 1, tzinfo=timezone.utc), "future", 9.0, 9.0, 9.0),
            ],
            "station_id string, observation_utc timestamp, source_file string, origin_temperature_c double, origin_wind_speed_ms double, origin_visibility_km double",
        )
        decoded = decode_weather({"HourlyDryBulbTemperature": "4", "HourlyWindSpeed": "5"})
        for name in FEATURE_COLUMNS:
            if name not in weather.columns:
                weather = weather.withColumn(name, F.lit(decoded[name]).cast("double" if name in NUMERIC_FEATURES else "int"))
        joined = features.build_origin_features(
            cutoffs, weather, {"files": {}}
        ).cache()
        try:
            features.validate_cohort(flights, joined)
            rows = {row.flight_id: row for row in joined.collect()}
            self.assertEqual(rows["a"].source_file, "at_departure")
            self.assertEqual(joined.columns[:len(cutoffs.columns)], cutoffs.columns)
            self.assertEqual(joined.select(cutoffs.columns).schema, cutoffs.schema)
            self.assertEqual({key: row.parent_payload for key, row in rows.items()},
                             {key: key * 4096 for key in ("a", "b", "c")})
            self.assertEqual(rows["b"].origin_weather_age_minutes, 0)
            self.assertEqual(rows["a"].origin_available_utc, rows["a"].scheduled_departure_utc)
            self.assertEqual(rows["a"].origin_visibility_missing, 1)
            self.assertEqual(rows["c"].origin_weather_missing, 1)
            self.assertEqual(
                features.required_station_years(cutoffs, 120),
                [("USW00094846", 2024, "America/Chicago")],
            )
            with self.assertRaisesRegex(ValueError, "cohort"):
                features.validate_cohort(
                    flights, joined.withColumn("ArrDel15", F.lit(1))
                )
        finally:
            joined.unpersist()

    def test_cleaning_quarantines_time_disagreement_and_preserves_zero(self):
        rows = [
            [
                "USW00094846",
                "2024-07-01T00:51:00",
                "FM-15",
                "METAR KORD 010651Z",
                "20.0",
                "0",
                "16.093",
            ],
            [
                "USW00094846",
                "2024-07-01T00:51:00",
                "FM-16",
                "SPECI KORD 010651Z",
                "21.0",
                "3",
                "1.5V",
            ],
            [
                "USW00094846",
                "2024-07-01T01:51:00",
                "FM-15",
                "METAR KORD 010751Z",
                "20.0s",
                "1",
                "2.5V",
            ],
            [
                "USW00094846",
                "2024-07-01T02:51:00",
                "FM-15",
                "METAR KORD 010951Z",
                "20.0",
                "1",
                "3",
            ],
            ["USW00094846", "2024-07-01T08:51:00", "FM-15", "METAR KORD 010851Z", "19.0", "2", "3"],
            ["USW00094846", "2024-07-01T00:00:00", "SOD", "", "", "", ""],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2024").mkdir()
            path = root / "2024/LCD_USW00094846_2024.csv"
            path.write_bytes(raw_csv(rows))
            manifest = {
                "complete": True,
                "timestamp_basis": "local_standard",
                "units": {"temperature": "C", "wind_speed": "m/s", "visibility": "km"},
                "files": {
                    "file": {
                        "status": "verified",
                        "station_id": "USW00094846",
                        "year": 2024,
                        "time_zone": "America/Chicago",
                    }
                },
            }
            clean_data = clean.clean_weather(
                self.spark,
                root,
                root / "silver",
                root / "quarantine",
                manifest,
                root / "report",
            )
            result = clean_data.orderBy("observation_utc").collect()
            self.assertEqual(len(result), 3)
            utc_clocks = clean_data.select(
                F.date_format("observation_utc", "yyyy-MM-dd HH:mm").alias("utc")
            ).orderBy("utc").collect()
            self.assertEqual([row.utc for row in utc_clocks], [
                "2024-07-01 06:51", "2024-07-01 07:51", "2024-07-01 08:51",
            ])
            self.assertEqual(
                clean_data.select(
                    F.date_format("observation_utc", "yyyy-MM-dd HH:mm").alias("utc")
                )
                .orderBy("utc")
                .first()
                .utc,
                "2024-07-01 06:51",
            )
            self.assertEqual(result[0].origin_wind_speed_ms, 0)
            self.assertIsNone(result[1].origin_temperature_c)
            quality = json.loads((root / "report/_metadata.json").read_text())["quality"]
            self.assertEqual(quality["invalid_rows"], 1)
            self.assertEqual(quality["duplicate_observations"], 1)

    def test_station_year_requests_cover_utc_and_local_year_boundaries(self):
        for prefix in ("origin", "destination"):
            cutoffs = self.spark.createDataFrame(
                [("USW00003937", "America/Chicago", datetime(2025, 1, 1, 1))],
                f"{prefix}_station_id string, {prefix}_time_zone string, scheduled_departure_utc timestamp",
            )
            self.assertEqual(features.required_station_years(cutoffs, 120, prefix), [
                ("USW00003937", 2024, "America/Chicago"),
                ("USW00003937", 2025, "America/Chicago"),
            ])

    def test_expanded_fields_flow_from_cleaning_into_training_inputs(self):
        values = dict(zip(SOURCE_FIELDS, [
            "20", "4", "16.093", "T", "8", "090", "70", "-2", "1013.2",
            "-RA:02 FG:2 |RA FG |RA", "BKN:07 0.91",
        ]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bronze/2024/LCD_USW00094846_2024.csv"
            source.parent.mkdir(parents=True)
            source.write_bytes(raw_csv([[
                "USW00094846", "2024-01-01T17:50:00", "FM-15", "METAR KORD 011750Z",
                *[values[name] for name in SOURCE_FIELDS],
            ]]))
            manifest = {
                "complete": True, "timestamp_basis": "unvalidated",
                "units": {}, "files": {"fixture": {
                    "station_id": "USW00094846", "year": 2024,
                    "time_zone": "America/Chicago", "status": "downloaded",
                }},
            }
            observations = clean.clean_weather(
                self.spark, root / "bronze", root / "silver", root / "quarantine",
                manifest, root / "silver",
            )
            with patch.object(self.spark, "createDataFrame", side_effect=AssertionError("Unexpected reclean")):
                reused = clean.clean_weather(
                    self.spark, root / "bronze", root / "reused", root / "unused_quarantine",
                    manifest, root / "reused",
                )
                self.assertEqual(reused.collect(), observations.collect())
                self.assertFalse((root / "unused_quarantine").exists())
                self.assertEqual(
                    json.loads((root / "reused/_metadata.json").read_text()),
                    json.loads((root / "silver/_metadata.json").read_text()),
                )
            flights = self.spark.createDataFrame(
                [("a", date(2024, 1, 1), "ORD", "1200", "train", 0)],
                "flight_id string, FlightDate date, Origin string, CRSDepTime string, split string, ArrDel15 int",
            )
            cutoffs = features.build_flight_cutoffs(self.spark, flights, self.references())
            enriched = features.build_origin_features(cutoffs, observations, manifest).cache()
            try:
                features.validate_cohort(flights, enriched)
                values = enriched.select(*features.WEATHER_NUMERIC, *features.WEATHER_FLAGS).first()
                self.assertTrue(all(math.isfinite(values[name]) for name in features.WEATHER_NUMERIC))
                self.assertTrue(all(values[name] in (0, 1) for name in features.WEATHER_FLAGS))
                row = enriched.first()
                self.assertEqual(row.origin_weather_age_minutes, 10)
                self.assertEqual(row.origin_wind_direction_sin, 1)
                self.assertEqual(row.origin_cloud_ceiling_m, 91)
                self.assertEqual(row.origin_precipitation_trace, 1)
                self.assertEqual(row.origin_precipitation_missing, 0)
                self.assertEqual(row.origin_weather_rain, 1)
                self.assertEqual(row.origin_weather_fog, 1)
            finally:
                enriched.unpersist()

if __name__ == "__main__":
    unittest.main()
