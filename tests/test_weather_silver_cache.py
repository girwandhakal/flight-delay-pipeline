"""Weather Silver reuse must skip CSV work without accepting stale outputs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from scripts.origin_weather import clean


class WeatherSilverCacheTests(unittest.TestCase):
    def test_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "bronze/2024/LCD_USW00094846_2024.csv"
            raw.parent.mkdir(parents=True)
            raw.write_text("fixture")
            silver = root / "silver/first"
            silver.mkdir(parents=True)
            output = silver / "part-000.parquet"
            output.write_bytes(b"fixture")
            success = silver / "_SUCCESS"
            success.touch()
            manifest = {"complete": True, "files": {"fixture": {
                "status": "downloaded", "station_id": "USW00094846",
                "year": 2024, "time_zone": "America/Chicago",
            }}}
            inputs = {
                "files": clean.file_fingerprints([raw]),
                "stations": [["USW00094846", 2024, "America/Chicago"]],
                "code": {name: clean.sha256(Path(clean.__file__).with_name(name))
                         for name in ("clean.py", "measurements.py", "download.py")},
                "spark": "3.5.7",
            }
            metadata = {"quality": {"retained_observations": 1}, "station_quality": [],
                        "source_contract": {}, "cleaning_cache": {
                            "inputs": inputs, "outputs": clean.file_fingerprints([output]),
                        }}
            metadata_path = silver / "_metadata.json"
            original = json.dumps(metadata)
            metadata_path.write_text(original)

            def run():
                spark = MagicMock()
                spark.version = "3.5.7"
                spark.createDataFrame.side_effect = RuntimeError("cleaning started")
                result = clean.clean_weather(spark, root / "bronze", root / "silver/next",
                                            root / "quarantine", manifest, root / "reports")
                spark.createDataFrame.assert_not_called()
                spark.read.csv.assert_not_called()
                spark.read.parquet.assert_called_once_with(str(silver), pathGlobFilter="*.parquet")
                self.assertIs(result, spark.read.parquet.return_value)

            run()
            self.assertEqual(json.loads((root / "reports/_metadata.json").read_text()), metadata)
            # Manifest retrieval timestamps do not change the cleaning inputs.
            manifest["files"]["fixture"]["retrieved_utc"] = "later"
            run()
            for change in ("raw", "timezone", "code", "output", "success", "metadata"):
                with self.subTest(change=change):
                    metadata_path.write_text(original)
                    success.touch()
                    manifest["files"]["fixture"]["time_zone"] = "America/Chicago"
                    if change == "raw":
                        raw.write_text("changed fixture")
                    elif change == "timezone":
                        manifest["files"]["fixture"]["time_zone"] = "America/New_York"
                    elif change == "code":
                        changed = json.loads(original)
                        changed["cleaning_cache"]["inputs"]["code"]["clean.py"] = "old"
                        metadata_path.write_text(json.dumps(changed))
                    elif change == "output":
                        output.unlink()
                    elif change == "success":
                        success.unlink()
                    else:
                        metadata_path.write_text("{}")
                    with self.assertRaisesRegex(RuntimeError, "cleaning started"):
                        run()
                    # Refresh fingerprints so each case independently tests its own invalidation.
                    raw.write_text("fixture")
                    output.write_bytes(b"fixture")
                    metadata["cleaning_cache"]["inputs"]["files"] = clean.file_fingerprints([raw])
                    metadata["cleaning_cache"]["outputs"] = clean.file_fingerprints([output])
                    original = json.dumps(metadata)
