"""Small downloader fixtures; no network or production-data writes."""

from pathlib import Path
import sys
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bts import download as bts


class DownloadTests(unittest.TestCase):
    def test_fresh_clone_creates_configured_folders_and_skips_existing_csvs(self):
        for output_directory in (None, "data/custom/raw/bts"):
            with self.subTest(output_directory=output_directory), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings = {"start_month": "2024-12", "end_month": "2025-01"}
                if output_directory is not None:
                    settings["output_directory"] = output_directory
                (root / "config.json").write_text(json.dumps({"datasets": {"bts": settings}}), encoding="utf-8")
                self.assertFalse((root / "data").exists())
                settings = bts.load_config(root)
                content = io.BytesIO()
                with zipfile.ZipFile(content, "w") as archive:
                    archive.writestr("../../escape.csv", "a,b\n1,2\n")
                payload = content.getvalue()

                def response(*args, **kwargs):
                    stream = io.BytesIO(payload)
                    stream.headers = {"Content-Length": str(len(payload))}
                    return stream

                with patch.object(bts, "ROOT", root), patch.object(bts, "load_config", return_value=settings), \
                     patch.object(bts, "urlopen", side_effect=response) as request:
                    self.assertEqual(bts.main(), 0)
                    self.assertEqual(request.call_count, 2)
                    for layer in ("bronze", "silver", "gold", "quarantine"):
                        self.assertTrue((root / "data" / layer / "bts").is_dir())
                    self.assertEqual(len(list(root.rglob("*.csv"))), 2)
                    for year, month in ((2024, 12), (2025, 1)):
                        folder = root / (output_directory or "data/bronze/bts") / str(year) / f"{month:02d}"
                        self.assertEqual(len(list(folder.glob("*.csv"))), 1)
                        self.assertEqual(next(folder.glob("*.csv")).read_text(), "a,b\n1,2\n")
                    self.assertEqual(bts.main(), 0)
                    self.assertEqual(request.call_count, 2)

    def test_month_range_across_year(self):
        self.assertEqual(list(bts.iter_months("2023-11", "2024-02")),
                         [(2023, 11), (2023, 12), (2024, 1), (2024, 2)])
        self.assertEqual(list(bts.iter_months("2024-01", "2024-01")), [(2024, 1)])
        for start, end in [("2024-03", "2024-01"), ("2024-13", "2025-01"), ("2024-1", "2024-03")]:
            with self.assertRaises(ValueError):
                list(bts.iter_months(start, end))

    def test_failure_does_not_stop_later_months(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = {"start_month": "2024-01", "end_month": "2024-03", "output_directory": "data/bronze/bts"}
            with patch.object(bts, "ROOT", Path(directory)), patch.object(bts, "load_config", return_value=settings), \
                 patch.object(bts, "download_file", side_effect=[OSError("unavailable"), None, None]) as download, \
                 patch.object(bts, "extract_csv") as extract:
                self.assertEqual(bts.main(), 1)
                self.assertEqual(download.call_count, 3)
                self.assertEqual(extract.call_count, 2)

    # Failed extraction must leave the ZIP available for another attempt.
    def test_failed_extraction_preserves_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "flights.zip"
            destination = archive.with_suffix(".csv")
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("flights.csv", "a,b\n1,2\n")
            with patch.object(bts.shutil, "copyfileobj", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    bts.extract_csv(archive, destination)
            self.assertTrue(archive.exists())
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".csv.part").exists())

    # A progress-wrapped download must save the exact ZIP bytes.
    def test_download_with_progress(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            archive.writestr("flights.csv", "a,b\n1,2\n")
        payload = content.getvalue()
        for known_size in (True, False):
            with self.subTest(known_size=known_size), tempfile.TemporaryDirectory() as directory:
                response = io.BytesIO(payload)
                response.headers = {"Content-Length": str(len(payload))} if known_size else {}
                destination = Path(directory) / "flights.zip"
                with patch.object(bts, "urlopen", return_value=response):
                    self.assertEqual(bts.download_file("https://example.invalid/flights.zip", destination), destination)
                self.assertEqual(destination.read_bytes(), payload)

if __name__ == "__main__":
    unittest.main()
