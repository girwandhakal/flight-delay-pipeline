"""Small downloader fixtures; no network or production-data writes."""

from pathlib import Path
import sys
import io
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_bts as bts


class DownloadTests(unittest.TestCase):
    def test_extract_and_rerun_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "data/bronze/bts/2024/01"
            folder.mkdir(parents=True)
            archive = folder / "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../../escape.csv", "a,b\n1,2\n")
            with patch.object(bts, "urlopen", side_effect=AssertionError("Unexpected network request")):
                result = archive.with_suffix(".csv")
                bts.extract_csv(archive, result)
                self.assertEqual(result.read_text(), "a,b\n1,2\n")
                self.assertFalse(archive.exists())
                self.assertEqual(bts.extract_csv(archive, result), result)
            self.assertEqual(len(list(root.rglob("*.csv"))), 1)
            self.assertFalse((root / "escape.csv").exists())

    def test_month_range_across_year(self):
        self.assertEqual(list(bts.iter_months("2023-11", "2024-02")),
                         [(2023, 11), (2023, 12), (2024, 1), (2024, 2)])
        self.assertEqual(list(bts.iter_months("2024-01", "2024-01")), [(2024, 1)])
        for start, end in [("2024-03", "2024-01"), ("2024-13", "2025-01"), ("2024-1", "2024-03")]:
            with self.assertRaises(ValueError):
                list(bts.iter_months(start, end))

    def test_url(self):
        self.assertEqual(bts.build_download_url(2024, 1),
                         "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip")

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

    def test_invalid_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.zip"
            path.write_bytes(b"not a ZIP")
            with self.assertRaises(zipfile.BadZipFile):
                bts.validate_archive(path)


if __name__ == "__main__":
    unittest.main()
