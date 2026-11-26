"""Check result updates, history snapshots and failure recovery without Spark."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


from scripts.common.results import update_results


class ResultsTests(unittest.TestCase):
    def test_updates_preserve_other_models_and_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp) / "results"
            update_results({"bts": {"training": {"fit": 1}, "validation": {"f1": 0.2}},
                            "bts_origin_weather": {"validation": {"f1": 0.3}}}, tmp)
            for run in range(3):
                update_results({"bts": {"test": {"run": run, "subgroups": [{"run": run}]}}}, tmp)
            path = Path(tmp) / "results.json"
            result = json.loads(path.read_text())
            self.assertEqual(result["bts"]["test"], {"run": 2, "subgroups": [{"run": 2}]})
            self.assertEqual(result["bts"]["training"], {"fit": 1})
            self.assertEqual(result["bts"]["validation"], {"f1": 0.2})
            self.assertEqual(result["bts_origin_weather"], {"validation": {"f1": 0.3}})
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_retraining_clears_only_that_models_stale_test_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp) / "results"
            update_results({"bts": {"test": {"f1": 0.1}, "run_id": "old"},
                            "bts_origin_weather": {"test": {"f1": 0.2}}}, tmp)
            path = update_results({"bts": {"training": {"fit": 2}, "validation": {}}}, tmp)
            result = json.loads(path.read_text())
            self.assertNotIn("test", result["bts"])
            self.assertNotIn("run_id", result["bts"])
            self.assertIn("test", result["bts_origin_weather"])

    def test_failed_serialization_or_publication_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp) / "results"
            path = update_results({"bts": {"test": {"run": 1}}}, tmp)
            previous = path.read_bytes()
            with self.assertRaises(ValueError):
                update_results({"bts": {"test": float("nan")}}, tmp)
            with patch.object(Path, "replace", side_effect=OSError("publication failed")):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    update_results({"bts": {"test": {"run": 2}}}, tmp)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])


    def test_corrupt_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text("broken")
            with self.assertRaises(json.JSONDecodeError):
                update_results({"bts": {"validation": {}}}, tmp)
            self.assertEqual(path.read_text(), "broken")




if __name__ == "__main__":
    unittest.main()
