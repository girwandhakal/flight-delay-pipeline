"""Check single-file result updates and failure recovery without Spark."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.bts.evaluate import evaluate, save_evaluation_results
from scripts.results import update_results
from pipeline import main as pipeline


class ResultsTests(unittest.TestCase):
    def test_updates_preserve_other_models_and_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            update_results({"bts": {"training": {"fit": 1}, "validation": {"f1": 0.2}},
                            "bts_origin_weather": {"validation": {"f1": 0.3}}}, tmp)
            for run in range(3):
                save_evaluation_results({"run": run}, pd.DataFrame({"run": [run]}), tmp)
            path = Path(tmp) / "results.json"
            result = json.loads(path.read_text())
            self.assertEqual(result["bts"]["test"], {"run": 2, "subgroups": [{"run": 2}]})
            self.assertEqual(result["bts"]["training"], {"fit": 1})
            self.assertEqual(result["bts"]["validation"], {"f1": 0.2})
            self.assertEqual(result["bts_origin_weather"], {"validation": {"f1": 0.3}})
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_retraining_clears_only_that_models_stale_test_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            update_results({"bts": {"test": {"f1": 0.1}, "run_id": "old"},
                            "bts_origin_weather": {"test": {"f1": 0.2}}}, tmp)
            path = update_results({"bts": {"training": {"fit": 2}, "validation": {}}}, tmp)
            result = json.loads(path.read_text())
            self.assertNotIn("test", result["bts"])
            self.assertNotIn("run_id", result["bts"])
            self.assertIn("test", result["bts_origin_weather"])

    def test_failed_serialization_or_publication_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = update_results({"bts": {"test": {"run": 1}}}, tmp)
            previous = path.read_bytes()
            with self.assertRaises(ValueError):
                update_results({"bts": {"test": float("nan")}}, tmp)
            with patch.object(Path, "replace", side_effect=OSError("publication failed")):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    update_results({"bts": {"test": {"run": 2}}}, tmp)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_evaluation_uses_current_model_and_rejects_an_unrelated_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "current_model"
            update_results({"bts": {"training": {"models_dir": str(model)}}}, root)
            with self.assertRaisesRegex(FileNotFoundError, "current_model"):
                evaluate(MagicMock(), results_dir=root)
            with self.assertRaisesRegex(ValueError, "current BTS training"):
                evaluate(MagicMock(), models_dir=root / "old_model", results_dir=root)

    def test_corrupt_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text("broken")
            with self.assertRaises(json.JSONDecodeError):
                update_results({"bts": {"validation": {}}}, tmp)
            self.assertEqual(path.read_text(), "broken")


class ComparisonPublicationTests(unittest.TestCase):
    def test_both_models_publish_together_and_failed_comparison_preserves_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                path = root / "data/gold/bts" / split
                path.mkdir(parents=True)
                (path / "_SUCCESS").touch()
            gold = root / "enriched"
            gold.mkdir()
            (gold / "_metadata.json").write_text("{}")
            latest = update_results({"bts": {"test": {"old": True}},
                                     "bts_origin_weather": {"test": {"old": True}}}, root / "results")
            publish = lambda updates: update_results(updates, root / "results")
            with (patch.object(pipeline, "ROOT", root),
                  patch.object(pipeline.weather_features, "main", return_value=(gold, gold)),
                  patch.object(pipeline.bts_transform, "get_spark_session", return_value=MagicMock(version="fixture")),
                  patch.object(pipeline, "load_config", return_value=({}, {}, {})),
                  patch.object(pipeline, "update_results", side_effect=publish),
                  patch.object(pipeline.bts_train, "train", return_value={"fit": "new"}) as train,
                  patch.object(pipeline, "compare_predictions", side_effect=ValueError("cohort mismatch")) as compare):
                previous = latest.read_bytes()
                with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                    pipeline.run_origin_weather()
                self.assertEqual(latest.read_bytes(), previous)
                self.assertTrue(all(call.kwargs["publish_results"] is False for call in train.call_args_list))
                compare.side_effect = None
                compare.return_value = {name: {"validation": {"f1": 0.2}}
                                        for name in ("bts", "bts_origin_weather")}
                self.assertEqual(pipeline.run_origin_weather(), 0)
                result = json.loads(latest.read_text())
                for name in ("bts", "bts_origin_weather"):
                    self.assertEqual(result[name]["training"], {"fit": "new"})
                    self.assertEqual(result[name]["validation"], {"f1": 0.2})
                    self.assertNotIn("test", result[name])
                self.assertEqual(result["bts"]["run_id"], result["bts_origin_weather"]["run_id"])
                self.assertEqual([p.name for p in (root / "results").iterdir()], ["results.json"])

    def test_old_comparison_cannot_attach_test_scores_to_new_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "models/origin_weather_comparison/old"
            run.mkdir(parents=True)
            (run / "_SUCCESS").touch()
            update_results({"bts": {"run_id": "new"}}, root / "results")
            with (patch.object(pipeline, "ROOT", root),
                  patch.object(pipeline.bts_transform, "get_spark_session") as spark):
                with self.assertRaisesRegex(ValueError, "current comparison"):
                    pipeline.evaluate_origin_weather("old")
                spark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
