"""Check baseline training on small fixtures without reading the real flight datasets."""

from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrameReader

from tests.conftest import get_test_spark_session
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bts import train as bts


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark_session("test_bts_train")

    def flights(self):
        rows = [("AA" if i % 2 else "DL", "ORD", "LAX", i // 4 + 1, i % 7 + 1, i % 24,
                 (i + 3) % 24, 120.0 + i, 800.0 + i * 5, int(i % 3 == 0), date(2024, i // 4 + 1, i % 4 + 1)) for i in range(36)]
        rows = [tuple(row[:9]) + (0.25, 8.0, 4.0, 0.5, 10.0, 6.0, 0, 0, 0) + tuple(row[9:]) for row in rows]
        return self.spark.createDataFrame(rows, bts.INPUT_FEATURES + [bts.TARGET_COL, "FlightDate"]).coalesce(2)

    def test_evaluation_matches_hand_counts(self):
        data = self.spark.createDataFrame([(1, 0.9, 1), (0, 0.8, 1), (1, 0.6, 1), (0, 0.2, 0), (1, 0.1, 0)],
                                          [bts.TARGET_COL, "p_delayed", "prediction"])
        result = bts.evaluate_predictions(data)
        self.assertEqual(result["confusion_matrix"], {"tp": 2, "fp": 1, "tn": 1, "fn": 1})
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertGreaterEqual(result["pr_auc"], 0)
        self.assertLessEqual(result["roc_auc"], 1)
        self.assertEqual(bts.confusion_metrics(0, 0, 3, 2)["precision"], 0.0)
        self.assertEqual(bts.confusion_metrics(0, 2, 3, 0)["recall"], 0.0)
        self.assertEqual(bts.confusion_metrics(0, 0, 3, 0)["f1"], 0.0)
        self.assertEqual(bts.confusion_metrics(0, 0, 0, 0)["accuracy"], 0.0)
        single_class = bts.evaluate_predictions(data.filter("ArrDel15 = 1"))
        self.assertIsNone(single_class["pr_auc"])
        self.assertIsNone(single_class["roc_auc"])

    def test_threshold_maximum_and_tie_rule(self):
        data = self.spark.createDataFrame([(1, 0.8), (1, 0.8), (0, 0.2), (0, 0.2)], [bts.TARGET_COL, "p_delayed"])
        with patch.object(data, "agg", wraps=data.agg) as aggregate:
            threshold, search = bts.tune_threshold(data)
        self.assertEqual(aggregate.call_count, 1)
        self.assertEqual(threshold, 0.8)
        best = search.loc[search["threshold"] == threshold].iloc[0]
        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(int(best["tp"]), 2)
        self.assertEqual(int(best["fp"]), 0)
        self.assertEqual(search["f1"].max(), best["f1"])

    def test_training_persistence_and_holdout_isolation(self):
        data = self.flights()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, models, results = root / "gold", root / "models", root / "results"
            data.write.parquet(str(gold / "train"))
            # The validation prevalence deliberately differs from training.
            validation = data.filter("ArrDel15 = 0").limit(12).union(data.filter("ArrDel15 = 1").limit(2))
            validation.write.parquet(str(gold / "val"))
            (gold / "test").mkdir()
            (gold / "test" / "unreadable.txt").write_text("This is intentionally not Parquet.")
            original = DataFrameReader.parquet
            reads = []

            def read(reader, *paths, **options):
                reads.extend(paths)
                return original(reader, *paths, **options)

            original_fit = bts.Pipeline.fit
            fit_dates = []
            original_classifier_fit = bts.LogisticRegression.fit
            original_gbt_fit = bts.GBTClassifier.fit
            classifier_fits = []
            gbt_fits = []

            def classifier_fit(estimator, dataset, *args, **kwargs):
                classifier_fits.append(1)
                return original_classifier_fit(estimator, dataset, *args, **kwargs)

            def gbt_fit(estimator, dataset, *args, **kwargs):
                gbt_fits.append(1)
                return original_gbt_fit(estimator, dataset, *args, **kwargs)

            def fit(pipeline, dataset, *args, **kwargs):
                self.assertEqual(dataset.storageLevel, StorageLevel.DISK_ONLY)
                dates = [row.FlightDate for row in dataset.select("FlightDate").collect()]
                fit_dates.append((min(dates), max(dates)))
                model = original_fit(pipeline, dataset, *args, **kwargs)
                self.assertNotIn("Month", model.stages[0].getInputCols())
                return model

            logistic_params = {**bts.LOGISTIC_PARAMS, "maxIter": 1}
            tree_params = {**bts.GBT_PARAMS, "maxDepth": 2, "maxIter": 2}
            with (patch.object(DataFrameReader, "parquet", read),
                  patch.object(bts.Pipeline, "fit", fit),
                  patch.object(bts.LogisticRegression, "fit", classifier_fit),
                  patch.object(bts.GBTClassifier, "fit", gbt_fit),
                  patch.object(bts, "LOGISTIC_PARAMS", logistic_params),
                  patch.object(bts, "GBT_PARAMS", tree_params)):
                summary = bts.train(self.spark, gold, models, results, include_gbt=True)
            self.assertEqual(len(classifier_fits), 1)
            self.assertEqual(len(gbt_fits), 1)
            self.assertEqual(set(reads), {str(gold / "train"), str(gold / "val")})
            self.assertEqual(summary["tuning"]["method"], "fixed hyperparameters")
            self.assertFalse(summary["tuning"]["search_performed"])
            self.assertNotIn("candidate_comparison", summary)
            self.assertEqual(fit_dates, [(date(2024, 1, 1), date(2024, 9, 4))] * 2)
            self.assertEqual(summary["training_rows"], 36)
            self.assertEqual(summary["validation_rows"], 14)
            self.assertEqual(set(summary["reference_baselines"]), {"always_ontime", "always_delayed", "train_prevalence"})
            self.assertAlmostEqual(summary["reference_baselines"]["train_prevalence"]["training_prevalence"], 1 / 3)
            self.assertEqual(summary["selected_model"]["reg_param"], logistic_params["regParam"])
            self.assertEqual(summary["selected_model"]["elastic_net_param"], logistic_params["elasticNetParam"])
            self.assertEqual(set(path.name for path in results.iterdir()), {"results.json"})
            self.assertEqual(set(path.name for path in models.iterdir()), {"baseline_pipeline", "threshold.json", "gbt"})
            self.assertEqual(json.loads((results / "results.json").read_text())["bts"]["training"], summary)
            saved = PipelineModel.load(str(models / "baseline_pipeline"))
            classifier = saved.stages[-1]
            self.assertEqual(classifier.getRegParam(), logistic_params["regParam"])
            self.assertEqual(classifier.getElasticNetParam(), logistic_params["elasticNetParam"])
            self.assertEqual(classifier.getMaxIter(), logistic_params["maxIter"])
            threshold = json.loads((models / "threshold.json").read_text())["threshold"]
            scores = bts.predict(saved, validation, threshold)
            metrics = bts.evaluate_predictions(scores)
            expected = summary["selected_model"]["validation_metrics_at_selected_threshold"]
            self.assertEqual(metrics["confusion_matrix"], expected["confusion_matrix"])
            self.assertAlmostEqual(metrics["f1"], expected["f1"])
            challenger = summary["algorithm_comparison"]["gradient_boosted_trees"]
            self.assertNotIn("internal_pr_auc", challenger)
            self.assertEqual(challenger["max_depth"], tree_params["maxDepth"])
            self.assertEqual(challenger["max_iter"], tree_params["maxIter"])
            self.assertEqual(set(challenger["timing_seconds"]), {
                "preprocessor_fit", "classifier_fit", "validation", "save_model_and_threshold",
            })
            self.assertTrue(all(seconds >= 0 for seconds in challenger["timing_seconds"].values()))
            tree = PipelineModel.load(str(models / "gbt/baseline_pipeline"))
            self.assertEqual(tree.stages[-1].getMaxDepth(), challenger["max_depth"])
            tree_threshold = json.loads((models / "gbt/threshold.json").read_text())["threshold"]
            tree_metrics = bts.evaluate_predictions(bts.predict(tree, validation, tree_threshold))
            self.assertEqual(tree_metrics["confusion_matrix"], challenger["validation_metrics_at_selected_threshold"]["confusion_matrix"])
            published = json.loads((results / "results.json").read_text())["bts"]["validation"]
            self.assertEqual(published["selected"], expected)

            default_models = root / "default"
            with (patch.object(bts, "LOGISTIC_PARAMS", logistic_params),
                  patch.object(bts, "GBTClassifier", side_effect=AssertionError("Unexpected GBT"))):
                default = bts.train(self.spark, gold, default_models, results)
            self.assertEqual(set(default["algorithm_comparison"]), {"logistic_regression"})
            self.assertEqual(set(default_models.iterdir()), {
                default_models / "baseline_pipeline", default_models / "threshold.json",
            })
            self.assertEqual(set(default["timing_seconds"]), {"prepare_cohorts", "reference_baselines"})
            self.assertEqual(default["selected_model"], summary["selected_model"])
            self.assertEqual(json.loads((results / "results.json").read_text())["bts"]["training"], default)


if __name__ == "__main__":
    unittest.main()
