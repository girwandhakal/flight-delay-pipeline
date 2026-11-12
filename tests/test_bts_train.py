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
from pyspark.sql import DataFrameReader, SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bts import train as bts


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder.master("local[2]").appName("test_bts_train")
                     .config("spark.sql.shuffle.partitions", "2").getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def flights(self):
        rows = [("AA" if i % 2 else "DL", "ORD", "LAX", i // 4 + 1, i % 7 + 1, i % 24,
                 (i + 3) % 24, 120.0 + i, 800.0 + i * 5, int(i % 3 == 0), date(2024, i // 4 + 1, i % 4 + 1)) for i in range(36)]
        return self.spark.createDataFrame(rows, bts.CATEGORICAL_FEATURES + bts.NUMERIC_FEATURES + [bts.TARGET_COL, "FlightDate"]).coalesce(2)

    def test_preprocessor_keeps_unseen_categories(self):
        data = self.flights()
        model = bts.build_preprocessor().fit(data)
        unseen = self.spark.createDataFrame([("NEW", "XXX", "YYY", 12, 7, 23, 1, 150.0, 900.0, 1, date(2024, 12, 1))], data.schema)
        row = model.transform(unseen).first()
        self.assertEqual(row["Reporting_Airline_idx"], len(model.stages[0].labelsArray[0]))
        self.assertEqual(row["Month_idx"], len(model.stages[0].labelsArray[3]))
        self.assertEqual(model.transform(unseen).count(), 1)
        self.assertEqual(row[bts.TARGET_COL], 1)
        self.assertEqual(row["features"].size, model.transform(data).first()["features"].size)

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

    def test_chronological_search_rejects_too_few_months(self):
        data = self.flights().filter("Month <= 3")
        with patch.object(DataFrameReader, "parquet", return_value=data):
            with self.assertRaisesRegex(ValueError, "at least four training months"):
                bts.train(self.spark)

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
            fold_fits = []

            def classifier_fit(estimator, dataset, *args, **kwargs):
                if dataset.columns == ["features", bts.TARGET_COL]:
                    self.assertEqual(dataset.storageLevel, StorageLevel.DISK_ONLY)
                    fold_fits.append(1)
                return original_classifier_fit(estimator, dataset, *args, **kwargs)

            def fit(pipeline, dataset, *args, **kwargs):
                dates = [row.FlightDate for row in dataset.select("FlightDate").collect()]
                fit_dates.append((min(dates), max(dates)))
                model = original_fit(pipeline, dataset, *args, **kwargs)
                if len(pipeline.getStages()) == 3:
                    # The scoring month must never enter the fitted category vocabulary.
                    self.assertEqual(len(model.stages[0].labelsArray[3]), max(dates).month)
                return model

            with patch.object(DataFrameReader, "parquet", read), patch.object(bts.Pipeline, "fit", fit), patch.object(bts.LogisticRegression, "fit", classifier_fit):
                summary = bts.train(self.spark, gold, models, results)
            self.assertEqual(len(fold_fits), 3 * len(bts.REG_PARAMS) * len(bts.ELASTIC_NET_PARAMS))
            self.assertEqual(set(reads), {str(gold / "train"), str(gold / "val")})
            self.assertEqual(summary["tuning"]["method"], "expanding-window chronological cross-validation")
            self.assertEqual(fit_dates, [(date(2024, 1, 1), date(2024, month, 4)) for month in (6, 7, 8, 9)])
            self.assertEqual(summary["tuning"]["folds"], [
                {"train_before": f"2024-{month:02d}-01", "validation_month": f"2024-{month:02d}"}
                for month in (7, 8, 9)
            ])
            for candidate in summary["candidate_comparison"]:
                self.assertEqual(len(candidate["fold_pr_auc"]), 3)
                self.assertAlmostEqual(candidate["internal_pr_auc"], sum(candidate["fold_pr_auc"]) / 3)
            self.assertEqual(summary["training_rows"], 36)
            self.assertEqual(summary["validation_rows"], 14)
            self.assertEqual({(r["reg_param"], r["elastic_net_param"]) for r in summary["candidate_comparison"]},
                             {(r, e) for r in bts.REG_PARAMS for e in bts.ELASTIC_NET_PARAMS})
            self.assertEqual(set(summary["reference_baselines"]), {"always_ontime", "always_delayed", "train_prevalence"})
            self.assertAlmostEqual(summary["reference_baselines"]["train_prevalence"]["training_prevalence"], 1 / 3)
            chosen = max(summary["candidate_comparison"], key=lambda row: row["internal_pr_auc"])
            self.assertEqual(summary["selected_model"]["reg_param"], chosen["reg_param"])
            self.assertEqual(summary["selected_model"]["elastic_net_param"], chosen["elastic_net_param"])
            self.assertEqual(set(path.name for path in results.iterdir()), {"results.json"})
            self.assertEqual(set(path.name for path in models.iterdir()), {"baseline_pipeline", "threshold.json"})
            self.assertEqual(json.loads((results / "results.json").read_text())["bts"]["training"], summary)
            saved = PipelineModel.load(str(models / "baseline_pipeline"))
            classifier = saved.stages[-1]
            self.assertEqual(classifier.getRegParam(), chosen["reg_param"])
            self.assertEqual(classifier.getElasticNetParam(), chosen["elastic_net_param"])
            threshold = json.loads((models / "threshold.json").read_text())["threshold"]
            scores = bts.predict(saved, validation, threshold)
            metrics = bts.evaluate_predictions(scores)
            expected = summary["selected_model"]["validation_metrics_at_selected_threshold"]
            self.assertEqual(metrics["confusion_matrix"], expected["confusion_matrix"])
            self.assertAlmostEqual(metrics["f1"], expected["f1"])


if __name__ == "__main__":
    unittest.main()
