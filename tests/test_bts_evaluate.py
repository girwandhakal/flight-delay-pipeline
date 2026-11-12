"""Unit tests for BTS baseline test evaluation using synthetic Spark fixtures."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bts import evaluate as bts_eval
from scripts.bts import train as bts_tr


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("test_bts_evaluate")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def synthetic_flights(self, count=40):
        rows = [
            (
                "AA" if i % 2 == 0 else "DL",
                "ORD" if i % 3 == 0 else ("DFW" if i % 3 == 1 else "ATL"),
                "LAX" if i % 2 == 0 else "JFK",
                12 if i % 2 == 0 else 1,
                (i % 7) + 1,
                i % 24,
                (i + 3) % 24,
                120.0 + i,
                800.0 + (i * 10),
                int(i % 4 == 0),
            )
            for i in range(count)
        ]
        columns = bts_tr.CATEGORICAL_FEATURES + bts_tr.NUMERIC_FEATURES + [bts_eval.TARGET_COL]
        return self.spark.createDataFrame(rows, columns).coalesce(2)

    def test_subgroup_aggregation_correctness(self):
        # Fixture with distinct airlines, origins, departure hours, and known predictions
        df = self.spark.createDataFrame(
            [
                ("AA", "ORD", 8, 1, 1),
                ("AA", "DFW", 8, 0, 0),
                ("DL", "ORD", 14, 1, 0),
                ("DL", "ATL", 14, 0, 1),
                ("UA", "ATL", 20, 0, 0),
            ],
            ["Reporting_Airline", "Origin", "DepHour", bts_eval.TARGET_COL, "prediction"],
        )
        subgroup_df = bts_eval.compute_subgroups(df)

        self.assertIsInstance(subgroup_df, pd.DataFrame)
        self.assertIn("subgroup_type", subgroup_df.columns)
        self.assertIn("subgroup_value", subgroup_df.columns)

        # 1. Sum of total_flights across all categories for airline must equal total input rows (5)
        airline_slice = subgroup_df[subgroup_df["subgroup_type"] == "airline"]
        self.assertEqual(airline_slice["total_flights"].sum(), 5)
        self.assertEqual(set(airline_slice["subgroup_value"]), {"AA", "DL", "UA"})

        # 2. For every slice row, tp + fp + tn + fn must equal total_flights
        for _, row in subgroup_df.iterrows():
            total_sum = row["tp"] + row["fp"] + row["tn"] + row["fn"]
            self.assertEqual(total_sum, row["total_flights"])
            self.assertEqual(row["tp"] + row["fn"], row["delayed_flights"])

    def test_zero_division_guardrails(self):
        # When tp + fp = 0 (zero positive predictions)
        metrics = bts_eval.confusion_metrics(tp=0, fp=0, tn=10, fn=5)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 10 / 15)

        # When tp + fn = 0 (no positive labels in dataset)
        metrics_no_pos = bts_eval.confusion_metrics(tp=0, fp=5, tn=10, fn=0)
        self.assertEqual(metrics_no_pos["recall"], 0.0)
        self.assertEqual(metrics_no_pos["f1"], 0.0)

        # When total = 0
        metrics_empty = bts_eval.confusion_metrics(tp=0, fp=0, tn=0, fn=0)
        self.assertEqual(metrics_empty["precision"], 0.0)
        self.assertEqual(metrics_empty["recall"], 0.0)
        self.assertEqual(metrics_empty["f1"], 0.0)
        self.assertEqual(metrics_empty["accuracy"], 0.0)

    def test_artifact_persistence(self):
        data = self.synthetic_flights(count=30)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            gold_dir = root / "gold"
            models_dir = root / "models"
            results_dir = root / "results"

            # Create test parquet partition
            test_dir = gold_dir / "test"
            data.write.parquet(str(test_dir))

            # Create mock pipeline model and threshold.json
            preprocessor = bts_tr.build_preprocessor().fit(data)
            lr = LogisticRegression(labelCol=bts_eval.TARGET_COL, regParam=0.1).fit(preprocessor.transform(data))
            model = PipelineModel(stages=preprocessor.stages + [lr])
            pipeline_dir = models_dir / "baseline_pipeline"
            model.write().save(str(pipeline_dir))

            (models_dir / "threshold.json").write_text(json.dumps({"threshold": 0.25}))
            summary = bts_eval.evaluate(
                self.spark,
                gold_dir=gold_dir,
                models_dir=models_dir,
                results_dir=results_dir,
            )

            # Check that files were persisted
            self.assertEqual([p.name for p in results_dir.iterdir()], ["results.json"])

            # Validate summary structure
            loaded_summary = json.loads((results_dir / "results.json").read_text())["bts"]["test"]
            self.assertEqual(loaded_summary["evaluation_split"], "test")
            self.assertEqual(loaded_summary["total_records"], 30)
            self.assertEqual(loaded_summary["selected_threshold"], 0.25)
            self.assertIn("metrics_at_selected_threshold", loaded_summary)
            self.assertIn("metrics_at_default_0_5", loaded_summary)
            self.assertIn("reference_baselines", loaded_summary)
            self.assertIn("always_ontime", loaded_summary["reference_baselines"])
            self.assertIn("always_delayed", loaded_summary["reference_baselines"])
            # Validate CSV structure
            loaded_csv = pd.DataFrame(loaded_summary["subgroups"])
            expected_cols = [
                "subgroup_type", "subgroup_value", "total_flights", "delayed_flights",
                "prevalence", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"
            ]
            self.assertEqual(list(loaded_csv.columns), expected_cols)
            self.assertGreater(len(loaded_csv), 0)


if __name__ == "__main__":
    unittest.main()
