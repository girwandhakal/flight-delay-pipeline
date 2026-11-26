"""Check frozen scoring publication and subgroup metrics."""

from datetime import date
import json
import unittest
from unittest.mock import Mock

import pandas as pd
import pytest

import shutil

from scripts.common.io import sha256, write_json
from scripts.common.results import update_results
from scripts.common.stage_cache import resolve_stage
from scripts.modeling import evaluate as scoring
from scripts.modeling import run
from tests.conftest import get_test_spark_session


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark_session("test_model_evaluation")

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
            ["Reporting_Airline", "Origin", "DepHour", scoring.TARGET_COL, "prediction"],
        )
        subgroup_df = scoring.compute_subgroups(df)

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


@pytest.fixture
def frozen_run(tmp_path):
    def build_gold(directory):
        test = directory / "test"
        test.mkdir()
        (test / "_SUCCESS").touch()
        (test / "part.parquet").write_bytes(b"fixture")
        return {}

    gold = resolve_stage(tmp_path / run.GOLD_DIR, lambda: {}, ("test",), (), build_gold)
    run_dir = tmp_path / run.RUNS_DIR / "current"
    variants = {}
    for name, parent in (("unweighted", run_dir), ("weighted", run_dir / "weighted_logistic")):
        model = parent / "model"
        model.mkdir(parents=True)
        (model / "metadata").write_bytes(b"frozen model")
        threshold = parent / "threshold.json"
        write_json(threshold, {"threshold": 0.25})
        model_path, threshold_path = (file.relative_to(run_dir).as_posix() for file in (model, threshold))
        variants[name] = {
            "model": model_path, "threshold_file": threshold_path, "threshold": 0.25,
            "sha256": run.inference_hashes(run_dir, model_path, threshold_path),
        }
    (run_dir / "_SUCCESS").touch()
    write_json(run_dir / "run.json", {
        "run_id": run_dir.name, "gold": gold.relative_to(tmp_path).as_posix(),
        "gold_identity": run.gold_identity(gold, ("test",)), "variants": variants,
    })
    update_results({
        "historical": {"test": {"f1": 0.1}},
        run.RESULT_KEY: {"run_id": run_dir.name, "run_sha256": sha256(run_dir / "run.json"), "test": {"attempt_id": "old"}},
    }, tmp_path / "results")
    return run_dir, gold


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["run_id", "stale", "threshold", "gold_escape", "missing_model", "missing_test_gold"])
def test_frozen_scoring_rejects_invalid_inputs(tmp_path, frozen_run, kind):
    run_dir, gold = frozen_run
    current = tmp_path / "results/results.json"
    run_id = run_dir.name
    if kind == "run_id":
        run_id = "../escape"
    elif kind == "stale":
        update_results({run.RESULT_KEY: {"run_id": "newer"}}, tmp_path / "results")
    elif kind == "gold_escape":
        manifest = run_dir / "run.json"
        payload = json.loads(manifest.read_text())
        payload["gold"] = "data/gold/other"
        write_json(manifest, payload)
    elif kind == "threshold":
        write_json(run_dir / "threshold.json", {"threshold": 0.99})
    elif kind == "missing_model":
        shutil.rmtree(run_dir / "weighted_logistic/model")
    else:
        (gold / "test/_SUCCESS").unlink()
    previous = current.read_bytes()
    spark = Mock()
    with pytest.raises(ValueError):
        scoring.evaluate(spark, tmp_path, run_id)
    spark.read.parquet.assert_not_called()
    assert current.read_bytes() == previous


@pytest.mark.unit
def test_scoring_publishes_all_variants_together_and_retries(tmp_path, frozen_run, monkeypatch):
    run_dir, _ = frozen_run
    current = tmp_path / "results/results.json"
    previous = current.read_bytes()
    monkeypatch.setattr(scoring, "validate_weather", Mock())
    monkeypatch.setattr(scoring, "predict", Mock())
    monkeypatch.setattr(scoring.PipelineModel, "load", Mock())
    score = Mock(side_effect=[{"total_records": 2}, OSError("second variant failed")])
    monkeypatch.setattr(scoring, "score_variant", score)
    with pytest.raises(OSError, match="second variant failed"):
        scoring.evaluate(Mock(), tmp_path, run_dir.name)
    assert current.read_bytes() == previous
    failed_attempt, = (run_dir / "test").iterdir()

    score.side_effect = None
    score.return_value = {"total_records": 2}
    completed = scoring.evaluate(Mock(), tmp_path, run_dir.name)
    assert completed["attempt_id"] != failed_attempt.name
    assert set(completed["models"]) == {"unweighted", "weighted"}
    assert json.loads(current.read_text())[run.RESULT_KEY]["test"] == completed

    update_results({run.RESULT_KEY: {"run_id": "new", "training": {}}}, tmp_path / "results")
    published = json.loads(current.read_text())
    assert "test" not in published[run.RESULT_KEY]
    assert published["historical"]["test"] == {"f1": 0.1}


def test_score_variant_saves_flight_identity_and_frozen_threshold(spark, tmp_path):
    day = date(2025, 1, 5)
    scores = spark.createDataFrame([
        ("a", day, 1, 0.4, 1, "AA", "ORD", 8),
        ("b", day, 0, 0.1, 0, "AA", "ORD", 8),
    ], ["flight_id", "FlightDate", "ArrDel15", "p_delayed", "prediction", "Reporting_Airline", "Origin", "DepHour"])
    output = tmp_path / "predictions"
    report = scoring.score_variant(scores, output, 0.25)
    assert report["metrics_at_selected_threshold"]["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert report["metrics_at_default_0_5"]["confusion_matrix"] == {"tp": 0, "fp": 0, "tn": 1, "fn": 1}
    saved = spark.read.parquet(str(output)).orderBy("flight_id").collect()
    assert [(row.flight_id, row.FlightDate, row.ArrDel15, row.threshold) for row in saved] == [
        ("a", day, 1, 0.25), ("b", day, 0, 0.25),
    ]
