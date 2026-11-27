"""Check model metrics, threshold selection, and the full feature vector."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import pytest
from pyspark.sql import DataFrameReader, functions as F

from scripts.enroute_weather import build_features as features
from scripts.modeling import train
from tests.conftest import get_test_spark_session


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark_session("test_model_training")

    def test_evaluation_matches_hand_counts(self):
        data = self.spark.createDataFrame([(1, 0.9, 1), (0, 0.8, 1), (1, 0.6, 1), (0, 0.2, 0), (1, 0.1, 0)],
                                          [train.TARGET_COL, "p_delayed", "prediction"])
        result = train.evaluate_predictions(data)
        self.assertEqual(result["confusion_matrix"], {"tp": 2, "fp": 1, "tn": 1, "fn": 1})
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertGreaterEqual(result["pr_auc"], 0)
        self.assertLessEqual(result["roc_auc"], 1)
        self.assertEqual(train.confusion_metrics(0, 0, 3, 2)["precision"], 0.0)
        self.assertEqual(train.confusion_metrics(0, 2, 3, 0)["recall"], 0.0)
        self.assertEqual(train.confusion_metrics(0, 0, 3, 0)["f1"], 0.0)
        self.assertEqual(train.confusion_metrics(0, 0, 0, 0)["accuracy"], 0.0)
        single_class = train.evaluate_predictions(data.filter("ArrDel15 = 1"))
        self.assertIsNone(single_class["pr_auc"])
        self.assertIsNone(single_class["roc_auc"])

    def test_threshold_maximum_and_tie_rule(self):
        data = self.spark.createDataFrame([(1, 0.8), (1, 0.8), (0, 0.2), (0, 0.2)], [train.TARGET_COL, "p_delayed"])
        with patch.object(data, "agg", wraps=data.agg) as aggregate:
            threshold, search = train.tune_threshold(data)
        self.assertEqual(aggregate.call_count, 1)
        self.assertEqual(threshold, 0.8)
        best = search.loc[search["threshold"] == threshold].iloc[0]
        self.assertEqual(best["f1"], 1.0)
        self.assertEqual(int(best["tp"]), 2)
        self.assertEqual(int(best["fp"]), 0)
        self.assertEqual(search["f1"].max(), best["f1"])


def test_preprocessor_appends_enroute_columns_and_uses_training_only_medians(spark):
    weather, flags = train.SURFACE_NUMERIC, train.SURFACE_FLAGS
    empty = "enroute_sigmet_max_top_fl"
    data = spark.range(3).select(
        *[F.lit("AA").alias(name) for name in train.CATEGORICAL_FEATURES],
        F.lit(1).alias("Month"),
        *[F.lit(1.0).alias(name) for name in train.NUMERIC_FEATURES + train.ROLLING_FEATURES
          + weather + list(train.ROTATION_NUMERIC)],
        *[F.lit(0).alias(name) for name in train.HOLIDAY_FEATURES + flags
          + list(train.ROTATION_FLAGS) + list(features.ENROUTE_FLAGS)],
        *[F.lit(None).cast("double").alias(pair[0]) for pair in (train.SLACK_FEATURES, train.RECOVERY_FEATURES)],
        *[F.lit(1).alias(pair[1]) for pair in (train.SLACK_FEATURES, train.RECOVERY_FEATURES)],
        *[F.when(F.col("id") < 2, 30.0).alias(name) if name != empty else F.lit(None).cast("double").alias(name)
          for name in features.ENROUTE_NUMERIC],
        "id",
    )
    model = train.build_preprocessor(empty_enroute=(empty,)).fit(data.where("id < 2"))
    assert model.stages[-1].getInputCols() == [
        "Reporting_Airline_vec", "Origin_vec", "Dest_vec", "DayOfWeek_vec", "DepHour_vec", "ArrHour_vec",
        "CRSElapsedTime", "Distance", "MonthSin", "MonthCos",
        *[f"{name}_filled" for name in train.ROLLING_FEATURES], *train.HOLIDAY_FEATURES,
        *[f"{name}_imputed" for name in train.WEATHER_NUMERIC], *train.WEATHER_FLAGS,
        *[f"{name}_imputed" for name in train.DESTINATION_WEATHER_NUMERIC], *train.DESTINATION_WEATHER_FLAGS,
        *[f"{name}_filled" for name in train.ROTATION_NUMERIC], *train.ROTATION_FLAGS,
        *[f"{name}_filled" for name in features.FEATURES],
        f"{train.SLACK_FEATURES[0]}_filled", train.SLACK_FEATURES[1],
        f"{train.RECOVERY_FEATURES[0]}_filled", train.RECOVERY_FEATURES[1],
    ]
    row = model.transform(data.where("id = 2")).first()
    assert row.dest_taf_ceiling_m_filled == 30.0
    assert row.enroute_sigmet_max_top_fl_filled == 0.0
    assert list(row.features)[-4:] == [0.0, 1.0, 0.0, 1.0]



def test_training_reads_only_development_and_persists_both_variants(spark, tmp_path, monkeypatch, full_feature_rows):
    data = {
        "train": full_feature_rows("train", date(2024, 1, 5)),
        "val": full_feature_rows("val", date(2024, 10, 5)),
    }

    def read_data(_reader, path):
        return data[Path(path).name]

    monkeypatch.setattr(DataFrameReader, "parquet", read_data)
    monkeypatch.setattr(train, "LOGISTIC_PARAMS", {**train.LOGISTIC_PARAMS, "maxIter": 1})
    monkeypatch.setattr(train, "MLP_PARAMS", {**train.MLP_PARAMS, "maxIter": 1})
    summary = train.train(spark, tmp_path / "gold", tmp_path / "model", {})
    algorithms = summary["algorithm_comparison"]
    assert set(algorithms) == {"logistic_regression", train.WEIGHTED_LOGISTIC, train.RANDOM_FOREST, train.MLP}
    assert algorithms[train.MLP]["layers"][0] > 0
    assert algorithms[train.MLP]["layers"][-1] == 2
    assert algorithms[train.WEIGHTED_LOGISTIC]["class_weights"] == pytest.approx({"on_time": 0.75, "delayed": 1.5})
    for reports in algorithms.values():
        model = Path(reports["models_dir"])
        assert (model / "model").is_dir()
        assert (model / "validation_predictions").is_dir()
        assert (model / "threshold.json").is_file()
        assert set(reports["validation_per_class_at_selected_threshold"]) == {"on_time", "delayed"}


def _build_fake_cached_run(
    root: Path,
    run_id: str,
    gold_dir: Path,
    splits: dict,
    *,
    include_cache_key: bool = True,
    cache_key: str = "",
) -> Path:
    """Create a minimal verified model run directory on disk for cache-hit tests."""
    from scripts.common.io import write_json
    from scripts.modeling import run as modeling_run

    run_dir = root / "models" / "flight_delay" / run_id
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata").write_text("{}", encoding="utf-8")
    threshold_path = run_dir / "threshold.json"
    write_json(threshold_path, {"threshold": 0.25})
    write_json(run_dir / "validation.json", {"run_id": run_id, "algorithm_comparison": {}})

    gold_id = modeling_run.gold_identity(gold_dir, ("train", "val", "test"))
    manifest = {
        "run_id": run_id,
        "splits": splits,
        "gold": gold_dir.relative_to(root.resolve()).as_posix(),
        "gold_identity": gold_id,
        "input_columns": list(train.INPUT_COLUMNS),
        "sources": {"sha256": {}},
        "variants": {
            "logistic_regression": {
                "model": "model",
                "threshold_file": "threshold.json",
                "threshold": 0.25,
                "sha256": modeling_run.inference_hashes(run_dir, "model", "threshold.json"),
            }
        },
    }
    if include_cache_key:
        manifest["training_cache_key"] = cache_key or modeling_run.training_cache_key(gold_id, splits)
    write_json(run_dir / "run.json", manifest)
    (run_dir / "_SUCCESS").touch()
    write_json(root / "results" / "results.json", {"flight_delay": {"run_id": run_id}})
    return run_dir


def _build_fake_gold(root: Path) -> Path:
    """Create minimal en-route Gold partitions and _stage.json via resolve_stage."""
    from scripts.common.stage_cache import resolve_stage

    def build_gold(directory: Path) -> dict:
        for split in ("train", "val", "test"):
            split_dir = directory / split
            split_dir.mkdir()
            (split_dir / "part-00000.parquet").write_bytes(b"fixture")
            (split_dir / "_SUCCESS").touch()
        return {}

    return resolve_stage(root / "data" / "gold" / "enroute", lambda: {}, ("train", "val", "test"), (), build_gold)


def test_model_cache_hit_and_legacy_bootstrap(tmp_path: Path, monkeypatch) -> None:
    """Verify cache hit for both modern and legacy (no training_cache_key) runs without calling train.train."""
    from scripts.common.context import PipelineContext
    from scripts.modeling import run as modeling_run

    gold_dir = _build_fake_gold(tmp_path)
    splits = {"train": {"start_date": "2024-01-01", "end_date": "2024-09-30"}}

    # Legacy run without training_cache_key (mirrors 20261127T000608624406Z_9783a601).
    _build_fake_cached_run(tmp_path, "legacy_run_001", gold_dir, splits, include_cache_key=False)
    monkeypatch.setattr(train, "train", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not train")))

    ctx = PipelineContext(root=tmp_path, config={"splits": splits}, settings={}, run_id="fresh_run", gold_dir=gold_dir)
    report = modeling_run.run(ctx)
    assert ctx.run_id == "legacy_run_001"
    assert report == tmp_path / "models" / "flight_delay" / "legacy_run_001" / "validation.json"

    # Modern run with matching training_cache_key.
    _build_fake_cached_run(tmp_path, "modern_run_002", gold_dir, splits, include_cache_key=True)
    ctx2 = PipelineContext(root=tmp_path, config={"splits": splits}, settings={}, run_id="fresh_run_2", gold_dir=gold_dir)
    report2 = modeling_run.run(ctx2)
    assert ctx2.run_id == "modern_run_002"
    assert report2 == tmp_path / "models" / "flight_delay" / "modern_run_002" / "validation.json"


def test_model_cache_invalidation_and_force_retrain(tmp_path: Path, monkeypatch) -> None:
    """Verify cache miss on key mismatch, tampered model hash, or force_retrain=True."""
    from scripts.modeling import run as modeling_run

    gold_dir = _build_fake_gold(tmp_path)
    splits = {"train": {"start_date": "2024-01-01", "end_date": "2024-09-30"}}
    run_dir = _build_fake_cached_run(tmp_path, "cached_run_001", gold_dir, splits, include_cache_key=True)

    gold_id = modeling_run.gold_identity(gold_dir, ("train", "val", "test"))
    expected_key = modeling_run.training_cache_key(gold_id, splits)
    assert modeling_run._find_cached_run(tmp_path, gold_id, splits, expected_key) is not None

    # Cache key mismatch invalidates modern run.
    assert modeling_run._find_cached_run(tmp_path, gold_id, splits, "wrong_key") is None

    # Tampered model artifact invalidates cached run.
    (run_dir / "model" / "metadata").write_text("tampered", encoding="utf-8")
    assert modeling_run._find_cached_run(tmp_path, gold_id, splits, expected_key) is None

    # force_retrain=True bypasses even a valid cache.
    _build_fake_cached_run(tmp_path, "cached_run_002", gold_dir, splits, include_cache_key=True)
    called = []
    monkeypatch.setattr(train, "train", lambda *args, **kwargs: called.append(True) or (_ for _ in ()).throw(RuntimeError("retrain_invoked")))
    with pytest.raises(RuntimeError, match="retrain_invoked"):
        modeling_run.train_and_publish(None, tmp_path, "new_run_003", gold_dir, splits, {}, {}, force_retrain=True)
    assert called == [True]
