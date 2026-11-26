"""Unit tests for controlled feature ablation (Issue #5)."""

from pathlib import Path
import json
from datetime import date

import pytest

from pipeline.main import parse_args
from scripts.common.context import PipelineContext
from scripts.common.results import update_results
from scripts.modeling import ablation, train
from scripts.modeling.train import ABLATION_GROUPS, build_preprocessor


def test_ablation_groups_definition() -> None:
    """Verify the four progressive ablation groups and their flags."""
    assert tuple(ABLATION_GROUPS) == (
        "bts_only",
        "bts_surface_weather",
        "bts_surface_rotation",
        "full",
    )
    assert ABLATION_GROUPS["bts_only"] == {"surface_weather": False, "rotation": False, "enroute": False}
    assert ABLATION_GROUPS["bts_surface_weather"] == {"surface_weather": True, "rotation": False, "enroute": False}
    assert ABLATION_GROUPS["bts_surface_rotation"] == {"surface_weather": True, "rotation": True, "enroute": False}
    assert ABLATION_GROUPS["full"] == {"surface_weather": True, "rotation": True, "enroute": True}


def test_build_preprocessor_feature_groups_and_invalid(spark) -> None:
    """Verify build_preprocessor accepts all ablation groups and rejects unknown ones."""
    dims = {}
    for group in ABLATION_GROUPS:
        pipeline = build_preprocessor(feature_group=group)
        assembler = pipeline.getStages()[-1]
        dims[group] = len(assembler.getInputCols())
    assert dims["bts_only"] < dims["bts_surface_weather"] < dims["bts_surface_rotation"] < dims["full"]
    with pytest.raises(ValueError, match="Unknown feature_group"):
        build_preprocessor(feature_group="invalid_group")


def test_pipeline_main_ablation_flag() -> None:
    """Verify --ablation CLI flag parsing in pipeline/main.py."""
    assert parse_args([]).ablation is False
    assert parse_args(["--ablation"]).ablation is True


def test_run_ablation_reuses_phase1_full_and_publishes(spark, tmp_path: Path, full_feature_rows, monkeypatch) -> None:
    """Verify run_ablation reuses Phase 1 full metrics and publishes under results.json['ablation']."""
    gold_dir = tmp_path / "data" / "gold" / "enroute" / "gen_001"
    for split, day in [("train", date(2024, 1, 5)), ("val", date(2024, 10, 5)), ("test", date(2025, 1, 5))]:
        full_feature_rows(split, day).coalesce(1).write.parquet(str(gold_dir / split))
    (gold_dir / "_SUCCESS").touch()

    monkeypatch.setattr(train, "LOGISTIC_PARAMS", {**train.LOGISTIC_PARAMS, "maxIter": 1})
    phase1_val = {"f1": 0.61, "precision": 0.6, "recall": 0.62, "accuracy": 0.7, "tp": 6, "fp": 4, "tn": 7, "fn": 3, "pr_auc": 0.65, "roc_auc": 0.72}
    phase1_test = {"f1": 0.59, "precision": 0.58, "recall": 0.6, "accuracy": 0.68, "tp": 6, "fp": 4, "tn": 6, "fn": 4, "pr_auc": 0.63, "roc_auc": 0.7}
    update_results(
        {
            "flight_delay": {
                "run_id": "run_ablation_test",
                "training": {
                    "vector_dimension": 123,
                    "algorithm_comparison": {
                        train.WEIGHTED_LOGISTIC: {
                            "best_threshold": 0.45,
                            "validation_metrics_at_selected_threshold": phase1_val,
                            "validation_metrics_at_default_0_5": phase1_val,
                        }
                    },
                },
                "test": {
                    "models": {
                        train.WEIGHTED_LOGISTIC: {
                            "selected_threshold": 0.45,
                            "metrics_at_selected_threshold": phase1_test,
                            "metrics_at_default_0_5": phase1_test,
                        }
                    }
                },
            }
        },
        tmp_path / "results",
    )

    ctx = PipelineContext(
        root=tmp_path,
        config={},
        settings={},
        run_id="run_ablation_test",
        spark=spark,
        gold_dir=gold_dir,
    )
    payload = ablation.run_ablation(ctx)
    assert payload["reused_phase1_full"] is True
    assert payload["feature_groups"]["full"]["vector_dimension"] == 123
    assert payload["feature_groups"]["full"]["algorithms"][train.WEIGHTED_LOGISTIC]["reused_from_run_id"] == "run_ablation_test"
    assert payload["feature_groups"]["full"]["algorithms"][train.WEIGHTED_LOGISTIC]["test_metrics_at_selected_threshold"] == phase1_test
    assert "ablation" in ctx.timings

    published = json.loads((tmp_path / "results" / "results.json").read_text(encoding="utf-8"))
    assert set(published["ablation"]["feature_groups"]) == set(ABLATION_GROUPS)
    assert published["flight_delay"]["run_id"] == "run_ablation_test"
