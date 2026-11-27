"""Unit tests for the modular reporting figure generator and pipeline hook."""

import csv
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline import main as pipeline
from scripts.common.context import PipelineContext
from scripts.reporting import figures


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _seed_synthetic_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "flight_delay": {
            "training": {
                "algorithm_comparison": {
                    "logistic_regression": {
                        "validation_metrics_at_selected_threshold": {"pr_auc": 0.58, "roc_auc": 0.81, "f1": 0.54}
                    },
                    "random_forest": {
                        "validation_metrics_at_selected_threshold": {"pr_auc": 0.59, "roc_auc": 0.80, "f1": 0.53}
                    },
                }
            },
            "test": {
                "models": {
                    "logistic_regression": {
                        "metrics_at_selected_threshold": {"pr_auc": 0.66, "roc_auc": 0.82, "f1": 0.59},
                        "reference_baselines": {
                            "always_ontime": {"pr_auc": 0.20},
                            "always_delayed": {"f1": 0.33},
                        },
                        "subgroups": [
                            {"subgroup_type": "departure_hour", "subgroup_value": "5", "total_flights": 30000, "prevalence": 0.12, "f1": 0.18},
                            {"subgroup_type": "departure_hour", "subgroup_value": "17", "total_flights": 70000, "prevalence": 0.24, "f1": 0.66},
                        ],
                    },
                    "random_forest": {
                        "metrics_at_selected_threshold": {"pr_auc": 0.67, "roc_auc": 0.82, "f1": 0.58},
                    },
                }
            },
        },
        "ablation": {
            "feature_groups": {
                "bts_only": {
                    "vector_dimension": 783,
                    "algorithms": {
                        "logistic_regression": {
                            "validation_metrics_at_selected_threshold": {"pr_auc": 0.29, "roc_auc": 0.67, "f1": 0.33},
                            "test_metrics_at_selected_threshold": {"pr_auc": 0.41, "roc_auc": 0.69, "f1": 0.42},
                        }
                    },
                },
                "full": {
                    "vector_dimension": 892,
                    "algorithms": {
                        "logistic_regression": {
                            "validation_metrics_at_selected_threshold": {"pr_auc": 0.58, "roc_auc": 0.81, "f1": 0.54},
                            "test_metrics_at_selected_threshold": {"pr_auc": 0.66, "roc_auc": 0.82, "f1": 0.59},
                        }
                    },
                },
            }
        },
    }
    (results_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")

    benchmark = {
        "data_size_scaling": [
            {"data_fraction": 0.1, "wall_time_seconds": 100.0, "peak_memory_gb": 4.5},
            {"data_fraction": 1.0, "wall_time_seconds": 280.0, "peak_memory_gb": 7.5},
        ],
        "worker_scaling": [
            {"workers": "local[2]", "wall_time_seconds": 390.0, "speedup": 1.0},
            {"workers": "local[4]", "wall_time_seconds": 280.0, "speedup": 1.4},
        ],
    }
    (results_dir / "benchmark_results.json").write_text(json.dumps(benchmark), encoding="utf-8")

    _write_csv(
        results_dir / "feature_importance_groups.csv",
        ["model", "feature_group", "total", "share", "feature_count"],
        [
            {"model": "logistic_regression", "feature_group": "bts_schedule", "total": 10.0, "share": 0.89, "feature_count": 781},
            {"model": "logistic_regression", "feature_group": "aircraft_rotation_slack", "total": 1.0, "share": 0.04, "feature_count": 10},
            {"model": "random_forest", "feature_group": "bts_schedule", "total": 0.22, "share": 0.22, "feature_count": 19},
            {"model": "random_forest", "feature_group": "aircraft_rotation_slack", "total": 0.69, "share": 0.69, "feature_count": 10},
        ],
    )
    _write_csv(
        results_dir / "feature_importance_random_forest.csv",
        ["index", "feature_name", "base_column", "feature_group", "attr_type", "importance"],
        [
            {"index": 1, "feature_name": "rotation_outcome_missing", "base_column": "rotation_outcome_missing", "feature_group": "aircraft_rotation_slack", "attr_type": "numeric", "importance": 0.18},
            {"index": 2, "feature_name": "origin_departure_delay_rate_2h", "base_column": "origin_departure_delay_rate_2h", "feature_group": "bts_schedule", "attr_type": "numeric", "importance": 0.04},
        ],
    )
    _write_csv(
        results_dir / "root_cause" / "cohort_overview.csv",
        ["cohort", "flights", "delayed_flights", "delay_rate", "mean_arr_delay_min", "share_of_total"],
        [
            {"cohort": "usable_clean", "flights": 1000, "delayed_flights": 100, "delay_rate": 0.10, "mean_arr_delay_min": -4.0, "share_of_total": 0.60},
            {"cohort": "prior_outcome_unavailable", "flights": 100, "delayed_flights": 99, "delay_rate": 0.99, "mean_arr_delay_min": 107.0, "share_of_total": 0.04},
        ],
    )
    _write_csv(
        results_dir / "root_cause" / "bts_causes_by_cohort.csv",
        [
            "cohort_group",
            "LateAircraftDelay_primary_share",
            "CarrierDelay_primary_share",
            "NASDelay_primary_share",
            "WeatherDelay_primary_share",
            "SecurityDelay_primary_share",
        ],
        [
            {
                "cohort_group": "usable_clean",
                "LateAircraftDelay_primary_share": 0.06,
                "CarrierDelay_primary_share": 0.44,
                "NASDelay_primary_share": 0.46,
                "WeatherDelay_primary_share": 0.04,
                "SecurityDelay_primary_share": 0.00,
            },
            {
                "cohort_group": "prior_outcome_unavailable",
                "LateAircraftDelay_primary_share": 0.85,
                "CarrierDelay_primary_share": 0.08,
                "NASDelay_primary_share": 0.04,
                "WeatherDelay_primary_share": 0.03,
                "SecurityDelay_primary_share": 0.00,
            },
        ],
    )


@pytest.mark.unit
def test_generate_all_figures_creates_pngs(tmp_path):
    results_dir = tmp_path / "results"
    _seed_synthetic_results(results_dir)

    written = figures.generate_all_figures(results_dir)
    assert len(written) == 5
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0


@pytest.mark.unit
def test_generate_all_figures_skips_missing_sources(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    written = figures.generate_all_figures(results_dir)
    assert written == []


@pytest.mark.unit
def test_pipeline_reporting_only_skips_spark(tmp_path, monkeypatch):
    _seed_synthetic_results(tmp_path / "results")
    spark_mock = Mock(side_effect=AssertionError("Spark must not start in --reporting-only mode"))
    monkeypatch.setattr(pipeline, "spark_session", spark_mock)
    monkeypatch.setattr(
        pipeline.PipelineContext,
        "create",
        staticmethod(lambda: PipelineContext(root=tmp_path, config={}, settings={}, run_id="report_only_001", sources={})),
    )

    assert pipeline.main(["--reporting-only"]) == 0
    assert len(list((tmp_path / "results" / "figures").glob("*.png"))) == 5
    spark_mock.assert_not_called()
