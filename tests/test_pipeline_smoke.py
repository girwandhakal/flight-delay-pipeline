"""Exercise orchestration and model publication on synthetic inputs."""

from contextlib import contextmanager
from datetime import date
import json
import os
from unittest.mock import Mock

import pytest
from pyspark.sql import DataFrameReader, functions as F

from pipeline import main as pipeline
from scripts.common.context import PipelineContext
from scripts.common.results import record_stage_timings, update_results
from scripts.common.stage_cache import resolve_stage
from scripts.modeling import evaluate as scoring
from scripts.modeling import run, train

SPLITS = {
    "train": {"start_date": "2024-01-01", "end_date": "2024-09-30"},
    "val": {"start_date": "2024-10-01", "end_date": "2024-11-30"},
    "test": {"start_date": "2024-12-01", "end_date": "2025-01-31"},
}
SOURCES = {"sha256": {}, "rotation": {"raw": []}}


@pytest.mark.unit
def test_main_publishes_validation(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    config = {"splits": SPLITS}
    spark = Mock()
    gold = tmp_path / run.GOLD_DIR / "generation"

    def build_fake_gold(directory):
        for split in SPLITS:
            partition = directory / split
            partition.mkdir()
            (partition / "_SUCCESS").touch()
            (partition / "part.parquet").write_bytes(b"fixture")
        return {}

    gold = resolve_stage(gold.parent, lambda: {}, tuple(SPLITS), (), build_fake_gold)

    def fit(spark, gold, output, splits):
        model = output / "model"
        model.mkdir()
        (model / "metadata").write_text("fixture")
        (output / "threshold.json").write_text('{"threshold": 0.25}')
        selected = {
            "best_threshold": 0.25,
            "validation_metrics_at_selected_threshold": {"f1": 0.54},
            "validation_metrics_at_default_0_5": {"f1": 0.44},
        }
        return {
            "selected_model": selected,
            "input_columns": [],
            "block_time": {},
            "algorithm_comparison": {"logistic_regression": {**selected, "models_dir": str(output)}},
        }

    trainer = Mock(side_effect=fit)

    @contextmanager
    def fake_spark_session(_app_name="flight_delay"):
        try:
            yield spark
        finally:
            spark.stop()

    def make_context():
        return PipelineContext(
            root=tmp_path,
            config=config,
            settings={"reference_directory": tmp_path},
            run_id="test_run_001",
            sources={"sha256": {}},
        )

    bts_run = Mock(side_effect=lambda ctx: setattr(ctx, "gold_dir", tmp_path / "bts"))
    origin_run = Mock(side_effect=lambda ctx: setattr(ctx, "origin_gold_dir", tmp_path / "origin"))
    dest_run = Mock(side_effect=lambda ctx: setattr(ctx, "destination_gold_dir", tmp_path / "destination"))
    rot_run = Mock(side_effect=lambda ctx: setattr(ctx, "rotation_gold_dir", tmp_path / "rotation"))
    enroute_run = Mock(side_effect=lambda ctx: setattr(ctx, "gold_dir", gold))
    fi_run = Mock()
    eval_run = Mock()

    monkeypatch.setattr(pipeline.PipelineContext, "create", staticmethod(make_context))
    monkeypatch.setattr(pipeline, "spark_session", fake_spark_session)
    monkeypatch.setattr(pipeline.bts, "run", bts_run)
    monkeypatch.setattr(pipeline.origin_weather, "run", origin_run)
    monkeypatch.setattr(pipeline.destination_weather, "run", dest_run)
    monkeypatch.setattr(pipeline.aircraft_rotation, "run", rot_run)
    monkeypatch.setattr(pipeline.enroute_weather, "run", enroute_run)
    monkeypatch.setattr(pipeline.feature_importance, "run", fi_run)
    monkeypatch.setattr(pipeline.evaluate, "run", eval_run)
    monkeypatch.setattr(train, "train", trainer)

    assert pipeline.main([]) == 0
    report, = (tmp_path / run.RUNS_DIR).glob("*/validation.json")
    assert list((tmp_path / "models").iterdir()) == [tmp_path / run.RUNS_DIR]
    assert (report.parent / "_SUCCESS").is_file()
    assert json.loads(report.read_text())["validation"]["selected"] == {"f1": 0.54, "threshold": 0.25}
    current = tmp_path / "results/results.json"
    saved = current.read_bytes()
    assert list(json.loads(saved)) == [run.RESULT_KEY]
    trainer.assert_called_once_with(spark, gold, report.parent, SPLITS)
    fi_run.assert_called_once()
    eval_run.assert_called_once()
    spark.stop.assert_called_once()

    # Early stage failure must stop downstream stages while still cleaning up Spark.
    bts_run.side_effect = RuntimeError("bts failed")
    with pytest.raises(RuntimeError, match="bts failed"):
        pipeline.main([])
    origin_run.assert_called_once()
    trainer.assert_called_once()
    assert spark.stop.call_count == 2

    # Publication failure must delete the aborted run directory while preserving existing results and prior run.
    bts_run.side_effect = None
    monkeypatch.setattr(
        pipeline.PipelineContext,
        "create",
        staticmethod(
            lambda: PipelineContext(
                root=tmp_path,
                config=config,
                settings={"reference_directory": tmp_path},
                run_id="test_run_002",
                sources={"sha256": {}},
            )
        ),
    )
    monkeypatch.setattr(run, "update_results", Mock(side_effect=OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        pipeline.main([])
    assert current.read_bytes() == saved
    assert not (tmp_path / run.RUNS_DIR / "test_run_002").exists()
    assert list((tmp_path / run.RUNS_DIR).iterdir()) == [report.parent]
    assert list((tmp_path / run.RUNS_DIR).glob("*/_SUCCESS")) == [report.parent / "_SUCCESS"]
    assert spark.stop.call_count == 3

    timings_file = tmp_path / "results/stage_timings.json"
    assert len(json.loads(timings_file.read_text())) == 1
    record_stage_timings(
        "test_run_003",
        "2026-11-25T15:00:00-06:00",
        {"Stage 1": 12.345, "Stage 6": 4.567},
        17.0,
        tmp_path / "results",
    )
    history = json.loads(timings_file.read_text())
    assert len(history) == 2
    assert history[1]["run_id"] == "test_run_003"
    assert history[1]["total_seconds"] == 17.0
    assert history[1]["stages"] == {"Stage 1": 12.35, "Stage 6": 4.57}


@pytest.mark.unit
def test_model_retention_rollback_and_legacy_cleanup(tmp_path, monkeypatch):
    config = {"splits": SPLITS}
    spark = Mock()

    def build_fake_gold(directory):
        for split in SPLITS:
            partition = directory / split
            partition.mkdir()
            (partition / "_SUCCESS").touch()
            (partition / "part.parquet").write_bytes(b"fixture")
        return {}

    enroute_gold = resolve_stage(tmp_path / run.GOLD_DIR, lambda: {}, tuple(SPLITS), (), build_fake_gold)

    def fit(_spark, _gold, output, _splits):
        model = output / "model"
        model.mkdir()
        (model / "metadata").write_text("fixture")
        (output / "threshold.json").write_text('{"threshold": 0.25}')
        selected = {
            "best_threshold": 0.25,
            "validation_metrics_at_selected_threshold": {"f1": 0.54},
            "validation_metrics_at_default_0_5": {"f1": 0.44},
        }
        return {
            "selected_model": selected,
            "input_columns": [],
            "block_time": {},
            "algorithm_comparison": {"logistic_regression": {**selected, "models_dir": str(output)}},
        }

    models_root = tmp_path / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    (models_root / ".gitkeep").write_text("")
    (models_root / "custom_keep").mkdir()
    for name in run.LEGACY_MODEL_DIRS:
        legacy = models_root / name / "old_run"
        legacy.mkdir(parents=True)
        (legacy / "weights.bin").write_bytes(b"legacy")

    # Verify Windows MAX_PATH (>260 chars) nested Spark metadata trees are measured and pruned safely.
    deep_dir = models_root / "destination_weather_comparison" / ("nested_" * 20) / "metadata"
    os.makedirs(run.fs_path(deep_dir), exist_ok=True)
    with open(run.fs_path(deep_dir / "part-00000.snappy.parquet"), "wb") as handle:
        handle.write(b"deep")

    # Refuse legacy cleanup before any active completed flight_delay run exists.
    with pytest.raises(ValueError, match="active flight_delay run"):
        run.prune_legacy_models(tmp_path, require_active_run=True)
    for name in run.LEGACY_MODEL_DIRS:
        assert (models_root / name).is_dir()

    # Roll back partial run_dir when training fails mid-run.
    monkeypatch.setattr(train, "train", Mock(side_effect=RuntimeError("fit failed")))
    with pytest.raises(RuntimeError, match="fit failed"):
        run.train_and_publish(spark, tmp_path, "run_fail", enroute_gold, SPLITS, {}, SOURCES)
    assert not (tmp_path / run.RUNS_DIR / "run_fail").exists()
    assert (models_root / "dw").is_dir()

    # First successful run removes legacy dirs and keeps only run_001.
    monkeypatch.setattr(train, "train", Mock(side_effect=fit))
    run.train_and_publish(spark, tmp_path, "run_001", enroute_gold, SPLITS, {}, SOURCES)
    assert [p.name for p in (tmp_path / run.RUNS_DIR).iterdir()] == ["run_001"]
    assert (models_root / ".gitkeep").is_file()
    assert (models_root / "custom_keep").is_dir()
    for name in run.LEGACY_MODEL_DIRS:
        assert not (models_root / name).exists()

    # Second successful run replaces run_001 with run_002, leaving strictly one subfolder.
    run.train_and_publish(spark, tmp_path, "run_002", enroute_gold, SPLITS, {}, SOURCES)
    assert [p.name for p in (tmp_path / run.RUNS_DIR).iterdir()] == ["run_002"]
    assert json.loads((tmp_path / "results/results.json").read_text())[run.RESULT_KEY]["run_id"] == "run_002"
    with pytest.raises(ValueError):
        run.prune_model_runs(tmp_path, "../escape")


def test_full_feature_fit_and_frozen_test_score(spark, tmp_path, monkeypatch, full_feature_rows):
    identifiers = [("train", date(2024, 1, 5)), ("val", date(2024, 10, 5)), ("test", date(2025, 1, 5))]

    def build_gold(directory):
        for split, day in identifiers:
            full_feature_rows(split, day).coalesce(1).write.parquet(str(directory / split))
        return {}

    gold = resolve_stage(tmp_path / run.GOLD_DIR, lambda: {}, tuple(SPLITS), (), build_gold)
    monkeypatch.setattr(train, "LOGISTIC_PARAMS", {**train.LOGISTIC_PARAMS, "maxIter": 1})
    update_results({"historical": {"test": {"f1": 0.1}}}, tmp_path / "results")
    actual_parquet = DataFrameReader.parquet
    reads = []

    def track_reads(reader, *paths, **kwargs):
        reads.extend(paths)
        return actual_parquet(reader, *paths, **kwargs)

    monkeypatch.setattr(DataFrameReader, "parquet", track_reads)
    report = run.train_and_publish(spark, tmp_path, "full_features", gold, SPLITS, {}, SOURCES)
    assert set(reads) == {str(gold / "train"), str(gold / "val")}
    algorithms = json.loads(report.read_text())["algorithm_comparison"]
    assert set(algorithms) == {"logistic_regression", train.WEIGHTED_LOGISTIC}
    assert algorithms[train.WEIGHTED_LOGISTIC]["class_weights"] == pytest.approx({"on_time": 0.75, "delayed": 1.5})
    reads.clear()
    summary = scoring.evaluate(spark, tmp_path, "full_features")
    assert reads == [str(gold / "test")]

    test_ids = spark.read.parquet(str(gold / "test")).select("flight_id", "FlightDate", train.TARGET_COL)
    for name, model in summary["models"].items():
        threshold = algorithms[name]["best_threshold"]
        predictions = spark.read.parquet(str(tmp_path / model["predictions"]))
        assert predictions.select("threshold").distinct().first().threshold == threshold
        assert predictions.where(F.col("prediction") != (F.col("p_delayed") >= threshold).cast("int")).count() == 0
        assert sorted(predictions.select(*test_ids.columns).collect()) == sorted(test_ids.collect())
        assert set(algorithms[name]["validation_per_class_at_selected_threshold"]) == {"on_time", "delayed"}
    published = json.loads((tmp_path / "results/results.json").read_text())
    assert published[run.RESULT_KEY]["test"] == summary
    assert published["historical"]["test"] == {"f1": 0.1}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "expect_benchmark", "expect_ablation", "expected_sessions"),
    [
        ([], False, False, ["flight_delay"]),
        (["--benchmark"], True, False, ["flight_delay"]),
        (["--ablation"], False, True, ["flight_delay"]),
        (["--benchmark", "--ablation"], True, True, ["flight_delay", "flight_delay_ablation"]),
    ],
)
def test_pipeline_four_flag_combinations(
    tmp_path, monkeypatch, argv, expect_benchmark, expect_ablation, expected_sessions
):
    """Verify Phase 1, Phase 2 (benchmark), Phase 3 (ablation), and Spark session lifecycle across all 4 modes."""
    sessions_opened: list[str] = []
    call_order: list[str] = []
    spark = Mock()

    @contextmanager
    def fake_spark_session(app_name="flight_delay"):
        sessions_opened.append(app_name)
        call_order.append(f"spark_enter:{app_name}")
        try:
            yield spark
        finally:
            spark.stop()
            call_order.append(f"spark_exit:{app_name}")

    gold_dir = tmp_path / "data" / "gold" / "enroute" / "gen_001"
    gold_dir.mkdir(parents=True)
    report_path = tmp_path / "models" / "flight_delay" / "run_001" / "validation.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}")

    def make_context():
        return PipelineContext(
            root=tmp_path,
            config={"splits": SPLITS, "benchmark": {"workers": ["local[2]"], "data_fractions": [1.0], "fixed_worker_for_size_scaling": "local[2]", "seed": 42}},
            settings={"reference_directory": tmp_path},
            run_id="run_001",
            sources={"sha256": {}},
        )

    monkeypatch.setattr(pipeline.PipelineContext, "create", staticmethod(make_context))
    monkeypatch.setattr(pipeline, "spark_session", fake_spark_session)
    monkeypatch.setattr(pipeline.bts, "run", lambda ctx: call_order.append("bts"))
    monkeypatch.setattr(pipeline.origin_weather, "run", lambda ctx: call_order.append("origin_weather"))
    monkeypatch.setattr(pipeline.destination_weather, "run", lambda ctx: call_order.append("destination_weather"))
    monkeypatch.setattr(pipeline.aircraft_rotation, "run", lambda ctx: call_order.append("aircraft_rotation"))
    monkeypatch.setattr(
        pipeline.enroute_weather,
        "run",
        lambda ctx: (setattr(ctx, "gold_dir", gold_dir), call_order.append("enroute_weather")),
    )
    monkeypatch.setattr(pipeline.modeling, "run", lambda ctx: (call_order.append("modeling"), report_path)[1])
    monkeypatch.setattr(pipeline.feature_importance, "run", lambda ctx: call_order.append("feature_importance"))
    monkeypatch.setattr(pipeline.evaluate, "run", lambda ctx: call_order.append("evaluate"))

    bench_mock = Mock(side_effect=lambda **kwargs: call_order.append("benchmark"))
    ablation_mock = Mock(side_effect=lambda ctx: call_order.append("ablation"))
    monkeypatch.setattr(pipeline, "run_benchmark_suite", bench_mock)
    monkeypatch.setattr(pipeline.ablation, "run_ablation", ablation_mock)

    assert pipeline.main(argv) == 0
    assert sessions_opened == expected_sessions
    assert spark.stop.call_count == len(expected_sessions)
    assert bench_mock.called is expect_benchmark
    assert ablation_mock.called is expect_ablation

    if expect_benchmark:
        assert call_order.index("spark_exit:flight_delay") < call_order.index("benchmark")
        assert bench_mock.call_args.kwargs["gold_dir"] == gold_dir
    if expect_benchmark and expect_ablation:
        assert call_order.index("benchmark") < call_order.index("spark_enter:flight_delay_ablation")
        assert call_order.index("spark_enter:flight_delay_ablation") < call_order.index("ablation")
    elif expect_ablation:
        assert call_order.index("evaluate") < call_order.index("ablation") < call_order.index("spark_exit:flight_delay")

    timings = json.loads((tmp_path / "results" / "stage_timings.json").read_text(encoding="utf-8"))
    assert len(timings) == 1
    assert timings[0]["run_id"] == "run_001"
    assert ("benchmark" in timings[0]["stages"]) is expect_benchmark


@pytest.mark.unit
def test_pipeline_force_retrain_flag_and_cache_hit_skips(tmp_path, monkeypatch):
    """Verify --force-retrain flag and Phase 1 cache-hit reuse of feature_importance and evaluate."""
    assert pipeline.parse_args([]).force_retrain is False
    assert pipeline.parse_args(["--force-retrain"]).force_retrain is True

    spark = Mock()

    @contextmanager
    def fake_spark_session(app_name="flight_delay"):
        yield spark

    gold_dir = tmp_path / "data" / "gold" / "enroute" / "gen_001"
    gold_dir.mkdir(parents=True)
    cached_run_id = "cached_run_999"
    report_path = tmp_path / "models" / "flight_delay" / cached_run_id / "validation.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}")

    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / "feature_importance.json").write_text(
        json.dumps({"run_id": cached_run_id}), encoding="utf-8"
    )
    (tmp_path / "results" / "results.json").write_text(
        json.dumps({"flight_delay": {"run_id": cached_run_id, "test": {"run_id": cached_run_id}}}),
        encoding="utf-8",
    )

    seen_flags = []

    def make_context():
        return PipelineContext(
            root=tmp_path,
            config={"splits": SPLITS},
            settings={"reference_directory": tmp_path},
            run_id="fresh_run_001",
            sources={"sha256": {}},
        )

    def fake_modeling_run(ctx):
        seen_flags.append(ctx.force_retrain)
        ctx.run_id = cached_run_id
        return report_path

    fi_mock = Mock()
    eval_mock = Mock()
    ablation_mock = Mock()

    monkeypatch.setattr(pipeline.PipelineContext, "create", staticmethod(make_context))
    monkeypatch.setattr(pipeline, "spark_session", fake_spark_session)
    monkeypatch.setattr(pipeline.bts, "run", lambda ctx: None)
    monkeypatch.setattr(pipeline.origin_weather, "run", lambda ctx: None)
    monkeypatch.setattr(pipeline.destination_weather, "run", lambda ctx: None)
    monkeypatch.setattr(pipeline.aircraft_rotation, "run", lambda ctx: None)
    monkeypatch.setattr(pipeline.enroute_weather, "run", lambda ctx: setattr(ctx, "gold_dir", gold_dir))
    monkeypatch.setattr(pipeline.modeling, "run", fake_modeling_run)
    monkeypatch.setattr(pipeline.feature_importance, "run", fi_mock)
    monkeypatch.setattr(pipeline.evaluate, "run", eval_mock)
    monkeypatch.setattr(pipeline.ablation, "run_ablation", ablation_mock)

    # Without --force-retrain: matching feature_importance & test eval are skipped, ablation gets cached run_id.
    assert pipeline.main(["--ablation"]) == 0
    assert seen_flags[-1] is False
    fi_mock.assert_not_called()
    eval_mock.assert_not_called()
    assert ablation_mock.call_args.args[0].run_id == cached_run_id

    # With --force-retrain: feature_importance and evaluate both run.
    assert pipeline.main(["--force-retrain"]) == 0
    assert seen_flags[-1] is True
    fi_mock.assert_called_once()
    eval_mock.assert_called_once()
