"""Exercise orchestration and model publication on synthetic inputs."""

from contextlib import contextmanager
from datetime import date
import json
from unittest.mock import Mock

import pytest
from pyspark.sql import DataFrameReader, functions as F

from pipeline import main as pipeline
from scripts.common.context import PipelineContext
from scripts.common.results import update_results
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

    monkeypatch.setattr(pipeline.PipelineContext, "create", staticmethod(make_context))
    monkeypatch.setattr(pipeline, "spark_session", fake_spark_session)
    monkeypatch.setattr(pipeline.bts, "run", bts_run)
    monkeypatch.setattr(pipeline.origin_weather, "run", origin_run)
    monkeypatch.setattr(pipeline.destination_weather, "run", dest_run)
    monkeypatch.setattr(pipeline.aircraft_rotation, "run", rot_run)
    monkeypatch.setattr(pipeline.enroute_weather, "run", enroute_run)
    monkeypatch.setattr(train, "train", trainer)

    assert pipeline.main() == 0
    report, = (tmp_path / run.RUNS_DIR).glob("*/validation.json")
    assert list((tmp_path / "models").iterdir()) == [tmp_path / run.RUNS_DIR]
    assert (report.parent / "_SUCCESS").is_file()
    assert json.loads(report.read_text())["validation"]["selected"] == {"f1": 0.54, "threshold": 0.25}
    current = tmp_path / "results/results.json"
    saved = current.read_bytes()
    assert list(json.loads(saved)) == [run.RESULT_KEY]
    trainer.assert_called_once_with(spark, gold, report.parent, SPLITS)
    spark.stop.assert_called_once()

    # Early stage failure must stop downstream stages while still cleaning up Spark.
    bts_run.side_effect = RuntimeError("bts failed")
    with pytest.raises(RuntimeError, match="bts failed"):
        pipeline.main()
    origin_run.assert_called_once()
    trainer.assert_called_once()
    assert spark.stop.call_count == 2

    # Publication failure must preserve existing results.json and remove the _SUCCESS marker.
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
        pipeline.main()
    assert current.read_bytes() == saved
    assert list((tmp_path / run.RUNS_DIR).glob("*/_SUCCESS")) == [report.parent / "_SUCCESS"]
    assert spark.stop.call_count == 3


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
