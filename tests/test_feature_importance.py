"""Unit tests for feature importance and coefficient extraction (RQ3)."""

import csv
import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import Row

from scripts.common.context import PipelineContext
from scripts.modeling import feature_importance


def _build_synthetic_df(spark):
    rows = [
        Row(
            Origin="ORD",
            CRSElapsedTime=120.0,
            origin_departure_delay_rate_2h=0.45,
            origin_temperature_c=15.0,
            outbound_schedule_slack_minutes=30.0,
            dest_taf_ceiling_m=0.0,
            ArrDel15=1,
        ),
        Row(
            Origin="ATL",
            CRSElapsedTime=90.0,
            origin_departure_delay_rate_2h=0.10,
            origin_temperature_c=22.0,
            outbound_schedule_slack_minutes=50.0,
            dest_taf_ceiling_m=1.5,
            ArrDel15=0,
        ),
        Row(
            Origin="DFW",
            CRSElapsedTime=150.0,
            origin_departure_delay_rate_2h=0.35,
            origin_temperature_c=18.0,
            outbound_schedule_slack_minutes=20.0,
            dest_taf_ceiling_m=2.0,
            ArrDel15=1,
        ),
        Row(
            Origin="ORD",
            CRSElapsedTime=60.0,
            origin_departure_delay_rate_2h=0.05,
            origin_temperature_c=25.0,
            outbound_schedule_slack_minutes=75.0,
            dest_taf_ceiling_m=0.0,
            ArrDel15=0,
        ),
    ]
    return spark.createDataFrame(rows)


def test_logistic_and_rf_extraction_and_export(spark, tmp_path: Path) -> None:
    """Verify metadata mapping, group aggregation, and JSON/CSV export for LR and RF."""
    df = _build_synthetic_df(spark)
    numeric_cols = [
        "CRSElapsedTime",
        "origin_departure_delay_rate_2h",
        "origin_temperature_c",
        "outbound_schedule_slack_minutes",
        "dest_taf_ceiling_m",
    ]

    lr_pipeline = Pipeline(stages=[
        StringIndexer(inputCol="Origin", outputCol="Origin_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["Origin_idx"], outputCols=["Origin_vec"], handleInvalid="keep", dropLast=False),
        VectorAssembler(inputCols=["Origin_vec", *numeric_cols], outputCol="features"),
        LogisticRegression(featuresCol="features", labelCol="ArrDel15", regParam=0.1, maxIter=10),
    ])
    lr_model = lr_pipeline.fit(df)

    rf_pipeline = Pipeline(stages=[
        StringIndexer(inputCol="Origin", outputCol="Origin_idx", handleInvalid="keep"),
        VectorAssembler(inputCols=["Origin_idx", *numeric_cols], outputCol="features"),
        RandomForestClassifier(featuresCol="features", labelCol="ArrDel15", numTrees=5, maxDepth=3, seed=42),
    ])
    rf_model = rf_pipeline.fit(df)

    run_id = "test_run_001"
    run_dir = tmp_path / "models" / "flight_delay" / run_id
    lr_model.write().overwrite().save(str(run_dir / "model"))
    rf_model.write().overwrite().save(str(run_dir / "random_forest" / "model"))

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(
        json.dumps({"flight_delay": {"run_id": run_id}}),
        encoding="utf-8",
    )

    out_json = feature_importance.analyze_run(
        spark,
        root=tmp_path,
        run_id=run_id,
        results_dir=results_dir,
        top_n=5,
    )

    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert set(payload["models"].keys()) == {"logistic_regression", "random_forest"}

    lr_groups = {g["feature_group"]: g for g in payload["models"]["logistic_regression"]["group_totals"]}
    assert set(lr_groups.keys()) == {
        "bts_schedule",
        "surface_weather",
        "aircraft_rotation_slack",
        "enroute_weather",
    }
    assert abs(sum(g["share"] for g in lr_groups.values()) - 1.0) < 1e-6

    rf_groups = {g["feature_group"]: g for g in payload["models"]["random_forest"]["group_totals"]}
    assert abs(sum(g["total"] for g in rf_groups.values()) - 1.0) < 1e-6

    lr_csv = results_dir / "feature_importance_logistic_regression.csv"
    rf_csv = results_dir / "feature_importance_random_forest.csv"
    groups_csv = results_dir / "feature_importance_groups.csv"
    assert lr_csv.is_file()
    assert rf_csv.is_file()
    assert groups_csv.is_file()

    with lr_csv.open("r", encoding="utf-8", newline="") as handle:
        lr_rows = list(csv.DictReader(handle))
    assert any(row["feature_name"].startswith("Origin=") for row in lr_rows)

    ctx = PipelineContext(
        root=tmp_path,
        config={},
        settings={},
        run_id=run_id,
        spark=spark,
    )
    run_out = feature_importance.run(ctx)
    assert run_out == results_dir / "feature_importance.json"
    assert run_out.is_file()
    assert "feature_importance" in ctx.timings
