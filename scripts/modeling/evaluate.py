"""Score a completed flight-delay model run on frozen test data."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame, SparkSession, functions as F

from scripts.common.config import ROOT, project_path
from scripts.common.spark import get_spark_session
from scripts.common.results import records, update_results

from .run import GOLD_DIR, RESULT_KEY, RUNS_DIR, format_classification_report
from .train import TARGET_COL, confusion_metrics, predict, validate_weather


def compute_subgroup_slice(
    scores: DataFrame,
    group_col: str,
    subgroup_type: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Aggregate slice-based metrics inside Spark to avoid collecting full dataset into driver memory."""
    label = F.col(TARGET_COL)
    pred = F.col("prediction")

    agg_df = scores.groupBy(group_col).agg(
        F.count("*").alias("total_flights"),
        F.count(F.when(label == 1, 1)).alias("delayed_flights"),
        F.count(F.when((label == 1) & (pred == 1), 1)).alias("tp"),
        F.count(F.when((label == 0) & (pred == 1), 1)).alias("fp"),
        F.count(F.when((label == 0) & (pred == 0), 1)).alias("tn"),
        F.count(F.when((label == 1) & (pred == 0), 1)).alias("fn"),
    )

    if limit is not None:
        agg_df = agg_df.orderBy(F.col("total_flights").desc()).limit(limit)
    else:
        agg_df = agg_df.orderBy(F.col(group_col).asc())

    rows = agg_df.collect()
    subgroup_records = []
    for r in rows:
        total = int(r["total_flights"])
        delayed = int(r["delayed_flights"])
        tp = int(r["tp"])
        fp = int(r["fp"])
        tn = int(r["tn"])
        fn = int(r["fn"])
        cm = confusion_metrics(tp, fp, tn, fn)
        subgroup_records.append(
            {
                "subgroup_type": subgroup_type,
                "subgroup_value": str(r[group_col]),
                "total_flights": total,
                "delayed_flights": delayed,
                "prevalence": delayed / total if total > 0 else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": cm["precision"],
                "recall": cm["recall"],
                "f1": cm["f1"],
                "accuracy": cm["accuracy"],
            }
        )
    return subgroup_records


def compute_subgroups(scores: DataFrame) -> pd.DataFrame:
    """Compute slice-based error analysis across reporting airlines, top origin airports, and departure hours."""
    airline_records = compute_subgroup_slice(scores, "Reporting_Airline", "airline")
    origin_records = compute_subgroup_slice(scores, "Origin", "origin", limit=20)
    hour_records = compute_subgroup_slice(scores, "DepHour", "departure_hour")
    all_records = airline_records + origin_records + hour_records
    return pd.DataFrame(all_records)


def score_variant(scores: DataFrame, output: Path, threshold: float) -> dict[str, Any]:
    """Write predictions and summarize the two operating points on the same rows."""
    scores = scores.withColumn("prediction_0_5", (F.col("p_delayed") >= 0.5).cast("int")).cache()
    try:
        label = F.col(TARGET_COL)
        pred = F.col("prediction")
        pred_0_5 = F.col("prediction_0_5")
        counts = (
            scores.agg(
                F.count("*").alias("total"),
                F.count(F.when(label == 1, 1)).alias("positive"),
                F.count(F.when((label == 1) & (pred == 1), 1)).alias("tp"),
                F.count(F.when((label == 0) & (pred == 1), 1)).alias("fp"),
                F.count(F.when((label == 0) & (pred == 0), 1)).alias("tn"),
                F.count(F.when((label == 1) & (pred == 0), 1)).alias("fn"),
                F.count(F.when((label == 1) & (pred_0_5 == 1), 1)).alias("tp_0_5"),
                F.count(F.when((label == 0) & (pred_0_5 == 1), 1)).alias("fp_0_5"),
                F.count(F.when((label == 0) & (pred_0_5 == 0), 1)).alias("tn_0_5"),
                F.count(F.when((label == 1) & (pred_0_5 == 0), 1)).alias("fn_0_5"),
            )
            .first()
            .asDict()
        )

        total_records = int(counts["total"])
        positive_records = int(counts["positive"])
        delayed_prevalence = (
            positive_records / total_records if total_records > 0 else 0.0
        )

        pr_auc = roc_auc = None
        if 0 < positive_records < total_records:
            evaluator = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol="p_delayed")
            pr_auc = evaluator.setMetricName("areaUnderPR").evaluate(scores)
            roc_auc = evaluator.setMetricName("areaUnderROC").evaluate(scores)

        selected_metrics = confusion_metrics(
            counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        )
        selected_metrics["pr_auc"] = pr_auc
        selected_metrics["roc_auc"] = roc_auc

        default_metrics = confusion_metrics(
            counts["tp_0_5"], counts["fp_0_5"], counts["tn_0_5"], counts["fn_0_5"]
        )
        default_metrics["pr_auc"] = pr_auc
        default_metrics["roc_auc"] = roc_auc

        always_ontime_metrics = {
            **confusion_metrics(
                tp=0,
                fp=0,
                tn=total_records - positive_records,
                fn=positive_records,
            ),
            "pr_auc": delayed_prevalence,
            "roc_auc": 0.5,
        }
        always_delayed_metrics = {
            **confusion_metrics(
                tp=positive_records,
                fp=total_records - positive_records,
                tn=0,
                fn=0,
            ),
            "pr_auc": delayed_prevalence,
            "roc_auc": 0.5,
        }

        (scores.select("flight_id", "FlightDate", TARGET_COL, "p_delayed", "prediction")
         .withColumn("threshold", F.lit(threshold)).write.parquet(str(output)))
        return {
            "total_records": total_records, "positive_records": positive_records,
            "delayed_prevalence": delayed_prevalence, "selected_threshold": threshold,
            "metrics_at_selected_threshold": selected_metrics,
            "metrics_at_default_0_5": default_metrics,
            "reference_baselines": {"always_ontime": always_ontime_metrics, "always_delayed": always_delayed_metrics},
            "subgroups": records(compute_subgroups(scores)),
        }
    finally:
        scores.unpersist()


def load_frozen_run(root: Path, run_id: str) -> tuple[Path, Path, dict]:
    """Validate the current frozen run directory, manifest, and test partition before scoring."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Use the saved directory name as RUN_ID.")
    run_dir = project_path(root, RUNS_DIR / run_id)
    if run_dir.parent != project_path(root, RUNS_DIR) or not (run_dir / "_SUCCESS").is_file():
        raise ValueError("Test scoring requires a completed flight-delay run.")
    current = json.loads((root / "results/results.json").read_text(encoding="utf-8")).get(RESULT_KEY, {})
    if current.get("run_id") != run_id:
        raise ValueError("Test scoring requires the current flight-delay run.")

    manifest_path = project_path(run_dir, "run.json")
    if not manifest_path.is_file():
        raise ValueError("Frozen run manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id:
        raise ValueError("Frozen run identity changed.")

    gold = project_path(root, manifest["gold"])
    if gold.parent != project_path(root, GOLD_DIR) or not (gold / "test" / "_SUCCESS").is_file():
        raise ValueError("Frozen test Gold is missing or outside the en-route Gold directory.")

    variants = manifest["variants"]
    if not variants:
        raise ValueError("The frozen run contains no fitted model.")
    for variant in variants.values():
        model_dir = project_path(run_dir, variant["model"])
        threshold_file = project_path(run_dir, variant["threshold_file"])
        if not model_dir.exists() or not threshold_file.is_file():
            raise ValueError("Frozen model or threshold file is missing.")
        threshold = json.loads(threshold_file.read_text(encoding="utf-8"))
        if not 0 < variant["threshold"] < 1 or threshold["threshold"] != variant["threshold"]:
            raise ValueError("Frozen threshold differs from the manifest.")
    return run_dir, gold, variants


def evaluate(spark: SparkSession, root: Path, run_id: str) -> dict[str, Any]:
    """Score all saved variants and publish the result together."""
    run_dir, gold, variants = load_frozen_run(root, run_id)
    attempt_id = uuid4().hex[:16]
    output = project_path(run_dir, Path("test") / attempt_id)
    output.mkdir(parents=True, exist_ok=False)
    data = spark.read.parquet(str(gold / "test"))
    validate_weather(data, "frozen test inputs", require_values=False)

    reports = {}
    for name, variant in variants.items():
        model = PipelineModel.load(str(project_path(run_dir, variant["model"])))
        predictions = project_path(output, Path(name) / "predictions")
        threshold = variant["threshold"]
        reports[name] = {
            **score_variant(predict(model, data, threshold), predictions, threshold),
            "predictions": predictions.relative_to(root.resolve()).as_posix(),
        }
    summary = {
        "run_id": run_id, "attempt_id": attempt_id, "evaluation_split": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(), "models": reports,
    }
    update_results({RESULT_KEY: {"test": summary}}, root / "results")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", metavar="RUN_ID")
    args = parser.parse_args()
    spark = get_spark_session("flight_delay_test")
    try:
        summary = evaluate(spark, ROOT, args.run_id)
    finally:
        spark.stop()
    print(format_classification_report(args.run_id, summary["models"], split="TEST"), flush=True)
    print(f"Test scores saved for {args.run_id} (attempt {summary['attempt_id']}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
