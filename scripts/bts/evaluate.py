"""Evaluate the frozen BTS baseline arrival-delay model on the held-out test dataset."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from scripts.results import records, update_results

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]

# Configure Hadoop environment for Windows PySpark runtime if needed.
if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    hadoop_dir = ROOT / "envs" / "hadoop"
    if (hadoop_dir / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        os.environ["PATH"] = (
            str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")
        )

# Ensure PySpark workers use the active Python interpreter.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from .build_features import get_spark_session
from .train import confusion_metrics

TARGET_COL = "ArrDel15"


def compute_subgroup_slice(
    scores, group_col: str, subgroup_type: str, limit: int = None
) -> list:
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
    records = []
    for r in rows:
        total = int(r["total_flights"])
        delayed = int(r["delayed_flights"])
        tp = int(r["tp"])
        fp = int(r["fp"])
        tn = int(r["tn"])
        fn = int(r["fn"])
        cm = confusion_metrics(tp, fp, tn, fn)
        records.append(
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
    return records


def compute_subgroups(scores) -> pd.DataFrame:
    """Compute slice-based error analysis across reporting airlines, top origin airports, and departure hours."""
    airline_records = compute_subgroup_slice(scores, "Reporting_Airline", "airline")
    origin_records = compute_subgroup_slice(scores, "Origin", "origin", limit=20)
    hour_records = compute_subgroup_slice(scores, "DepHour", "departure_hour")
    all_records = airline_records + origin_records + hour_records
    return pd.DataFrame(all_records)


def save_evaluation_results(summary, subgroup_df, results_dir):
    """Replace BTS test results while preserving other models and stages."""
    update_results({"bts": {"test": {**summary, "subgroups": records(subgroup_df)}}}, results_dir)


def evaluate(spark, gold_dir=None, models_dir=None, results_dir=None) -> dict:
    """Execute test evaluation for the frozen baseline pipeline model and decision threshold."""
    results_dir = Path(results_dir or ROOT / "results")
    results_path = results_dir / "results.json"
    current = json.loads(results_path.read_text()) if results_path.exists() else {}
    training = current.get("bts", {}).get("training", {})
    gold_dir = Path(gold_dir or training.get("gold_dir", ROOT / "data/gold/bts"))
    models_dir = Path(models_dir or training.get("models_dir", ROOT / "models/bts"))
    if training.get("models_dir") and models_dir.resolve() != Path(training["models_dir"]).resolve():
        raise ValueError("Test evaluation must use the model recorded in the current BTS training results.")

    model_path = models_dir / "baseline_pipeline"
    threshold_path = models_dir / "threshold.json"
    test_data_path = gold_dir / "test"

    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {model_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Decision threshold file not found at {threshold_path}"
        )
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data partition not found at {test_data_path}")

    # Load frozen model pipeline and threshold metadata.
    print("\n--- [Step 1/4] Loading Frozen Pipeline and Threshold ---", flush=True)
    pipeline_model = PipelineModel.load(str(model_path))
    threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = float(threshold_data["threshold"])
    print(f"  Loaded model from: {model_path}")
    print(f"  Selected decision threshold: {threshold:.2f}", flush=True)

    # Read and transform test data.
    print(f"\n--- [Step 2/4] Scoring Test Partition ({test_data_path}) ---", flush=True)
    test_df = spark.read.parquet(str(test_data_path))
    scores = (
        pipeline_model.transform(test_df)
        .withColumn("p_delayed", vector_to_array("probability")[1])
        .withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
        .withColumn("prediction_0_5", (F.col("p_delayed") >= 0.50).cast("int"))
    ).cache()

    try:
        # Compute global confusion counts and metrics in a single aggregation.
        print("\n--- [Step 3/4] Computing Global Evaluation Metrics ---", flush=True)
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

        evaluator = BinaryClassificationEvaluator(
            labelCol=TARGET_COL, rawPredictionCol="p_delayed"
        )
        pr_auc = float(evaluator.setMetricName("areaUnderPR").evaluate(scores))
        roc_auc = float(evaluator.setMetricName("areaUnderROC").evaluate(scores))

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

        # Compute reference baselines.
        always_ontime_metrics = {
            "accuracy": (total_records - positive_records) / total_records
            if total_records > 0
            else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {
                "tp": 0,
                "fp": 0,
                "tn": total_records - positive_records,
                "fn": positive_records,
            },
            "pr_auc": delayed_prevalence,
            "roc_auc": 0.5,
        }
        always_delayed_metrics = {
            "accuracy": positive_records / total_records if total_records > 0 else 0.0,
            "precision": delayed_prevalence,
            "recall": 1.0,
            "f1": (2 * delayed_prevalence) / (1.0 + delayed_prevalence)
            if (1.0 + delayed_prevalence) > 0
            else 0.0,
            "confusion_matrix": {
                "tp": positive_records,
                "fp": total_records - positive_records,
                "tn": 0,
                "fn": 0,
            },
            "pr_auc": delayed_prevalence,
            "roc_auc": 0.5,
        }

        # Subgroup slice analysis.
        print("\n--- [Step 4/4] Performing Subgroup Error Analysis ---", flush=True)
        subgroup_df = compute_subgroups(scores)

        # Assemble summary object.
        rel_data_path = (
            str(test_data_path.relative_to(ROOT)).replace("\\", "/")
            if test_data_path.is_relative_to(ROOT)
            else str(test_data_path)
        )
        rel_model_path = (
            str(model_path.relative_to(ROOT)).replace("\\", "/")
            if model_path.is_relative_to(ROOT)
            else str(model_path)
        )

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_split": "test",
            "data_path": rel_data_path,
            "total_records": total_records,
            "positive_records": positive_records,
            "delayed_prevalence": delayed_prevalence,
            "model_path": rel_model_path,
            "selected_threshold": threshold,
            "metrics_at_selected_threshold": selected_metrics,
            "metrics_at_default_0_5": default_metrics,
            "reference_baselines": {
                "always_ontime": always_ontime_metrics,
                "always_delayed": always_delayed_metrics,
            },
        }

        # Save artifacts.
        save_evaluation_results(summary, subgroup_df, results_dir)
        print(f"  Saved evaluation artifacts to: {results_dir}", flush=True)

        return summary
    finally:
        scores.unpersist()


def main():
    spark = get_spark_session("bts_evaluation")
    try:
        summary = evaluate(spark)
        selected = summary["metrics_at_selected_threshold"]
        threshold = summary["selected_threshold"]
        cm = selected["confusion_matrix"]
        print("-" * 80)
        print(f"BTS Baseline Test Evaluation Summary (Threshold: {threshold:.2f}):")
        print("-" * 80)
        print(
            f"  Test Records:                       {summary['total_records']:,} (Prevalence: {summary['delayed_prevalence']:.2%})"
        )
        print(f"  Test PR-AUC:                        {selected['pr_auc']:.4f}")
        print(f"  Test ROC-AUC:                       {selected['roc_auc']:.4f}")
        print(f"  Test Delayed-Class F1:              {selected['f1']:.4f}")
        print(f"  Test Precision:                     {selected['precision']:.4f}")
        print(f"  Test Recall:                        {selected['recall']:.4f}")
        print(f"  Test Accuracy:                      {selected['accuracy']:.4f}")
        ontime_acc = summary["reference_baselines"]["always_ontime"]["accuracy"]
        delayed_f1 = summary["reference_baselines"]["always_delayed"]["f1"]
        print(f"  Reference Baseline (Always-On-Time) Accuracy: {ontime_acc:.4f}")
        print(f"  Reference Baseline (Always-Delayed) F1:       {delayed_f1:.4f}")
        print(f"  Confusion Matrix (Threshold {threshold:.2f}):")
        print(f"    True Positives (TP):  {cm['tp']:,}")
        print(f"    False Positives (FP): {cm['fp']:,}")
        print(f"    True Negatives (TN):  {cm['tn']:,}")
        print(f"    False Negatives (FN): {cm['fn']:,}")
        print("-" * 80)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
