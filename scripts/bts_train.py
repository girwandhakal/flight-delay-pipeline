"""Train and select the BTS-only logistic regression baseline."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[1]

# Configure Hadoop environment for Windows PySpark runtime if needed.
if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    hadoop_dir = ROOT / "envs" / "hadoop"
    if (hadoop_dir / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop_dir)
        os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")

# Ensure PySpark workers use the active Python interpreter.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from bts_transform import get_spark_session

CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Month",
    "DayOfWeek",
    "DepHour",
    "ArrHour",
]
NUMERIC_FEATURES = ["CRSElapsedTime", "Distance"]
TARGET_COL = "ArrDel15"
REG_PARAMS = [0.01, 0.1, 1.0]
THRESHOLDS = [round(i * 0.05, 2) for i in range(1, 20)]


def build_preprocessor() -> Pipeline:
    indices = [name + "_idx" for name in CATEGORICAL_FEATURES]
    vectors = [name + "_vec" for name in CATEGORICAL_FEATURES]
    return Pipeline(stages=[
        StringIndexer(inputCols=CATEGORICAL_FEATURES, outputCols=indices, handleInvalid="keep"),
        OneHotEncoder(inputCols=indices, outputCols=vectors, handleInvalid="keep", dropLast=False),
        VectorAssembler(inputCols=vectors + NUMERIC_FEATURES, outputCol="features", handleInvalid="error"),
    ])


# Calculate delayed-class metrics from confusion counts, including zero denominators.
def confusion_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    return {
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_predictions(predictions) -> dict:
    label = F.col(TARGET_COL)
    predicted = F.col("prediction")

    # Aggregate all 4 confusion counts in a single query.
    counts = predictions.agg(
        F.count(F.when((label == 1) & (predicted == 1), 1)).alias("tp"),
        F.count(F.when((label == 0) & (predicted == 1), 1)).alias("fp"),
        F.count(F.when((label == 0) & (predicted == 0), 1)).alias("tn"),
        F.count(F.when((label == 1) & (predicted == 0), 1)).alias("fn"),
    ).first().asDict()

    metrics = confusion_metrics(**counts)
    evaluator = BinaryClassificationEvaluator(labelCol=TARGET_COL, rawPredictionCol="p_delayed")
    metrics["pr_auc"] = evaluator.setMetricName("areaUnderPR").evaluate(predictions)
    metrics["roc_auc"] = evaluator.setMetricName("areaUnderROC").evaluate(predictions)
    return metrics


# Use the same probability threshold rule during selection and later scoring.
def predict(model, data, threshold: float = 0.5):
    return (
        model.transform(data)
        .withColumn("p_delayed", vector_to_array("probability")[1])
        .withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
    )


def tune_threshold(predictions):
    label = F.col(TARGET_COL)
    probability = F.col("p_delayed")

    # Build expressions to count total, positives, and (TP, FP) for each threshold.
    exprs = [
        F.count("*").alias("total"),
        F.count(F.when(label == 1, 1)).alias("positive"),
    ]
    for idx, t in enumerate(THRESHOLDS):
        exprs.append(F.count(F.when((label == 1) & (probability >= t), 1)).alias(f"tp_{idx}"))
        exprs.append(F.count(F.when((label == 0) & (probability >= t), 1)).alias(f"fp_{idx}"))

    # Single aggregation supplies confusion counts for every threshold in one pass.
    counts = predictions.agg(*exprs).first()
    total = counts["total"]
    positive = counts["positive"]
    negative = total - positive

    rows = []
    for idx, t in enumerate(THRESHOLDS):
        tp = counts[f"tp_{idx}"]
        fp = counts[f"fp_{idx}"]
        fn = positive - tp
        tn = negative - fp
        m = confusion_metrics(tp, fp, tn, fn)
        rows.append({
            "threshold": t,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    search_df = pd.DataFrame(rows)
    # Select threshold maximizing F1; resolve ties by picking higher threshold.
    best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    return best["threshold"], search_df


def train(spark, gold_dir=None, models_dir=None, results_dir=None):
    gold_dir = Path(gold_dir or ROOT / "data/gold/bts")
    models_dir = Path(models_dir or ROOT / "models/bts")
    results_dir = Path(results_dir or ROOT / "results/training_bts")

    train_data = spark.read.parquet(str(gold_dir / "train"))
    val_data = spark.read.parquet(str(gold_dir / "val"))

    print("\n--- [Step 1/4] Fitting Preprocessor on Training Flights ---", flush=True)
    preprocessor = build_preprocessor().fit(train_data)
    train_features = preprocessor.transform(train_data).select("features", TARGET_COL).cache()
    val_features = preprocessor.transform(val_data).select("features", TARGET_COL).cache()
    selected_scores = None

    try:
        training_stats = train_features.agg(
            F.count("*").alias("rows"),
            F.avg(TARGET_COL).alias("prevalence"),
        ).first()
        val_count = val_features.count()
        prevalence = training_stats["prevalence"]
        print(f"  Training cohort:   {training_stats['rows']:,} flights (delay prevalence: {prevalence:.2%})", flush=True)
        print(f"  Validation cohort: {val_count:,} flights", flush=True)

        # Reference baseline evaluations.
        print("\n--- [Step 2/4] Evaluating Reference Baselines on Validation Data ---", flush=True)
        references = {}
        for name, probability in [("always_ontime", 0.0), ("train_prevalence", prevalence)]:
            scores = (
                val_features.select(TARGET_COL)
                .withColumn("p_delayed", F.lit(probability))
                .withColumn("prediction", F.lit(int(probability >= 0.5)))
            )
            references[name] = evaluate_predictions(scores)
        references["train_prevalence"]["training_prevalence"] = prevalence
        print(f"  Always On-Time Reference:      Accuracy = {references['always_ontime']['accuracy']:.4f}, F1 = {references['always_ontime']['f1']:.4f}", flush=True)
        print(f"  Training Prevalence Reference: Accuracy = {references['train_prevalence']['accuracy']:.4f}, F1 = {references['train_prevalence']['f1']:.4f}", flush=True)

        # Candidate model evaluations across regularization parameters.
        print("\n--- [Step 3/4] Training Logistic Regression Candidates ---", flush=True)
        candidates = []
        selected_model = None
        selected_default = None
        selected_reg = None

        for idx, reg_param in enumerate(REG_PARAMS, start=1):
            print(f"  Candidate {idx}/{len(REG_PARAMS)} (regParam={reg_param}):", flush=True)
            model = LogisticRegression(
                featuresCol="features",
                labelCol=TARGET_COL,
                family="binomial",
                standardization=True,
                elasticNetParam=0.0,
                regParam=reg_param,
                maxIter=100,
                tol=1e-6,
            ).fit(train_features)

            scores = predict(model, val_features).select(TARGET_COL, "p_delayed", "prediction").cache()
            try:
                metrics = evaluate_predictions(scores)
            finally:
                scores.unpersist()

            candidates.append({"reg_param": reg_param, **metrics})
            print(f"    Validation PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1 (at 0.5): {metrics['f1']:.4f}", flush=True)

            # Choose candidate with highest validation PR-AUC; tie-break with higher regularization.
            if selected_model is None or (metrics["pr_auc"], reg_param) > (selected_default["pr_auc"], selected_reg):
                selected_model, selected_default, selected_reg = model, metrics, reg_param

        print(f"  => Selected Best Model: regParam={selected_reg} (PR-AUC: {selected_default['pr_auc']:.4f})", flush=True)

        # Select decision threshold on best candidate.
        print("\n--- [Step 4/4] Tuning Classification Threshold (Maximizing F1) ---", flush=True)
        selected_scores = predict(selected_model, val_features).select(TARGET_COL, "p_delayed", "prediction").cache()
        threshold, search_df = tune_threshold(selected_scores)

        best_row = search_df.loc[search_df["threshold"] == threshold].iloc[0]
        selected_metrics = confusion_metrics(
            tp=int(best_row["tp"]),
            fp=int(best_row["fp"]),
            tn=int(best_row["tn"]),
            fn=int(best_row["fn"]),
        )
        selected_metrics["pr_auc"] = selected_default["pr_auc"]
        selected_metrics["roc_auc"] = selected_default["roc_auc"]

        print(f"  Default threshold (0.50): F1 = {selected_default['f1']:.4f}, Precision = {selected_default['precision']:.4f}, Recall = {selected_default['recall']:.4f}", flush=True)
        print(f"  Optimal threshold ({threshold:.2f}): F1 = {selected_metrics['f1']:.4f}, Precision = {selected_metrics['precision']:.4f}, Recall = {selected_metrics['recall']:.4f}", flush=True)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_rows": training_stats["rows"],
            "validation_rows": val_count,
            "selected_model": {
                "reg_param": selected_reg,
                "best_threshold": threshold,
                "validation_metrics_at_selected_threshold": selected_metrics,
                "validation_metrics_at_default_0_5": selected_default,
            },
            "candidate_comparison": candidates,
            "reference_baselines": references,
        }

        # Save artifacts: pipeline model, threshold metadata, summary JSON, and threshold search CSV.
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        pipeline = PipelineModel(stages=preprocessor.stages + [selected_model])
        pipeline.write().overwrite().save(str(models_dir / "baseline_pipeline"))

        threshold_info = {
            "threshold": threshold,
            "positive_class": 1,
            "decision_rule": "p_delayed >= threshold",
        }
        (models_dir / "threshold.json").write_text(json.dumps(threshold_info, indent=2) + "\n", encoding="utf-8")
        search_df.to_csv(results_dir / "threshold_search.csv", index=False)
        (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        return summary
    finally:
        if selected_scores is not None:
            selected_scores.unpersist()
        val_features.unpersist()
        train_features.unpersist()


def main():
    spark = get_spark_session("bts_training")
    try:
        summary = train(spark)
        selected = summary["selected_model"]
        val_metrics = selected["validation_metrics_at_selected_threshold"]
        cm = val_metrics["confusion_matrix"]
        print("-" * 80)
        print("Selected BTS Baseline Model Summary:")
        print("-" * 80)
        print(f"  Selected Regularization (regParam): {selected['reg_param']}")
        print(f"  Optimized Decision Threshold:       {selected['best_threshold']:.2f}")
        print(f"  Validation PR-AUC:                  {val_metrics['pr_auc']:.4f}")
        print(f"  Validation ROC-AUC:                 {val_metrics['roc_auc']:.4f}")
        print(f"  Validation Delayed-Class F1:        {val_metrics['f1']:.4f}")
        print(f"  Validation Precision:               {val_metrics['precision']:.4f}")
        print(f"  Validation Recall:                  {val_metrics['recall']:.4f}")
        print(f"  Validation Accuracy:                {val_metrics['accuracy']:.4f}")
        print(f"  Confusion Matrix (Threshold {selected['best_threshold']:.2f}):")
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
