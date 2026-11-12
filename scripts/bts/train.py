"""Train BTS-only or BTS plus origin weather with the same chronological search."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from scripts.results import records, update_results

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder
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

from .build_features import get_spark_session, load_split_config
from scripts.origin_weather.build_features import WEATHER_NUMERIC, WEATHER_FLAGS

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
REG_PARAMS = [0.001, 0.01, 0.1, 1.0]
ELASTIC_NET_PARAMS = [0.0, 0.5, 1.0]
THRESHOLDS = [round(i * 0.05, 2) for i in range(1, 20)]


def build_preprocessor(feature_set="bts") -> Pipeline:
    if feature_set not in ("bts", "bts_origin_weather"):
        raise ValueError(f"Unknown feature set: {feature_set}")
    indices = [name + "_idx" for name in CATEGORICAL_FEATURES]
    vectors = [name + "_vec" for name in CATEGORICAL_FEATURES]
    stages = [
        StringIndexer(
            inputCols=CATEGORICAL_FEATURES, outputCols=indices, handleInvalid="keep"
        ),
        OneHotEncoder(
            inputCols=indices, outputCols=vectors, handleInvalid="keep", dropLast=False
        ),
    ]
    numeric = NUMERIC_FEATURES.copy()
    if feature_set == "bts_origin_weather":
        imputed = [name + "_imputed" for name in WEATHER_NUMERIC]
        stages.append(
            Imputer(inputCols=WEATHER_NUMERIC, outputCols=imputed, strategy="mean")
        )
        numeric += imputed + WEATHER_FLAGS
    stages.append(
        VectorAssembler(
            inputCols=vectors + numeric, outputCol="features", handleInvalid="error"
        )
    )
    return Pipeline(stages=stages)


def validate_weather(data, context):
    counts = data.agg(
        *[
            F.count(F.when(F.col(name).isNotNull() & ~F.isnan(name), 1)).alias(name)
            for name in WEATHER_NUMERIC
        ]
    ).first()
    for name in WEATHER_NUMERIC:
        if not counts[name]:
            raise ValueError(f"No valid training values for {name} in {context}")
    invalid = F.lit(False)
    for name in WEATHER_NUMERIC:
        invalid = invalid | (F.abs(F.col(name)) == float("inf"))
    for name in WEATHER_FLAGS:
        invalid = invalid | F.col(name).isNull() | ~F.col(name).isin(0, 1)
    if data.filter(invalid).limit(1).count():
        raise ValueError(f"Invalid weather values or flags in {context}")


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


def evaluate_predictions(
    predictions, label_col=TARGET_COL, pred_col="prediction", prob_col="p_delayed"
) -> dict:
    label = F.col(label_col)
    predicted = F.col(pred_col)

    # Aggregate all 4 confusion counts in a single query.
    counts = (
        predictions.agg(
            F.countDistinct(label).alias("classes"),
            F.count(F.when((label == 1) & (predicted == 1), 1)).alias("tp"),
            F.count(F.when((label == 0) & (predicted == 1), 1)).alias("fp"),
            F.count(F.when((label == 0) & (predicted == 0), 1)).alias("tn"),
            F.count(F.when((label == 1) & (predicted == 0), 1)).alias("fn"),
        )
        .first()
        .asDict()
    )

    classes = counts.pop("classes")
    metrics = confusion_metrics(**counts)
    if classes < 2:
        return {
            **metrics,
            "pr_auc": None,
            "roc_auc": None,
            "ranking_reason": "Fewer than two target classes",
        }
    evaluator = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol=prob_col
    )
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
        exprs.append(
            F.count(F.when((label == 1) & (probability >= t), 1)).alias(f"tp_{idx}")
        )
        exprs.append(
            F.count(F.when((label == 0) & (probability >= t), 1)).alias(f"fp_{idx}")
        )

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
        rows.append(
            {
                "threshold": t,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    search_df = pd.DataFrame(rows)
    # Select threshold maximizing F1; resolve ties by picking higher threshold.
    best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    return best["threshold"], search_df


def train(
    spark,
    gold_dir=None,
    models_dir=None,
    results_dir=None,
    feature_set="bts",
    save_predictions=False,
    publish_results=True,
):
    if feature_set not in ("bts", "bts_origin_weather"):
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set == "bts_origin_weather" and any(
        path is None for path in (gold_dir, models_dir)
    ):
        raise ValueError(
            "Origin-weather training requires enriched Gold and isolated model paths."
        )
    gold_dir = Path(gold_dir or ROOT / "data/gold/bts")
    models_dir = Path(models_dir or ROOT / "models/bts")
    results_dir = Path(results_dir or ROOT / "results")

    train_data = spark.read.parquet(str(gold_dir / "train"))
    val_data = spark.read.parquet(str(gold_dir / "val"))

    print("\n--- [Step 1/4] Preparing Training and Validation Flights ---", flush=True)
    columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COL]
    if feature_set == "bts_origin_weather":
        columns += WEATHER_NUMERIC + WEATHER_FLAGS
    metadata = ["flight_id", "FlightDate"] if save_predictions else []
    # Leave JVM heap for concurrent classifier tasks rather than duplicated training rows.
    train_features = train_data.select("FlightDate", *columns).persist(StorageLevel.DISK_ONLY)
    val_features = val_data.select(*metadata, *columns).persist(StorageLevel.DISK_ONLY)
    selected_scores = None

    try:
        training_stats = train_features.agg(
            F.count("*").alias("rows"),
            F.avg(TARGET_COL).alias("prevalence"),
        ).first()
        val_count = val_features.count()
        prevalence = training_stats["prevalence"]
        print(
            f"  Training cohort:   {training_stats['rows']:,} flights (delay prevalence: {prevalence:.2%})",
            flush=True,
        )
        print(f"  Validation cohort: {val_count:,} flights", flush=True)

        # Reference baseline evaluations.
        print(
            "\n--- [Step 2/4] Evaluating Reference Baselines on Validation Data ---",
            flush=True,
        )
        references = {}
        for name, probability in [
            ("always_ontime", 0.0),
            ("always_delayed", 1.0),
            ("train_prevalence", prevalence),
        ]:
            scores = (
                val_features.select(TARGET_COL)
                .withColumn("p_delayed", F.lit(probability))
                .withColumn("prediction", F.lit(int(probability >= 0.5)))
            )
            references[name] = evaluate_predictions(scores)
        references["train_prevalence"]["training_prevalence"] = prevalence
        print(
            f"  Always On-Time Reference:      Accuracy = {references['always_ontime']['accuracy']:.4f}, F1 = {references['always_ontime']['f1']:.4f}",
            flush=True,
        )
        print(
            f"  Always Delayed Reference:      Accuracy = {references['always_delayed']['accuracy']:.4f}, F1 = {references['always_delayed']['f1']:.4f}",
            flush=True,
        )
        print(
            f"  Training Prevalence Reference: Accuracy = {references['train_prevalence']['accuracy']:.4f}, F1 = {references['train_prevalence']['f1']:.4f}",
            flush=True,
        )

        print("\n--- [Step 3/4] Automatic Hyperparameter Search ---", flush=True)
        estimator = LogisticRegression(
            featuresCol="features",
            labelCol=TARGET_COL,
            family="binomial",
            standardization=True,
            maxIter=100,
            tol=1e-6,
        )
        grid = (
            ParamGridBuilder()
            .addGrid(estimator.regParam, REG_PARAMS)
            .addGrid(estimator.elasticNetParam, ELASTIC_NET_PARAMS)
            .build()
        )
        evaluator = BinaryClassificationEvaluator(
            labelCol=TARGET_COL,
            metricName="areaUnderPR",
        )
        months = [
            row.month
            for row in train_features.select(
                F.trunc("FlightDate", "month").alias("month")
            )
            .distinct()
            .orderBy("month")
            .collect()
        ]
        if None in months or len(months) < 4:
            raise ValueError(
                "Chronological cross-validation requires valid FlightDate values and at least four training months"
            )
        candidates = [
            {
                "reg_param": params[estimator.regParam],
                "elastic_net_param": params[estimator.elasticNetParam],
                "fold_pr_auc": [],
            }
            for params in grid
        ]
        folds = []
        print(
            f"  Searching {len(grid)} combinations across 3 chronological folds...",
            flush=True,
        )
        for month in months[-3:]:
            fold_train = train_features.filter(F.col("FlightDate") < F.lit(month))
            fold_val = train_features.filter(
                F.trunc("FlightDate", "month") == F.lit(month)
            )
            # Fit once per fold: preprocessing does not depend on classifier parameters.
            if feature_set == "bts_origin_weather":
                validate_weather(fold_train, f"fold before {month}")
            preprocessor = build_preprocessor(feature_set).fit(fold_train)
            encoded_train = (
                preprocessor.transform(fold_train)
                .select("features", TARGET_COL)
                .persist(StorageLevel.DISK_ONLY)
            )
            encoded_val = (
                preprocessor.transform(fold_val)
                .select("features", TARGET_COL)
                .persist(StorageLevel.DISK_ONLY)
            )
            try:
                for params, candidate in zip(grid, candidates):
                    model = estimator.fit(encoded_train, params)
                    candidate["fold_pr_auc"].append(
                        evaluator.evaluate(model.transform(encoded_val))
                    )
            finally:
                encoded_train.unpersist(blocking=True)
                encoded_val.unpersist(blocking=True)
            folds.append(
                {
                    "train_before": month.isoformat(),
                    "validation_month": month.strftime("%Y-%m"),
                }
            )
            print(
                f"  Finished fold: train before {month}, validate {month:%Y-%m}",
                flush=True,
            )
        for candidate in candidates:
            candidate["internal_pr_auc"] = sum(candidate["fold_pr_auc"]) / len(folds)
        best = max(range(len(grid)), key=lambda i: candidates[i]["internal_pr_auc"])
        if feature_set == "bts_origin_weather":
            validate_weather(train_features, "full training")
        selected_model = Pipeline(
            stages=build_preprocessor(feature_set).getStages() + [estimator]
        ).fit(train_features, grid[best])
        classifier = selected_model.stages[-1]
        selected_reg = classifier.getRegParam()
        selected_elastic_net = classifier.getElasticNetParam()
        print(
            f"  Selected regParam={selected_reg}, elasticNetParam={selected_elastic_net}",
            flush=True,
        )

        print("\n--- [Step 4/4] Validation and Threshold Selection ---", flush=True)
        selected_scores = (
            predict(selected_model, val_features)
            .select(*metadata, TARGET_COL, "p_delayed", "prediction")
            .cache()
        )
        selected_default = evaluate_predictions(selected_scores)
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

        print(
            f"  Default threshold (0.50): F1 = {selected_default['f1']:.4f}, Precision = {selected_default['precision']:.4f}, Recall = {selected_default['recall']:.4f}",
            flush=True,
        )
        print(
            f"  Optimal threshold ({threshold:.2f}): F1 = {selected_metrics['f1']:.4f}, Precision = {selected_metrics['precision']:.4f}, Recall = {selected_metrics['recall']:.4f}",
            flush=True,
        )

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_set": feature_set,
            "models_dir": str(models_dir),
            "splits": load_split_config(),
            "threshold_search": records(search_df),
            "input_columns": columns,
            "gold_dir": str(gold_dir),
            "vector_dimension": selected_model.transform(val_features.limit(1))
            .first()["features"]
            .size,
            "imputation_means": next(
                (
                    stage.surrogateDF.first().asDict()
                    for stage in selected_model.stages
                    if hasattr(stage, "surrogateDF")
                ),
                {},
            ),
            "training_rows": training_stats["rows"],
            "validation_rows": val_count,
            "selected_model": {
                "reg_param": selected_reg,
                "elastic_net_param": selected_elastic_net,
                "best_threshold": threshold,
                "validation_metrics_at_selected_threshold": selected_metrics,
                "validation_metrics_at_default_0_5": selected_default,
            },
            "tuning": {
                "method": "expanding-window chronological cross-validation",
                "metric": "areaUnderPR",
                "folds": folds,
                "combinations": len(grid),
                "aggregation": "mean PR-AUC across folds",
                "scope": "last three training months scored separately using only earlier flights; winner refitted on all training rows",
            },
            "candidate_comparison": candidates,
            "reference_baselines": references,
        }

        # Predictions and frozen inference settings belong beside the fitted model.
        models_dir.mkdir(parents=True, exist_ok=True)

        selected_model.write().overwrite().save(str(models_dir / "baseline_pipeline"))

        threshold_info = {
            "threshold": threshold,
            "positive_class": 1,
            "decision_rule": "p_delayed >= threshold",
        }
        (models_dir / "threshold.json").write_text(
            json.dumps(threshold_info, indent=2) + "\n", encoding="utf-8"
        )

        if save_predictions:
            (
                selected_scores.withColumn(
                    "prediction", (F.col("p_delayed") >= threshold).cast("int")
                )
                .withColumn("threshold", F.lit(threshold))
                .write.parquet(str(models_dir / "validation_predictions"))
            )

        if publish_results:
            update_results({feature_set: {
                "training": summary,
                "validation": {
                    "selected": selected_metrics,
                    "default_0_5": selected_default,
                    "threshold": threshold,
                },
            }}, results_dir)
        return summary
    finally:
        if selected_scores is not None:
            selected_scores.unpersist()
        val_features.unpersist(blocking=True)
        train_features.unpersist(blocking=True)


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
        print(f"  Selected Elastic Net:              {selected['elastic_net_param']}")
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
