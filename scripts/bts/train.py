"""Train BTS and weather variants once per algorithm with fixed hyperparameters."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from scripts.results import records, update_results

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]

from .build_features import get_spark_session, load_split_config
from scripts.origin_weather.build_features import WEATHER_NUMERIC, WEATHER_FLAGS
from scripts.destination_weather.build_features import DESTINATION_WEATHER_NUMERIC, DESTINATION_WEATHER_FLAGS
from .recent_delays import FEATURES as RECENT_CALENDAR_FEATURES, ROLLING_FEATURES, HOLIDAY_FEATURES

FEATURE_SETS = ("bts", "bts_origin_weather", "bts_origin_destination_weather")


def weather_columns(feature_set):
    numeric, flags = WEATHER_NUMERIC.copy(), WEATHER_FLAGS.copy()
    if feature_set == "bts_origin_destination_weather":
        numeric += DESTINATION_WEATHER_NUMERIC
        flags += DESTINATION_WEATHER_FLAGS
    return numeric, flags


CATEGORICAL_FEATURES = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "DayOfWeek",
    "DepHour",
    "ArrHour",
]
NUMERIC_FEATURES = ["CRSElapsedTime", "Distance"]
INPUT_FEATURES = CATEGORICAL_FEATURES[:3] + ["Month"] + CATEGORICAL_FEATURES[3:] + NUMERIC_FEATURES + RECENT_CALENDAR_FEATURES
TARGET_COL = "ArrDel15"
LOGISTIC_PARAMS = {"regParam": 0.01, "elasticNetParam": 0.0, "maxIter": 100, "tol": 1e-6}
GBT_PARAMS = {"maxDepth": 3, "maxIter": 50, "stepSize": 0.05, "seed": 42}
THRESHOLDS = [round(i * 0.05, 2) for i in range(1, 20)]


def build_preprocessor(feature_set="bts") -> Pipeline:
    if feature_set not in FEATURE_SETS:
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
    numeric = NUMERIC_FEATURES + ["MonthSin", "MonthCos"] + [name + "_filled" for name in ROLLING_FEATURES] + HOLIDAY_FEATURES
    if feature_set != "bts":
        weather_numeric, _ = weather_columns(feature_set)
        imputed = [name + "_imputed" for name in weather_numeric]
        stages.append(Imputer(inputCols=weather_numeric, outputCols=imputed, strategy="mean"))
        # Preserve exact vector ordering: origin numeric + flags, then destination numeric + flags.
        numeric += [name + "_imputed" for name in WEATHER_NUMERIC] + WEATHER_FLAGS
        if feature_set == "bts_origin_destination_weather":
            numeric += [name + "_imputed" for name in DESTINATION_WEATHER_NUMERIC] + DESTINATION_WEATHER_FLAGS
    # Calendar values must remain defined in months absent from the training vocabulary.
    stages.append(SQLTransformer(statement=(
        "SELECT *, SIN(2 * PI() * (Month - 1) / 12) AS MonthSin, "
        "COS(2 * PI() * (Month - 1) / 12) AS MonthCos, "
        + ", ".join(f"COALESCE({name}, 0D) AS {name}_filled" for name in ROLLING_FEATURES)
        + " FROM __THIS__"
    )))
    stages.append(
        VectorAssembler(
            inputCols=vectors + numeric, outputCol="features", handleInvalid="error"
        )
    )
    return Pipeline(stages=stages)


def validate_weather(data, context, feature_set="bts_origin_weather", require_values=True):
    numeric, flags = weather_columns(feature_set)
    counts = data.agg(
        *[
            F.count(F.when(F.col(name).isNotNull() & ~F.isnan(name), 1)).alias(name)
            for name in numeric
        ]
    ).first()
    for name in numeric:
        if require_values and not counts[name]:
            raise ValueError(f"No valid training values for {name} in {context}")
    invalid = F.lit(False)
    for name in numeric:
        invalid = invalid | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    for name in flags:
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
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    if feature_set != "bts" and any(
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

    print("\n--- [Step 1/3] Preparing Training and Validation Flights ---", flush=True)
    columns = INPUT_FEATURES + [TARGET_COL]
    if feature_set != "bts":
        numeric, flags = weather_columns(feature_set)
        columns += numeric + flags
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
        if feature_set != "bts":
            validate_weather(val_features, "validation", feature_set, require_values=False)
        val_count = val_features.count()
        prevalence = training_stats["prevalence"]
        print(
            f"  Training cohort:   {training_stats['rows']:,} flights (delay prevalence: {prevalence:.2%})",
            flush=True,
        )
        print(f"  Validation cohort: {val_count:,} flights", flush=True)

        # Reference baseline evaluations.
        print(
            "\n--- [Step 2/3] Evaluating Reference Baselines on Validation Data ---",
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

        if feature_set != "bts":
            validate_weather(train_features, "full training", feature_set)
        print("\n--- [Step 3/3] Fit Fixed Models and Select Validation Thresholds ---", flush=True)
        estimators = [
            ("logistic_regression", LogisticRegression(
                featuresCol="features", labelCol=TARGET_COL, family="binomial",
                standardization=True, **LOGISTIC_PARAMS)),
            ("gradient_boosted_trees", GBTClassifier(
                featuresCol="features", labelCol=TARGET_COL, **GBT_PARAMS)),
        ]
        algorithms = {}
        for name, estimator in estimators:
            print(f"  Fitting {name} on all training flights with fixed settings: "
                  f"{LOGISTIC_PARAMS if name == 'logistic_regression' else GBT_PARAMS}", flush=True)
            selected_model = Pipeline(
                stages=build_preprocessor(feature_set).getStages() + [estimator]
            ).fit(train_features)
            selected_scores = (
                predict(selected_model, val_features)
                .select(*metadata, TARGET_COL, "p_delayed", "prediction")
                .cache()
            )
            selected_default = evaluate_predictions(selected_scores)
            threshold, search_df = tune_threshold(selected_scores)
            best_row = search_df.loc[search_df["threshold"] == threshold].iloc[0]
            selected_metrics = confusion_metrics(
                **{key: int(best_row[key]) for key in ("tp", "fp", "tn", "fn")}
            )
            selected_metrics.update({key: selected_default[key] for key in ("pr_auc", "roc_auc")})
            # Keep a common logistic reference across weather variants; trees are a separate challenger.
            directory = models_dir if name == "logistic_regression" else models_dir / "gbt"
            directory.mkdir(parents=True, exist_ok=True)
            selected_model.write().overwrite().save(str(directory / "baseline_pipeline"))
            (directory / "threshold.json").write_text(json.dumps({
                "threshold": threshold, "positive_class": 1,
                "decision_rule": "p_delayed >= threshold",
            }, indent=2) + "\n", encoding="utf-8")
            if save_predictions:
                (selected_scores.withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
                 .withColumn("threshold", F.lit(threshold))
                 .write.parquet(str(directory / "validation_predictions")))
            algorithms[name] = {
                "algorithm": name,
                **({"reg_param": estimator.getRegParam(),
                    "elastic_net_param": estimator.getElasticNetParam(),
                    "max_iter": estimator.getMaxIter(), "tol": estimator.getTol()}
                   if name == "logistic_regression" else
                   {"max_depth": estimator.getMaxDepth(), "max_iter": estimator.getMaxIter(),
                    "step_size": estimator.getStepSize(), "seed": estimator.getSeed()}),
                "models_dir": str(directory),
                "best_threshold": threshold,
                "threshold_search": records(search_df),
                "validation_metrics_at_selected_threshold": selected_metrics,
                "validation_metrics_at_default_0_5": selected_default,
            }
            if name == "logistic_regression":
                vector_dimension = selected_model.transform(val_features.limit(1)).first()["features"].size
                imputation_means = next((stage.surrogateDF.first().asDict()
                                         for stage in selected_model.stages
                                         if hasattr(stage, "surrogateDF")), {})
            print(f"  {name}: validation PR-AUC={selected_default['pr_auc']}, "
                  f"F1={selected_metrics['f1']:.4f}, threshold={threshold:.2f}", flush=True)
            selected_scores.unpersist(blocking=True)
            selected_scores = None

        baseline = algorithms["logistic_regression"]

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_set": feature_set,
            "models_dir": str(models_dir),
            "splits": load_split_config(),
            "threshold_search": baseline["threshold_search"],
            "input_columns": columns,
            "gold_dir": str(gold_dir),
            "vector_dimension": vector_dimension,
            "imputation_means": imputation_means,
            "seasonal_encoding": "MonthSin=sin(2*pi*(Month-1)/12), MonthCos=cos(2*pi*(Month-1)/12)",
            "training_rows": training_stats["rows"],
            "validation_rows": val_count,
            "selected_model": {
                key: baseline[key] for key in (
                    "algorithm", "reg_param", "elastic_net_param", "best_threshold",
                    "validation_metrics_at_selected_threshold", "validation_metrics_at_default_0_5",
                )
            },
            "algorithm_comparison": algorithms,
            "tuning": {
                "method": "fixed hyperparameters",
                "search_performed": False,
                "scope": "Each algorithm fitted once on all training rows; F1 thresholds selected on validation only; logistic regression remains the reference",
            },
            "reference_baselines": references,
        }

        if publish_results:
            update_results({feature_set: {
                "training": summary,
                "validation": {
                    "selected": baseline["validation_metrics_at_selected_threshold"],
                    "default_0_5": baseline["validation_metrics_at_default_0_5"],
                    "threshold": baseline["best_threshold"],
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
        print(f"  Fixed Regularization (regParam): {selected['reg_param']}")
        print(f"  Fixed Elastic Net:              {selected['elastic_net_param']}")
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
