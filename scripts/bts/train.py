"""Train and validate BTS and weather models with fixed hyperparameters."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional, Union

from scripts.bts.cache import active
from scripts.progress import print_progress, stage
from scripts.results import records, update_results
from scripts.aircraft_rotation.build_features import ROTATION_FLAGS, ROTATION_NUMERIC
from scripts.enroute_weather.build_features import ENROUTE_FLAGS, ENROUTE_NUMERIC, FEATURES as ENROUTE_FEATURES

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[2]

from .build_features import get_spark_session, load_split_config
from scripts.origin_weather.build_features import WEATHER_NUMERIC, WEATHER_FLAGS
from scripts.destination_weather.build_features import DESTINATION_WEATHER_NUMERIC, DESTINATION_WEATHER_FLAGS
from .recent_delays import (
    FEATURES as RECENT_CALENDAR_FEATURES, ROLLING_FEATURES, HOLIDAY_FEATURES, UPSTREAM_FEATURES, UPSTREAM_ROLES,
)

FEATURE_SETS = ("bts", "bts_origin_weather", "bts_origin_destination_weather")
ROTATION_FEATURE_SET = "bts_origin_destination_weather_rotation"
ALL_FEATURE_SETS = (*FEATURE_SETS, ROTATION_FEATURE_SET)
ENROUTE_FEATURE_SET = "bts_origin_destination_weather_rotation_enroute"
ROTATION_FEATURE_SETS = (ROTATION_FEATURE_SET, ENROUTE_FEATURE_SET)
TRAINING_FEATURE_SETS = (*FEATURE_SETS, *ROTATION_FEATURE_SETS)


def weather_columns(feature_set: str) -> tuple[list[str], list[str]]:
    """Return numeric and binary flag column names for the given weather feature set."""
    numeric, flags = WEATHER_NUMERIC.copy(), WEATHER_FLAGS.copy()
    if feature_set in ("bts_origin_destination_weather", *ROTATION_FEATURE_SETS):
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


def build_preprocessor(feature_set: str = "bts", *,
                       empty_enroute: tuple[str, ...] = ()) -> Pipeline:
    """Build the PySpark ML pipeline stages for feature preprocessing.

    Encodes categorical features with StringIndexer and OneHotEncoder, adds
    cyclical month features and filled rolling features, imputes weather
    features with training means when present, and assembles a feature vector.
    """
    if feature_set not in TRAINING_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    include_rotation = feature_set in ROTATION_FEATURE_SETS
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
    upstream_fills = [
        f"COALESCE({name}, {'-1D' if name.endswith('_scheduled_arrivals') else '0D'}) AS {name}_filled"
        for name in UPSTREAM_FEATURES
    ]
    upstream_missing = [f"{role}_upstream_pressure_missing" for role in UPSTREAM_ROLES]
    upstream_fills += [
        f"CAST({role}_upstream_delay_pressure IS NULL AS DOUBLE) AS {name}"
        for role, name in zip(UPSTREAM_ROLES, upstream_missing)
    ]
    numeric += [f"{name}_filled" for name in UPSTREAM_FEATURES] + upstream_missing
    if feature_set != "bts":
        weather_numeric, _ = weather_columns(feature_set)
        imputed = [name + "_imputed" for name in weather_numeric]
        stages.append(Imputer(inputCols=weather_numeric, outputCols=imputed, strategy="mean"))
        # Preserve exact vector ordering: origin numeric + flags, then destination numeric + flags.
        numeric += [name + "_imputed" for name in WEATHER_NUMERIC] + WEATHER_FLAGS
        if feature_set in ("bts_origin_destination_weather", *ROTATION_FEATURE_SETS):
            numeric += [name + "_imputed" for name in DESTINATION_WEATHER_NUMERIC] + DESTINATION_WEATHER_FLAGS
    filled = [*ROLLING_FEATURES, *ROTATION_NUMERIC] if include_rotation else ROLLING_FEATURES
    if include_rotation:
        numeric += [name + "_filled" for name in ROTATION_NUMERIC] + list(ROTATION_FLAGS)
    # Calendar values must remain defined in months absent from the training vocabulary.
    stages.append(SQLTransformer(statement=(
        "SELECT *, SIN(2 * PI() * (Month - 1) / 12) AS MonthSin, "
        "COS(2 * PI() * (Month - 1) / 12) AS MonthCos, "
        + ", ".join([*(f"COALESCE({name}, 0D) AS {name}_filled" for name in filled), *upstream_fills])
        + " FROM __THIS__"
    )))
    if feature_set == ENROUTE_FEATURE_SET:
        impute = [name for name in ENROUTE_NUMERIC if name not in empty_enroute]
        fill = [*ENROUTE_FLAGS, *empty_enroute]
        stages.append(SQLTransformer(statement="SELECT *, " + ", ".join(
            f"COALESCE({name}, 0D) AS {name}_filled" for name in fill
        ) + " FROM __THIS__"))
        if impute:
            stages.append(Imputer(inputCols=impute, outputCols=[f"{name}_filled" for name in impute], strategy="median"))
        numeric += [f"{name}_filled" for name in ENROUTE_FEATURES]
    stages.append(
        VectorAssembler(
            inputCols=vectors + numeric, outputCol="features", handleInvalid="error"
        )
    )
    return Pipeline(stages=stages)


def validate_weather(
    data: DataFrame,
    context: str,
    feature_set: str = "bts_origin_weather",
    require_values: bool = True,
) -> None:
    """Validate that weather numeric features and binary flags conform to expected bounds."""
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


def confusion_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    """Calculate delayed-class classification metrics from raw confusion counts."""
    total = tp + fp + tn + fn
    return {
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_predictions(
    predictions: DataFrame,
    label_col: str = TARGET_COL,
    pred_col: str = "prediction",
    prob_col: str = "p_delayed",
) -> dict[str, Any]:
    """Evaluate binary classification metrics (PR-AUC, ROC-AUC, F1, accuracy) on predictions."""
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


def predict(
    model: PipelineModel,
    data: DataFrame,
    threshold: float = 0.5,
) -> DataFrame:
    """Transform features with model, extracting positive-class probability and applying threshold."""
    return (
        model.transform(data)
        .withColumn("p_delayed", vector_to_array("probability")[1])
        .withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
    )


def tune_threshold(predictions: DataFrame) -> tuple[float, pd.DataFrame]:
    """Search pre-defined thresholds on validation predictions to maximize F1 score."""
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
    spark: SparkSession,
    gold_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
    feature_set: str = "bts",
    save_predictions: bool = False,
    publish_results: bool = True,
    *,
    include_gbt: bool = False,
    model_name: str = "baseline_pipeline",
) -> dict[str, Any]:
    """Train and validate classification models on Gold feature sets.

    Fits preprocessing and classifiers on the training cohort, evaluates
    performance across decision thresholds on the validation cohort, optionally
    saves models and predictions, and updates project results.
    """
    if feature_set not in TRAINING_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    include_rotation = feature_set in ROTATION_FEATURE_SETS
    if include_rotation and (include_gbt or publish_results):
        raise ValueError("Aircraft rotation and en-route weather require an isolated combined-weather logistic comparison.")
    if feature_set != "bts" and any(
        path is None for path in (gold_dir, models_dir)
    ):
        raise ValueError(
            "Origin-weather training requires enriched Gold and isolated model paths."
        )
    gold_dir = Path(gold_dir or active(ROOT, "gold"))
    models_dir = Path(models_dir or ROOT / "models/bts")
    results_dir = Path(results_dir or ROOT / "results")

    train_data = spark.read.parquet(str(gold_dir / "train"))
    val_data = spark.read.parquet(str(gold_dir / "val"))

    columns = INPUT_FEATURES + [TARGET_COL]
    if feature_set != "bts":
        numeric, flags = weather_columns(feature_set)
        columns += numeric + flags
    if include_rotation:
        columns += [*ROTATION_NUMERIC, *ROTATION_FLAGS]
    if feature_set == ENROUTE_FEATURE_SET:
        columns += list(ENROUTE_FEATURES)
    metadata = ["flight_id", "FlightDate"] if save_predictions else []
    # Leave JVM heap for concurrent classifier tasks rather than duplicated training rows.
    train_features = train_data.select("FlightDate", *columns).persist(StorageLevel.DISK_ONLY)
    val_features = val_data.select(*metadata, *columns).persist(StorageLevel.DISK_ONLY)
    selected_scores = None
    timings = {}
    vector_dimension: Optional[int] = None
    imputation_means: dict[str, float] = {}
    empty_enroute = ()

    try:
        with stage(f"{feature_set}: prepare training and validation cohorts",
                   timings=timings, key="prepare_cohorts"):
            training_stats = train_features.agg(
                F.count("*").alias("rows"),
                F.avg(TARGET_COL).alias("prevalence"),
            ).first()
            if feature_set != "bts":
                validate_weather(val_features, "validation", feature_set, require_values=False)
                validate_weather(train_features, "full training", feature_set)
            if feature_set == ENROUTE_FEATURE_SET:
                observed = train_features.agg(*[F.count(name).alias(name) for name in ENROUTE_NUMERIC]).first()
                empty_enroute = tuple(name for name in ENROUTE_NUMERIC if not observed[name])
            val_count = val_features.count()
        prevalence = training_stats["prevalence"]
        print(
            f"  Training cohort:   {training_stats['rows']:,} flights (delay prevalence: {prevalence:.2%})",
            flush=True,
        )
        print(f"  Validation cohort: {val_count:,} flights", flush=True)

        references = {}
        with stage(f"{feature_set}: evaluate constant reference baselines",
                   timings=timings, key="reference_baselines"):
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

        estimators = [
            ("logistic_regression", LogisticRegression(
                featuresCol="features", labelCol=TARGET_COL, family="binomial",
                standardization=True, **LOGISTIC_PARAMS)),
        ]
        if include_gbt:
            estimators.append(("gradient_boosted_trees", GBTClassifier(
                featuresCol="features", labelCol=TARGET_COL, **GBT_PARAMS)))
        algorithms = {}
        for name, estimator in estimators:
            label = f"{feature_set} / {name}"
            algorithm_timings = {}
            print(f"  Fitting {name} on all training flights with fixed settings: "
                  f"{LOGISTIC_PARAMS if name == 'logistic_regression' else GBT_PARAMS}", flush=True)
            print_progress(label, 0, 3)
            with stage(f"{label}: fit preprocessing", timings=algorithm_timings, key="preprocessor_fit"):
                preprocessor = build_preprocessor(
                    feature_set, empty_enroute=empty_enroute,
                ).fit(train_features)
            print_progress(label, 1, 3)
            with stage(f"{label}: fit classifier", timings=algorithm_timings, key="classifier_fit"):
                classifier = estimator.fit(preprocessor.transform(train_features))
                selected_model = PipelineModel(stages=[*preprocessor.stages, classifier])
            print_progress(label, 2, 3)
            with stage(f"{label}: score validation and select threshold",
                       timings=algorithm_timings, key="validation"):
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
                if name == "logistic_regression":
                    vector_dimension = preprocessor.transform(val_features.limit(1)).first()["features"].size
                    imputation_means = next((item.surrogateDF.first().asDict()
                                             for item in preprocessor.stages
                                             if hasattr(item, "surrogateDF")), {})
            print_progress(label, 3, 3)
            # Keep a common logistic reference across weather variants; trees are a separate challenger.
            directory = models_dir if name == "logistic_regression" else models_dir / "gbt"
            with stage(f"{label}: save model and threshold",
                       timings=algorithm_timings, key="save_model_and_threshold"):
                directory.mkdir(parents=True, exist_ok=True)
                selected_model.write().overwrite().save(str(directory / model_name))
                (directory / "threshold.json").write_text(json.dumps({
                    "threshold": threshold, "positive_class": 1,
                    "decision_rule": "p_delayed >= threshold",
                }, indent=2) + "\n", encoding="utf-8")
            if save_predictions:
                with stage(f"{label}: write validation predictions",
                           timings=algorithm_timings, key="write_validation_predictions"):
                    (selected_scores.withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
                     .withColumn("threshold", F.lit(threshold))
                     .write.parquet(str(directory / "validation_predictions")))
            if name == "logistic_regression":
                model_params = {
                    "reg_param": estimator.getRegParam(),
                    "elastic_net_param": estimator.getElasticNetParam(),
                    "max_iter": estimator.getMaxIter(),
                    "tol": estimator.getTol(),
                }
            else:
                model_params = {
                    "max_depth": estimator.getMaxDepth(),
                    "max_iter": estimator.getMaxIter(),
                    "step_size": estimator.getStepSize(),
                    "seed": estimator.getSeed(),
                }
            algorithms[name] = {
                "algorithm": name,
                **model_params,
                "models_dir": str(directory),
                "best_threshold": threshold,
                "threshold_search": records(search_df),
                "validation_metrics_at_selected_threshold": selected_metrics,
                "validation_metrics_at_default_0_5": selected_default,
                "timing_seconds": algorithm_timings,
            }
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
            **({"enroute_all_missing_training_columns": list(empty_enroute)} if feature_set == ENROUTE_FEATURE_SET else {}),
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
            "timing_seconds": timings,
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


def main(*, include_gbt: bool = False) -> int:
    """Run baseline training on BTS features and report validation performance."""
    spark = get_spark_session("bts_training")
    try:
        summary = train(spark, include_gbt=include_gbt)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-gbt", action="store_true", help="Also train the gradient-boosted tree challenger.")
    sys.exit(main(include_gbt=parser.parse_args().include_gbt))
