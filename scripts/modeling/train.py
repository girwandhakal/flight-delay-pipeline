"""Fit and validate logistic regression on the full flight-delay dataset."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, SQLTransformer, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession, functions as F

from scripts.aircraft_rotation.build_features import RECOVERY_FEATURES, ROTATION_FLAGS, ROTATION_NUMERIC
from scripts.bts.block_time import POLICY as BLOCK_TIME_POLICY, SLACK_FEATURES
from scripts.bts.build_features import BTS_FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from scripts.bts.recent_delays import HOLIDAY_FEATURES, ROLLING_FEATURES
from scripts.destination_weather.build_features import DESTINATION_WEATHER_FLAGS, DESTINATION_WEATHER_NUMERIC
from scripts.enroute_weather.build_features import ENROUTE_FLAGS, ENROUTE_NUMERIC, FEATURES as ENROUTE_FEATURES
from scripts.origin_weather.build_features import WEATHER_FLAGS, WEATHER_NUMERIC
from scripts.common.progress import stage
from scripts.common.results import records

SURFACE_NUMERIC = [*WEATHER_NUMERIC, *DESTINATION_WEATHER_NUMERIC]
SURFACE_FLAGS = [*WEATHER_FLAGS, *DESTINATION_WEATHER_FLAGS]
TARGET_COL = "ArrDel15"
INPUT_COLUMNS = [
    *(name for name in BTS_FEATURE_COLUMNS if name not in SLACK_FEATURES), TARGET_COL,
    *SURFACE_NUMERIC, *SURFACE_FLAGS, *ROTATION_NUMERIC, *ROTATION_FLAGS, *ENROUTE_FEATURES,
    *SLACK_FEATURES, *RECOVERY_FEATURES,
]
LOGISTIC_PARAMS = {"regParam": 0.01, "elasticNetParam": 0.0, "maxIter": 100, "tol": 1e-6}
WEIGHTED_LOGISTIC = "class_weighted_logistic_regression"
RANDOM_FOREST = "random_forest"
# maxBins=512 accommodates >300 airports plus StringIndexer's unknown bucket.
RANDOM_FOREST_PARAMS = {
    "numTrees": 50,
    "maxDepth": 10,
    "maxBins": 512,
    "minInstancesPerNode": 50,
    "subsamplingRate": 0.8,
    "featureSubsetStrategy": "sqrt",
    "seed": 42,
}
CLASS_WEIGHT_COL = "class_weight"
THRESHOLDS = [round(i * 0.05, 2) for i in range(1, 20)]
ABLATION_GROUPS: dict[str, dict[str, bool]] = {
    "bts_only": {"surface_weather": False, "rotation": False, "enroute": False},
    "bts_surface_weather": {"surface_weather": True, "rotation": False, "enroute": False},
    "bts_surface_rotation": {"surface_weather": True, "rotation": True, "enroute": False},
    "full": {"surface_weather": True, "rotation": True, "enroute": True},
}


def build_preprocessor(
    *,
    empty_enroute: tuple[str, ...] = (),
    one_hot: bool = True,
    feature_group: str = "full",
) -> Pipeline:
    """Build the feature vector for a specified ablation feature group in its established order."""
    if feature_group not in ABLATION_GROUPS:
        raise ValueError(f"Unknown feature_group {feature_group!r}; expected one of {tuple(ABLATION_GROUPS)}")
    group = ABLATION_GROUPS[feature_group]
    indices = [name + "_idx" for name in CATEGORICAL_FEATURES]
    vectors = [name + "_vec" for name in CATEGORICAL_FEATURES]
    categorical_outputs = vectors if one_hot else indices
    filled = [*ROLLING_FEATURES, SLACK_FEATURES[0]]
    if group["rotation"]:
        filled.extend([*ROTATION_NUMERIC, RECOVERY_FEATURES[0]])
    stages = [
        StringIndexer(inputCols=CATEGORICAL_FEATURES, outputCols=indices, handleInvalid="keep"),
    ]
    if one_hot:
        # Linear models use one-hot indicators; tree models split directly on StringIndexer metadata.
        stages.append(OneHotEncoder(inputCols=indices, outputCols=vectors, handleInvalid="keep", dropLast=False))
    if group["surface_weather"]:
        stages.append(
            Imputer(inputCols=SURFACE_NUMERIC, outputCols=[name + "_imputed" for name in SURFACE_NUMERIC],
                    strategy="mean"),
        )
    stages.append(
        SQLTransformer(statement=(
            "SELECT *, SIN(2 * PI() * (Month - 1) / 12) AS MonthSin, "
            "COS(2 * PI() * (Month - 1) / 12) AS MonthCos, "
            + ", ".join(f"COALESCE({name}, 0D) AS {name}_filled" for name in filled)
            + " FROM __THIS__"
        )),
    )
    if group["enroute"]:
        enroute_imputed = [name for name in ENROUTE_NUMERIC if name not in empty_enroute]
        stages.append(
            SQLTransformer(statement="SELECT *, " + ", ".join(
                f"COALESCE({name}, 0D) AS {name}_filled" for name in (*ENROUTE_FLAGS, *empty_enroute)
            ) + " FROM __THIS__"),
        )
        if enroute_imputed:
            stages.append(Imputer(inputCols=enroute_imputed, outputCols=[f"{name}_filled" for name in enroute_imputed],
                                  strategy="median"))
    assembler_inputs = [
        *categorical_outputs, *NUMERIC_FEATURES, "MonthSin", "MonthCos",
        *(f"{name}_filled" for name in ROLLING_FEATURES), *HOLIDAY_FEATURES,
    ]
    if group["surface_weather"]:
        assembler_inputs.extend([
            *(f"{name}_imputed" for name in WEATHER_NUMERIC), *WEATHER_FLAGS,
            *(f"{name}_imputed" for name in DESTINATION_WEATHER_NUMERIC), *DESTINATION_WEATHER_FLAGS,
        ])
    if group["rotation"]:
        assembler_inputs.extend([*(f"{name}_filled" for name in ROTATION_NUMERIC), *ROTATION_FLAGS])
    if group["enroute"]:
        assembler_inputs.extend(f"{name}_filled" for name in ENROUTE_FEATURES)
    assembler_inputs.extend([f"{SLACK_FEATURES[0]}_filled", SLACK_FEATURES[1]])
    if group["rotation"]:
        assembler_inputs.extend([f"{RECOVERY_FEATURES[0]}_filled", RECOVERY_FEATURES[1]])
    stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="error"))
    return Pipeline(stages=stages)


def validate_weather(
    data: DataFrame,
    context: str,
    require_values: bool = True,
) -> None:
    """Validate surface weather and require values for training-time imputation."""
    if require_values:
        counts = data.agg(
            *(F.count(F.when(F.col(name).isNotNull() & ~F.isnan(name), 1)).alias(name)
              for name in SURFACE_NUMERIC)
        ).first()
        for name in SURFACE_NUMERIC:
            if not counts[name]:
                raise ValueError(f"No valid training values for {name} in {context}")
    invalid = F.lit(False)
    for name in SURFACE_NUMERIC:
        invalid = invalid | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    for name in SURFACE_FLAGS:
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


def per_class_metrics(delayed: dict[str, Any]) -> dict[str, dict[str, float]]:
    counts = delayed["confusion_matrix"]
    on_time = confusion_metrics(counts["tn"], counts["fn"], counts["tp"], counts["fp"])
    return {label: {key: metrics[key] for key in ("precision", "recall", "f1")}
            for label, metrics in (("delayed", delayed), ("on_time", on_time))}


def evaluate_predictions(
    predictions: DataFrame,
    label_col: str = TARGET_COL,
    pred_col: str = "prediction",
    prob_col: str = "p_delayed",
) -> dict[str, Any]:
    """Evaluate binary classification metrics (PR-AUC, ROC-AUC, F1, accuracy) on predictions."""
    label = F.col(label_col)
    predicted = F.col(pred_col)

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
    # Break F1 ties in favor of the higher threshold.
    best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    return best["threshold"], search_df


def train(spark: SparkSession, gold_dir: Path, models_dir: Path, splits: dict) -> dict[str, Any]:
    """Fit both model variants on training and select thresholds on validation."""
    train_data = spark.read.parquet(str(gold_dir / "train"))
    val_data = spark.read.parquet(str(gold_dir / "val"))

    # Leave JVM heap for model fitting while retaining identical inputs for both variants.
    train_features = train_data.select("FlightDate", *INPUT_COLUMNS).persist(StorageLevel.DISK_ONLY)
    val_features = val_data.select("flight_id", "FlightDate", *INPUT_COLUMNS).persist(StorageLevel.DISK_ONLY)
    selected_scores = None
    timings = {}
    vector_dimension: Optional[int] = None
    imputation_means: dict[str, float] = {}
    empty_enroute = ()

    try:
        with stage("Prepare training and validation cohorts",
                   timings=timings, key="prepare_cohorts"):
            training_stats = train_features.agg(
                F.count("*").alias("rows"),
                F.avg(TARGET_COL).alias("prevalence"),
            ).first()
            validate_weather(val_features, "validation", require_values=False)
            validate_weather(train_features, "full training")
            observed = train_features.agg(*(F.count(name).alias(name) for name in ENROUTE_NUMERIC)).first()
            empty_enroute = tuple(name for name in ENROUTE_NUMERIC if not observed[name])
            val_count = val_features.count()
        prevalence = training_stats["prevalence"]
        print(
            f"  Training cohort:   {training_stats['rows']:,} flights (delay prevalence: {prevalence:.2%})",
            flush=True,
        )
        print(f"  Validation cohort: {val_count:,} flights", flush=True)

        references = {}
        with stage("Evaluate constant references",
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

        logistic = LogisticRegression(featuresCol="features", labelCol=TARGET_COL, family="binomial",
                                      standardization=True, **LOGISTIC_PARAMS)
        estimators = [("logistic_regression", logistic, models_dir)]
        if 0 < prevalence < 1:
            class_weights = {"on_time": 0.5 / (1 - prevalence), "delayed": 0.5 / prevalence}
            estimators.append((WEIGHTED_LOGISTIC, logistic.copy({logistic.weightCol: CLASS_WEIGHT_COL}),
                               models_dir / "weighted_logistic"))
        else:
            print("  Skipping class-weighted logistic regression because training has one class.", flush=True)
        rf = RandomForestClassifier(featuresCol="features", labelCol=TARGET_COL, **RANDOM_FOREST_PARAMS)
        estimators.append((RANDOM_FOREST, rf, models_dir / RANDOM_FOREST))

        algorithms = {}
        logistic_preprocessor = None
        for name, estimator, directory in estimators:
            weighted = name == WEIGHTED_LOGISTIC
            is_tree = name == RANDOM_FOREST
            algorithm_timings = {}
            params = RANDOM_FOREST_PARAMS if is_tree else LOGISTIC_PARAMS
            print(f"  Fitting {name} on all training flights with fixed settings: {params}", flush=True)
            # Both logistic models share the one-hot mapping; tree models use indexed categories.
            if name == "logistic_regression":
                with stage(f"{name}: fit preprocessing", timings=algorithm_timings, key="preprocessor_fit"):
                    logistic_preprocessor = build_preprocessor(empty_enroute=empty_enroute, one_hot=True).fit(train_features)
                active_preprocessor = logistic_preprocessor
            elif is_tree:
                with stage(f"{name}: fit preprocessing", timings=algorithm_timings, key="preprocessor_fit"):
                    active_preprocessor = build_preprocessor(empty_enroute=empty_enroute, one_hot=False).fit(train_features)
            else:
                active_preprocessor = logistic_preprocessor
            with stage(f"{name}: fit classifier", timings=algorithm_timings, key="classifier_fit"):
                fit_features = active_preprocessor.transform(train_features)
                if weighted:
                    fit_features = fit_features.withColumn(
                        CLASS_WEIGHT_COL, F.when(F.col(TARGET_COL) == 1, class_weights["delayed"])
                        .otherwise(class_weights["on_time"]),
                    )
                classifier = estimator.fit(fit_features)
                selected_model = PipelineModel(stages=[*active_preprocessor.stages, classifier])
            with stage(f"{name}: score validation and select threshold",
                       timings=algorithm_timings, key="validation"):
                selected_scores = (
                    predict(selected_model, val_features)
                    .select("flight_id", "FlightDate", TARGET_COL, "p_delayed", "prediction")
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
                    vector_dimension = active_preprocessor.transform(val_features.limit(1)).first()["features"].size
                    imputation_means = next((item.surrogateDF.first().asDict()
                                             for item in active_preprocessor.stages
                                             if hasattr(item, "surrogateDF")), {})
            with stage(f"{name}: save model and threshold",
                       timings=algorithm_timings, key="save_model_and_threshold"):
                directory.mkdir(parents=True, exist_ok=True)
                selected_model.write().overwrite().save(str(directory / "model"))
                (directory / "threshold.json").write_text(json.dumps({
                    "threshold": threshold, "positive_class": 1,
                    "decision_rule": "p_delayed >= threshold",
                }, indent=2) + "\n", encoding="utf-8")
            with stage(f"{name}: write validation predictions",
                       timings=algorithm_timings, key="write_validation_predictions"):
                (selected_scores.withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
                 .withColumn("threshold", F.lit(threshold))
                 .write.parquet(str(directory / "validation_predictions")))
            if is_tree:
                model_params = {
                    "num_trees": estimator.getNumTrees(),
                    "max_depth": estimator.getMaxDepth(),
                    "max_bins": estimator.getMaxBins(),
                    "min_instances_per_node": estimator.getMinInstancesPerNode(),
                    "subsampling_rate": estimator.getSubsamplingRate(),
                    "feature_subset_strategy": estimator.getFeatureSubsetStrategy(),
                    "seed": estimator.getSeed(),
                }
            else:
                model_params = {
                    "reg_param": estimator.getRegParam(),
                    "elastic_net_param": estimator.getElasticNetParam(),
                    "max_iter": estimator.getMaxIter(),
                    "tol": estimator.getTol(),
                }
                if weighted:
                    model_params["class_weights"] = class_weights
            algorithms[name] = {
                "algorithm": name,
                **model_params,
                "models_dir": str(directory),
                "best_threshold": threshold,
                "threshold_search": records(search_df),
                "validation_metrics_at_selected_threshold": selected_metrics,
                "validation_metrics_at_default_0_5": selected_default,
                "validation_per_class_at_selected_threshold": per_class_metrics(selected_metrics),
                "timing_seconds": algorithm_timings,
            }
            pr_auc_text = f"{selected_default['pr_auc']:.4f}" if selected_default["pr_auc"] is not None else "N/A"
            print(
                f"  {name}: validation PR-AUC={pr_auc_text}, "
                f"F1={selected_metrics['f1']:.4f}, threshold={threshold:.2f}",
                flush=True,
            )
            selected_scores.unpersist(blocking=True)
            selected_scores = None

        baseline = algorithms["logistic_regression"]

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models_dir": str(models_dir),
            "splits": splits,
            "threshold_search": baseline["threshold_search"],
            "input_columns": INPUT_COLUMNS,
            "block_time": {
                "columns": [*SLACK_FEATURES, *RECOVERY_FEATURES],
                "policy": BLOCK_TIME_POLICY,
                "rotation_scope": "Final BTS tail assignments and prior gate arrivals through scheduled departure.",
            },
            "gold_dir": str(gold_dir),
            "vector_dimension": vector_dimension,
            "imputation_means": imputation_means,
            "enroute_all_missing_training_columns": list(empty_enroute),
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
                "scope": "Each algorithm is fitted once on all training rows. F1 thresholds use validation only. "
                         "Unweighted logistic regression remains the reference.",
            },
            "reference_baselines": references,
        }

        return summary
    finally:
        if selected_scores is not None:
            selected_scores.unpersist()
        val_features.unpersist(blocking=True)
        train_features.unpersist(blocking=True)
