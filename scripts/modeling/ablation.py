"""Run the controlled feature-set ablation study (RQ2) on the pinned Gold cohort."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import functions as F

from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.io import fs_path
from scripts.common.progress import DIVIDER, pipeline_stage, stage
from scripts.common.results import update_results
from scripts.enroute_weather.build_features import ENROUTE_NUMERIC

from . import train

ABLATION_RESULT_KEY = "ablation"
GOLD_DIR = Path("data/gold/enroute")


def _score_split(model: PipelineModel, split_df, threshold: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score a split at both default 0.5 and the validation-selected threshold."""
    scores = (
        train.predict(model, split_df)
        .select("flight_id", "FlightDate", train.TARGET_COL, "p_delayed", "prediction")
        .cache()
    )
    try:
        default_metrics = train.evaluate_predictions(scores)
        thresholded = scores.withColumn("prediction", (F.col("p_delayed") >= threshold).cast("int"))
        selected_metrics = train.confusion_metrics(
            **{
                key: int(row[key])
                for row in [
                    thresholded.agg(
                        F.sum(F.when((F.col(train.TARGET_COL) == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
                        F.sum(F.when((F.col(train.TARGET_COL) == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
                        F.sum(F.when((F.col(train.TARGET_COL) == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
                        F.sum(F.when((F.col(train.TARGET_COL) == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
                    ).first()
                ]
                for key in ("tp", "fp", "tn", "fn")
            }
        )
        selected_metrics["pr_auc"] = default_metrics["pr_auc"]
        selected_metrics["roc_auc"] = default_metrics["roc_auc"]
        return selected_metrics, default_metrics
    finally:
        scores.unpersist(blocking=True)


def print_ablation_summary(ablation_payload: dict[str, Any], results_path: Path) -> None:
    """Print a readable comparison table of validation and test metrics across feature groups."""
    print(f"\n{DIVIDER}", flush=True)
    print(f"ABLATION STUDY SUMMARY (Run ID: {ablation_payload['run_id']})", flush=True)
    print(DIVIDER, flush=True)
    header = f"  {'Feature Group':<22} {'Algorithm':<34} {'Val PR-AUC':>10} {'Val F1':>8} {'Test PR-AUC':>11} {'Test F1':>8} {'Test ROC':>8}"
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for group_name, group_data in ablation_payload["feature_groups"].items():
        for algo_name, metrics in group_data["algorithms"].items():
            val_m = metrics["validation_metrics_at_selected_threshold"]
            test_m = metrics["test_metrics_at_selected_threshold"]
            val_pr = f"{val_m['pr_auc']:.4f}" if val_m.get("pr_auc") is not None else "N/A"
            test_pr = f"{test_m['pr_auc']:.4f}" if test_m.get("pr_auc") is not None else "N/A"
            test_roc = f"{test_m['roc_auc']:.4f}" if test_m.get("roc_auc") is not None else "N/A"
            print(
                f"  {group_name:<22} {algo_name:<34} {val_pr:>10} {val_m['f1']:>8.4f} "
                f"{test_pr:>11} {test_m['f1']:>8.4f} {test_roc:>8}",
                flush=True,
            )
    print(DIVIDER, flush=True)
    print(f"  Published to: {results_path} (key: '{ABLATION_RESULT_KEY}')", flush=True)
    print(DIVIDER, flush=True)


def run_ablation(ctx: PipelineContext) -> dict[str, Any]:
    """Run the four-group controlled ablation study on the pinned Gold train/val/test splits."""
    spark = ctx.spark
    if spark is None:
        raise RuntimeError("PipelineContext.spark must be initialized before running ablation.")

    gold_dir = project_path(ctx.root, GOLD_DIR)
    for split in ("train", "val", "test"):
        split_dir = gold_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing Gold split at {split_dir}. Run `python pipeline/main.py` first to build Gold data."
            )

    with pipeline_stage("Controlled feature ablation study (RQ2)", ctx=ctx, key="ablation"):
        required_cols = [*train.INPUT_COLUMNS, "flight_id", "FlightDate"]
        train_df = spark.read.parquet(fs_path(gold_dir / "train")).select(*required_cols).persist(StorageLevel.MEMORY_AND_DISK)
        val_df = spark.read.parquet(fs_path(gold_dir / "val")).select(*required_cols).persist(StorageLevel.MEMORY_AND_DISK)
        test_df = spark.read.parquet(fs_path(gold_dir / "test")).select(*required_cols).persist(StorageLevel.MEMORY_AND_DISK)

        try:
            with stage("Load Gold cohorts and compute prevalence"):
                stats = train_df.agg(F.count("*").alias("rows"), F.avg(train.TARGET_COL).alias("prevalence")).first()
                prevalence = float(stats["prevalence"])
                train_rows = int(stats["rows"])
                val_rows = val_df.count()
                test_rows = test_df.count()
                observed = train_df.agg(*(F.count(name).alias(name) for name in ENROUTE_NUMERIC)).first()
                empty_enroute = tuple(name for name in ENROUTE_NUMERIC if not observed[name])

            logistic = LogisticRegression(
                featuresCol="features", labelCol=train.TARGET_COL, family="binomial",
                standardization=True, **train.LOGISTIC_PARAMS,
            )
            estimators: list[tuple[str, Any]] = [("logistic_regression", logistic)]
            if 0 < prevalence < 1:
                class_weights = {"on_time": 0.5 / (1 - prevalence), "delayed": 0.5 / prevalence}
                estimators.append((train.WEIGHTED_LOGISTIC, logistic.copy({logistic.weightCol: train.CLASS_WEIGHT_COL})))
            else:
                class_weights = {}
            estimators.append((
                train.RANDOM_FOREST,
                RandomForestClassifier(featuresCol="features", labelCol=train.TARGET_COL, **train.RANDOM_FOREST_PARAMS),
            ))

            feature_groups_out: dict[str, Any] = {}
            for group_name in train.ABLATION_GROUPS:
                print(f"\n  [Ablation] Feature group: {group_name}", flush=True)
                algorithms_out: dict[str, Any] = {}
                logistic_preprocessor = None
                vector_dimension = None

                for algo_name, estimator in estimators:
                    weighted = algo_name == train.WEIGHTED_LOGISTIC
                    is_tree = algo_name == train.RANDOM_FOREST
                    if algo_name == "logistic_regression":
                        logistic_preprocessor = train.build_preprocessor(
                            empty_enroute=empty_enroute, one_hot=True, feature_group=group_name,
                        ).fit(train_df)
                        active_preprocessor = logistic_preprocessor
                    elif is_tree:
                        active_preprocessor = train.build_preprocessor(
                            empty_enroute=empty_enroute, one_hot=False, feature_group=group_name,
                        ).fit(train_df)
                    else:
                        active_preprocessor = logistic_preprocessor

                    fit_features = active_preprocessor.transform(train_df)
                    if weighted:
                        fit_features = fit_features.withColumn(
                            train.CLASS_WEIGHT_COL,
                            F.when(F.col(train.TARGET_COL) == 1, class_weights["delayed"])
                            .otherwise(class_weights["on_time"]),
                        )
                    classifier = estimator.fit(fit_features)
                    model = PipelineModel(stages=[*active_preprocessor.stages, classifier])

                    if algo_name == "logistic_regression":
                        vector_dimension = active_preprocessor.transform(val_df.limit(1)).first()["features"].size

                    val_scores = (
                        train.predict(model, val_df)
                        .select("flight_id", "FlightDate", train.TARGET_COL, "p_delayed", "prediction")
                        .cache()
                    )
                    try:
                        val_default = train.evaluate_predictions(val_scores)
                        threshold, search_df = train.tune_threshold(val_scores)
                        best_row = search_df.loc[search_df["threshold"] == threshold].iloc[0]
                        val_selected = train.confusion_metrics(
                            **{key: int(best_row[key]) for key in ("tp", "fp", "tn", "fn")}
                        )
                        val_selected["pr_auc"] = val_default["pr_auc"]
                        val_selected["roc_auc"] = val_default["roc_auc"]
                    finally:
                        val_scores.unpersist(blocking=True)

                    test_selected, test_default = _score_split(model, test_df, threshold)
                    algorithms_out[algo_name] = {
                        "best_threshold": threshold,
                        "validation_metrics_at_selected_threshold": val_selected,
                        "validation_metrics_at_default_0_5": val_default,
                        "test_metrics_at_selected_threshold": test_selected,
                        "test_metrics_at_default_0_5": test_default,
                    }
                    print(
                        f"    {algo_name}: val_PR-AUC={val_selected['pr_auc']:.4f}, "
                        f"val_F1={val_selected['f1']:.4f}, test_PR-AUC={test_selected['pr_auc']:.4f}, "
                        f"test_F1={test_selected['f1']:.4f} (threshold={threshold:.2f})",
                        flush=True,
                    )

                feature_groups_out[group_name] = {
                    "vector_dimension": vector_dimension,
                    "algorithms": algorithms_out,
                }

            payload = {
                "run_id": ctx.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gold_dir": str(gold_dir),
                "cohorts": {"train_rows": train_rows, "val_rows": val_rows, "test_rows": test_rows},
                "feature_groups": feature_groups_out,
            }
            results_path = update_results(project_path(ctx.root, "results"), ABLATION_RESULT_KEY, payload)
            print_ablation_summary(payload, results_path)
            return payload
        finally:
            test_df.unpersist(blocking=True)
            val_df.unpersist(blocking=True)
            train_df.unpersist(blocking=True)
