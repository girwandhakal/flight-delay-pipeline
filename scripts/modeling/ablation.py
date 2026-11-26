"""Run the controlled feature-set ablation study (RQ2) on the pinned Gold cohort."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Optional

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F

from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.progress import DIVIDER, stage
from scripts.common.results import update_results
from scripts.enroute_weather.build_features import ENROUTE_NUMERIC

from . import train

ABLATION_RESULT_KEY = "ablation"
GOLD_DIR = Path("data/gold/enroute")


def _resolve_gold_dir(ctx: PipelineContext) -> Path:
    """Resolve the active Gold generation directory containing train, val, and test splits."""
    candidates: list[Path] = []
    if ctx.gold_dir is not None:
        candidates.append(ctx.gold_dir)

    results_path = project_path(ctx.root, "results/results.json")
    if results_path.is_file():
        active_run = json.loads(results_path.read_text(encoding="utf-8")).get("flight_delay", {}).get("run_id")
        if isinstance(active_run, str) and active_run:
            manifest_path = project_path(ctx.root, f"models/flight_delay/{active_run}/run.json")
            if manifest_path.is_file():
                manifest_gold = json.loads(manifest_path.read_text(encoding="utf-8")).get("gold")
                if isinstance(manifest_gold, str) and manifest_gold:
                    candidates.append(project_path(ctx.root, manifest_gold))

    gold_root = project_path(ctx.root, GOLD_DIR)
    if gold_root.exists():
        completed = sorted(gold_root.glob("*/_SUCCESS"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(marker.parent for marker in completed)
    candidates.append(gold_root)

    for candidate in candidates:
        if all((candidate / split).exists() for split in ("train", "val", "test")):
            return candidate
    raise FileNotFoundError(
        f"Missing Gold train/val/test splits under {gold_root}. Run `python pipeline/main.py` first to build Gold data."
    )


def _extract_phase1_full_metrics(root: Path, run_id: str) -> Optional[dict[str, Any]]:
    """Reuse Phase 1 full-feature metrics for class_weighted_logistic_regression when run_id matches."""
    results_path = project_path(root, "results/results.json")
    if not results_path.is_file():
        return None
    fd = json.loads(results_path.read_text(encoding="utf-8")).get("flight_delay", {})
    if fd.get("run_id") != run_id:
        return None

    training = fd.get("training", {})
    algo_val = training.get("algorithm_comparison", {}).get(train.WEIGHTED_LOGISTIC)
    algo_test = fd.get("test", {}).get("models", {}).get(train.WEIGHTED_LOGISTIC)
    if not isinstance(algo_val, dict) or not isinstance(algo_test, dict):
        return None

    return {
        "vector_dimension": training.get("vector_dimension"),
        "algorithms": {
            train.WEIGHTED_LOGISTIC: {
                "best_threshold": algo_val["best_threshold"],
                "validation_metrics_at_selected_threshold": algo_val["validation_metrics_at_selected_threshold"],
                "validation_metrics_at_default_0_5": algo_val["validation_metrics_at_default_0_5"],
                "test_metrics_at_selected_threshold": algo_test["metrics_at_selected_threshold"],
                "test_metrics_at_default_0_5": algo_test["metrics_at_default_0_5"],
                "reused_from_run_id": run_id,
            }
        },
    }


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

    gold_dir = _resolve_gold_dir(ctx)
    phase1_full = _extract_phase1_full_metrics(ctx.root, ctx.run_id)

    with stage("Controlled feature ablation study (RQ2)", timings=ctx.timings, key="ablation"):
        required_cols = [*train.INPUT_COLUMNS, "flight_id", "FlightDate"]
        train_df = spark.read.parquet(str(gold_dir / "train")).select(*required_cols).persist(StorageLevel.MEMORY_AND_DISK)
        val_df = spark.read.parquet(str(gold_dir / "val")).select(*required_cols).persist(StorageLevel.MEMORY_AND_DISK)
        test_df = spark.read.parquet(str(gold_dir / "test")).select(*required_cols).persist(StorageLevel.MEMORY_AND_DISK)

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
            if not 0 < prevalence < 1:
                raise ValueError("Class-weighted logistic regression requires both target classes in training.")
            class_weights = {"on_time": 0.5 / (1 - prevalence), "delayed": 0.5 / prevalence}
            estimators: list[tuple[str, Any]] = [
                (train.WEIGHTED_LOGISTIC, logistic.copy({logistic.weightCol: train.CLASS_WEIGHT_COL})),
            ]

            feature_groups_out: dict[str, Any] = {}
            group_timings: dict[str, float] = {}
            for group_name in train.ABLATION_GROUPS:
                if group_name == "full" and phase1_full is not None:
                    print("\n  [Ablation] Feature group: full (reusing Phase 1 metrics)", flush=True)
                    feature_groups_out[group_name] = phase1_full
                    group_timings[group_name] = 0.0
                    continue

                group_start = time.monotonic()
                print(f"\n  [Ablation] Feature group: {group_name}", flush=True)
                algorithms_out: dict[str, Any] = {}
                vector_dimension = None

                for algo_name, estimator in estimators:
                    active_preprocessor = train.build_preprocessor(
                        empty_enroute=empty_enroute, one_hot=True, feature_group=group_name,
                    ).fit(train_df)

                    fit_features = active_preprocessor.transform(train_df).withColumn(
                        train.CLASS_WEIGHT_COL,
                        F.when(F.col(train.TARGET_COL) == 1, class_weights["delayed"])
                        .otherwise(class_weights["on_time"]),
                    )
                    classifier = estimator.fit(fit_features)
                    model = PipelineModel(stages=[*active_preprocessor.stages, classifier])

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

                group_timings[group_name] = round(time.monotonic() - group_start, 2)
                feature_groups_out[group_name] = {
                    "vector_dimension": vector_dimension,
                    "timing_seconds": group_timings[group_name],
                    "algorithms": algorithms_out,
                }

            payload = {
                "run_id": ctx.run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gold_dir": str(gold_dir),
                "reused_phase1_full": phase1_full is not None,
                "cohorts": {"train_rows": train_rows, "val_rows": val_rows, "test_rows": test_rows},
                "timing_seconds": group_timings,
                "feature_groups": feature_groups_out,
            }
            results_path = update_results({ABLATION_RESULT_KEY: payload}, project_path(ctx.root, "results"))
            print_ablation_summary(payload, results_path)
            return payload
        finally:
            test_df.unpersist(blocking=True)
            val_df.unpersist(blocking=True)
            train_df.unpersist(blocking=True)
