"""Publish a completed training run, validation report, and human-readable summary."""

from pathlib import Path
from typing import Any

from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.io import sha256, write_json
from scripts.common.progress import DIVIDER, pipeline_stage
from scripts.common.results import update_results
from scripts.common.stage_cache import consumed

from . import train

RESULT_KEY = "flight_delay"
RUNS_DIR = Path("models") / RESULT_KEY
GOLD_DIR = Path("data/gold/enroute")


def gold_identity(gold: Path, splits: tuple[str, ...]) -> dict:
    """Identify a completed Gold generation and the requested partitions."""
    return {
        "manifest_sha256": sha256(gold / "_stage.json"),
        "partitions": consumed(gold, splits),
    }


def inference_hashes(run_dir: Path, model: str, threshold: str) -> dict[str, str]:
    """Identify every persisted model file and its threshold."""
    model_path = project_path(run_dir, model)
    files = sorted(path for path in model_path.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No saved model files: {model}")
    paths = [*files, project_path(run_dir, threshold)]
    names = [path.relative_to(run_dir).as_posix() for path in paths]
    return {name: sha256(project_path(run_dir, name)) for name in names}


def format_classification_report(
    run_id: str,
    models: dict[str, Any],
    split: str = "VALIDATION",
) -> str:
    """Format a plain-text classification report for all fitted model variants."""
    lines = [
        "",
        DIVIDER,
        f"{split} CLASSIFICATION REPORT (Run ID: {run_id})",
        DIVIDER,
    ]
    for name, info in models.items():
        if "validation_metrics_at_selected_threshold" in info:
            metrics = info["validation_metrics_at_selected_threshold"]
            threshold = info.get("best_threshold")
        else:
            metrics = info.get("metrics_at_selected_threshold", {})
            threshold = info.get("selected_threshold")

        pr_auc = metrics.get("pr_auc")
        roc_auc = metrics.get("roc_auc")
        pr_str = f"{pr_auc:.4f}" if isinstance(pr_auc, (int, float)) else "N/A"
        roc_str = f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else "N/A"
        thr_str = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "N/A"
        role = " (Primary)" if name == "logistic_regression" else ""
        lines.append(
            f"\nModel: {name}{role} | Threshold: {thr_str} | PR-AUC: {pr_str} | ROC-AUC: {roc_str}"
        )

        cm = metrics.get("confusion_matrix")
        if cm:
            tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
            on_time = train.confusion_metrics(tn, fn, tp, fp)
            delayed = metrics
            lines.append(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            lines.append(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
            lines.append(
                f"  {'on_time (0)':<12} {on_time['precision']:>10.4f} {on_time['recall']:>10.4f} "
                f"{on_time['f1']:>10.4f} {tn + fp:>10,}"
            )
            lines.append(
                f"  {'delayed (1)':<12} {delayed['precision']:>10.4f} {delayed['recall']:>10.4f} "
                f"{delayed['f1']:>10.4f} {tp + fn:>10,}"
            )
            lines.append(
                f"  Accuracy: {metrics.get('accuracy', 0.0):.4f} | "
                f"Confusion Matrix: TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}"
            )
        else:
            f1 = metrics.get("f1")
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
            lines.append(f"  F1-Score: {f1_str}")

    lines.append(DIVIDER)
    return "\n".join(lines)


def train_and_publish(
    spark, root: Path, run_id: str, gold: Path, splits: dict, coverage: dict, sources: dict,
) -> Path:
    """Train once, save artifacts and manifests, and publish validation results."""
    run_dir = project_path(root, RUNS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    gold = gold.resolve()
    if gold.parent != project_path(root, GOLD_DIR):
        raise ValueError("Training Gold must be under the en-route Gold directory.")

    partition_names = ("train", "val", "test")
    identity = gold_identity(gold, partition_names)
    summary = train.train(spark, gold, run_dir, splits)

    variants = {}
    for name, algorithm in summary["algorithm_comparison"].items():
        directory = Path(algorithm["models_dir"]).relative_to(run_dir)
        model = (directory / "model").as_posix()
        threshold = (directory / "threshold.json").as_posix()
        variants[name] = {
            "model": model,
            "threshold_file": threshold,
            "threshold": algorithm["best_threshold"],
            "sha256": inference_hashes(run_dir, model, threshold),
        }

    selected = summary["selected_model"]
    validation = {
        "selected": {
            **selected["validation_metrics_at_selected_threshold"],
            "threshold": selected["best_threshold"],
        },
        "common_0_5": selected["validation_metrics_at_default_0_5"],
    }
    report = run_dir / "validation.json"
    write_json(report, {
        "run_id": run_id,
        "validation": validation,
        "coverage": coverage,
        "algorithm_comparison": summary["algorithm_comparison"],
        "block_time": summary["block_time"],
        "scope": "Thresholds were selected on this validation cohort.",
    })
    manifest = run_dir / "run.json"
    write_json(manifest, {
        "run_id": run_id,
        "splits": splits,
        "gold": gold.relative_to(root.resolve()).as_posix(),
        "gold_identity": identity,
        "input_columns": summary["input_columns"],
        "sources": sources,
        "variants": variants,
    })

    marker = run_dir / "_SUCCESS"
    marker.touch(exist_ok=False)
    try:
        update_results({RESULT_KEY: {
            "run_id": run_id,
            "run_sha256": sha256(manifest),
            "training": summary,
            "validation": validation,
            "coverage": coverage,
        }}, root / "results")
    except BaseException:
        marker.unlink()
        raise

    print(format_classification_report(run_id, summary["algorithm_comparison"], split="VALIDATION"), flush=True)
    return report


def run(ctx: PipelineContext) -> Path:
    """Stage 6 entry point: fit models on full-feature Gold and publish validation."""
    with pipeline_stage(6, 6, "Model Training & Validation", timings=ctx.timings):
        return train_and_publish(
            ctx.spark,
            ctx.root,
            ctx.run_id,
            ctx.gold_dir,
            ctx.config["splits"],
            ctx.coverage,
            ctx.sources,
        )
