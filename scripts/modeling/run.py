"""Publish a completed training run, validation report, and human-readable summary."""

import json
from pathlib import Path
import re
import shutil
from typing import Any

from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.io import dir_size_bytes, fs_path, rmtree_retry, sha256, write_json
from scripts.common.progress import DIVIDER, pipeline_stage
from scripts.common.results import update_results
from scripts.common.stage_cache import consumed

from . import train

RESULT_KEY = "flight_delay"
RUNS_DIR = Path("models") / RESULT_KEY
GOLD_DIR = Path("data/gold/enroute")
LEGACY_MODEL_DIRS = (
    "bts",
    "class_weight_comparison",
    "destination_weather_comparison",
    "dw",
    "enroute",
    "origin_weather_comparison",
    "tier5_tree_comparison",
)


def _safe_prune_entry(candidate: Path, expected_parent: Path) -> int:
    """Validate containment, refuse symlinks, and remove an obsolete model entry."""
    if candidate.is_symlink():
        raise ValueError(f"Refusing symlinked model path: {candidate}")
    resolved_parent = expected_parent.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate.parent != resolved_parent or not resolved_candidate.is_relative_to(resolved_parent):
        raise ValueError(f"Refusing path outside {expected_parent}: {candidate}")
    if candidate.is_dir():
        freed = dir_size_bytes(candidate)
        rmtree_retry(candidate)
        return freed
    freed = candidate.stat().st_size
    candidate.unlink()
    return freed


def prune_model_runs(root: Path, keep_run_id: str) -> list[Path]:
    """Delete all superseded or incomplete runs under models/flight_delay except keep_run_id."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", keep_run_id):
        raise ValueError("Use the saved directory name as keep_run_id.")
    runs_root = project_path(root, RUNS_DIR)
    if not runs_root.exists():
        return []
    removed: list[Path] = []
    for candidate in sorted(runs_root.iterdir()):
        if candidate.name == keep_run_id:
            continue
        _safe_prune_entry(candidate, runs_root)
        removed.append(candidate)
    return removed


def prune_legacy_models(root: Path, *, require_active_run: bool = True) -> list[Path]:
    """Remove allowlisted pre-refactor directories under models/ once an active run exists."""
    models_root = project_path(root, "models")
    if not models_root.exists():
        return []
    if require_active_run:
        results_path = project_path(root, "results/results.json")
        if not results_path.is_file():
            raise ValueError("Legacy model cleanup requires results/results.json with an active flight_delay run.")
        current = json.loads(results_path.read_text(encoding="utf-8")).get(RESULT_KEY, {})
        active_run_id = current.get("run_id") if isinstance(current, dict) else None
        if not isinstance(active_run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", active_run_id):
            raise ValueError("Legacy model cleanup requires a valid active flight_delay run_id.")
        if not (project_path(root, RUNS_DIR / active_run_id) / "_SUCCESS").is_file():
            raise ValueError(f"Legacy model cleanup requires completed run {active_run_id}.")

    removed: list[Path] = []
    for name in LEGACY_MODEL_DIRS:
        candidate = models_root / name
        if candidate.exists() or candidate.is_symlink():
            _safe_prune_entry(candidate, models_root)
            removed.append(candidate)
    return removed


def _entry_size_bytes(path: Path) -> int:
    """Return total regular-file bytes for a file or directory without following symlinks."""
    if path.is_symlink():
        return 0
    if not path.is_dir():
        return path.stat().st_size
    return dir_size_bytes(path)


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
    marker = run_dir / "_SUCCESS"
    try:
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

        marker.touch(exist_ok=False)
        update_results({RESULT_KEY: {
            "run_id": run_id,
            "run_sha256": sha256(manifest),
            "training": summary,
            "validation": validation,
            "coverage": coverage,
        }}, root / "results")
    except BaseException:
        # Remove aborted run artifacts so only complete runs remain in models/flight_delay/.
        marker.unlink(missing_ok=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    # Measure obsolete bytes before pruning so one concise summary line can report space freed.
    runs_root = project_path(root, RUNS_DIR)
    models_root = project_path(root, "models")
    obsolete_bytes = sum(
        _entry_size_bytes(path) for path in runs_root.iterdir() if path.name != run_id
    ) + sum(
        _entry_size_bytes(models_root / name)
        for name in LEGACY_MODEL_DIRS
        if (models_root / name).exists()
    )
    removed = [*prune_model_runs(root, run_id), *prune_legacy_models(root, require_active_run=True)]
    if removed:
        noun = "directory" if len(removed) == 1 else "directories"
        size_text = f"{obsolete_bytes / (1024 ** 3):.2f} GB" if obsolete_bytes >= 1024 ** 3 else f"{obsolete_bytes / (1024 ** 2):.1f} MB"
        print(f"  Pruned {len(removed)} obsolete model {noun} ({size_text} reclaimed).", flush=True)

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
