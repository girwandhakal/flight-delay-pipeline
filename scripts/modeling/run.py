"""Publish a completed training run, validation report, and human-readable summary."""

import json
from pathlib import Path
import re
import shutil
from typing import Any, Optional

from scripts.common.config import project_path
from scripts.common.context import PipelineContext
from scripts.common.io import dir_size_bytes, fs_path, rmtree_retry, sha256, write_json
from scripts.common.progress import DIVIDER, pipeline_stage
from scripts.common.results import update_results
from scripts.common.stage_cache import consumed, contract_digest

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


def training_cache_key(gold_id: dict[str, Any], splits: dict[str, Any]) -> str:
    """Compute a deterministic cache key for Phase 1 model training inputs, params, and code."""
    run_file = Path(__file__).resolve()
    train_file = run_file.parent / "train.py"
    return contract_digest({
        "version": 1,
        "gold_identity": gold_id,
        "splits": splits,
        "input_columns": list(train.INPUT_COLUMNS),
        "thresholds": list(train.THRESHOLDS),
        "logistic_params": train.LOGISTIC_PARAMS,
        "random_forest_params": train.RANDOM_FOREST_PARAMS,
        "mlp_params": train.MLP_PARAMS,
        "code": {
            "train.py": sha256(train_file),
            "run.py": sha256(run_file),
        },
    })


def _validate_cached_run_dir(
    run_dir: Path,
    runs_root: Path,
    gold_id: dict[str, Any],
    splits: dict[str, Any],
    expected_key: str,
) -> Optional[Path]:
    """Return validation.json if run_dir is a complete, verified model run matching inputs."""
    if run_dir.is_symlink() or not run_dir.is_dir():
        return None
    resolved_root = runs_root.resolve()
    resolved_dir = run_dir.resolve()
    if resolved_dir.parent != resolved_root or not resolved_dir.is_relative_to(resolved_root):
        return None

    marker = run_dir / "_SUCCESS"
    manifest_path = run_dir / "run.json"
    report_path = run_dir / "validation.json"
    if not (marker.is_file() and manifest_path.is_file() and report_path.is_file()):
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("run_id") != run_dir.name
            or manifest.get("splits") != splits
            or manifest.get("gold_identity") != gold_id
            or manifest.get("input_columns") != list(train.INPUT_COLUMNS)
        ):
            return None
        # Legacy runs created before model caching lack training_cache_key; allow bootstrap reuse via data/hash checks.
        if "training_cache_key" in manifest and manifest["training_cache_key"] != expected_key:
            return None

        variants = manifest.get("variants")
        if not isinstance(variants, dict) or not variants:
            return None
        for variant in variants.values():
            if not isinstance(variant, dict):
                return None
            actual_hashes = inference_hashes(run_dir, variant["model"], variant["threshold_file"])
            if actual_hashes != variant.get("sha256"):
                return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return report_path


def _find_cached_run(
    root: Path,
    gold_id: dict[str, Any],
    splits: dict[str, Any],
    expected_key: str,
) -> Optional[tuple[str, Path]]:
    """Locate a verified completed run matching gold_id and splits, preferring the active run."""
    runs_root = project_path(root, RUNS_DIR)
    if not runs_root.exists():
        return None

    candidates: list[Path] = []
    results_path = project_path(root, "results/results.json")
    if results_path.is_file():
        try:
            active_id = json.loads(results_path.read_text(encoding="utf-8")).get(RESULT_KEY, {}).get("run_id")
            if isinstance(active_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", active_id):
                candidates.append(runs_root / active_id)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    completed = sorted(runs_root.glob("*/_SUCCESS"), key=lambda p: p.stat().st_mtime, reverse=True)
    for marker in completed:
        if marker.parent not in candidates:
            candidates.append(marker.parent)

    for candidate in candidates:
        report = _validate_cached_run_dir(candidate, runs_root, gold_id, splits, expected_key)
        if report is not None:
            return candidate.name, report
    return None


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
    *, force_retrain: bool = False,
) -> Path:
    """Train once (or reuse a verified cached run), save artifacts, and publish validation results."""
    gold = gold.resolve()
    if gold.parent != project_path(root, GOLD_DIR):
        raise ValueError("Training Gold must be under the en-route Gold directory.")

    partition_names = ("train", "val", "test")
    identity = gold_identity(gold, partition_names)
    cache_key = training_cache_key(identity, splits)

    if not force_retrain:
        cached = _find_cached_run(root, identity, splits, cache_key)
        if cached is not None:
            cached_run_id, cached_report = cached
            print(f"  [Cache HIT] Reusing Phase 1 models (run_id: {cached_run_id}).", flush=True)
            cached_payload = json.loads(cached_report.read_text(encoding="utf-8"))
            print(
                format_classification_report(
                    cached_run_id, cached_payload.get("algorithm_comparison", {}), split="VALIDATION"
                ),
                flush=True,
            )
            return cached_report

    reason = "force_retrain" if force_retrain else "cache_miss"
    print(f"  [Cache MISS] Training Phase 1 models (reason: {reason}, run_id: {run_id}).", flush=True)

    run_dir = project_path(root, RUNS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    marker = run_dir / "_SUCCESS"
    try:
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
            "training_cache_key": cache_key,
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
    """Stage 6 entry point: fit models on full-feature Gold (or reuse cache) and publish validation."""
    with pipeline_stage(6, 6, "Model Training & Validation", timings=ctx.timings):
        report = train_and_publish(
            ctx.spark,
            ctx.root,
            ctx.run_id,
            ctx.gold_dir,
            ctx.config["splits"],
            ctx.coverage,
            ctx.sources,
            force_retrain=ctx.force_retrain,
        )
        # Align ctx.run_id with the resolved run directory so downstream stages target the cached run.
        ctx.run_id = report.parent.name
        return report
