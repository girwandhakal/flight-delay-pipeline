"""Generate publication-ready figures from published results/ artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "class_weighted_logistic_regression": "Weighted LR",
    "random_forest": "Random Forest",
    "multilayer_perceptron": "MLP",
}

ABLATION_LABELS = {
    "bts_only": "BTS Only",
    "bts_surface_weather": "+ Surface Weather",
    "bts_surface_rotation": "+ Rotation & Slack",
    "full": "Full",
}

GROUP_LABELS = {
    "bts_schedule": "BTS & Schedule",
    "surface_weather": "Surface Weather",
    "aircraft_rotation_slack": "Rotation & Slack",
    "enroute_weather": "En-Route Weather",
}


def _apply_style() -> None:
    """Apply clean, monochrome publication-grade Matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
        }
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists, returning None otherwise."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]] | None:
    """Load CSV rows as dicts if the file exists, returning None otherwise."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_fig(fig: plt.Figure, out_path: Path) -> Path:
    """Save figure tightly and close it to free memory."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_worker_count(worker_spec: str) -> str:
    """Extract numeric worker count from strings like 'local[4]'."""
    match = re.search(r"\[(\d+)\]", worker_spec)
    return match.group(1) if match else worker_spec


def plot_model_comparison(results: dict[str, Any] | None, output_dir: Path) -> Path | None:
    """Plot Validation vs. Test PR-AUC, ROC-AUC, and F1 across all trained models."""
    if not results:
        return None
    fd = results.get("flight_delay", {})
    val_algos = fd.get("training", {}).get("algorithm_comparison", {})
    test_algos = fd.get("test", {}).get("models", {})
    model_keys = [k for k in MODEL_LABELS if k in val_algos and k in test_algos]
    if not model_keys:
        return None

    labels = [MODEL_LABELS[k] for k in model_keys]
    metrics_spec = [
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("f1", "F1 Score"),
    ]

    first_test = test_algos[model_keys[0]]
    baselines = first_test.get("reference_baselines", {})
    always_ontime_pr = baselines.get("always_ontime", {}).get("pr_auc")
    always_delayed_f1 = baselines.get("always_delayed", {}).get("f1")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    x = np.arange(len(model_keys))
    width = 0.36

    for ax, (metric_key, metric_title) in zip(axes, metrics_spec):
        val_vals = [
            float(val_algos[k].get("validation_metrics_at_selected_threshold", {}).get(metric_key, 0.0))
            for k in model_keys
        ]
        test_vals = [
            float(test_algos[k].get("metrics_at_selected_threshold", {}).get(metric_key, 0.0))
            for k in model_keys
        ]
        b1 = ax.bar(x - width / 2, val_vals, width, label="Validation", color="#D9D9D9", edgecolor="#222222", linewidth=0.8)
        b2 = ax.bar(x + width / 2, test_vals, width, label="Test", color="#333333", edgecolor="#111111", linewidth=0.8)

        if metric_key == "pr_auc" and always_ontime_pr is not None:
            ax.axhline(always_ontime_pr, color="#222222", linestyle=":", linewidth=1.2, label="Prevalence")
        elif metric_key == "roc_auc":
            ax.axhline(0.5, color="#222222", linestyle=":", linewidth=1.2, label="Random")
        elif metric_key == "f1" and always_delayed_f1 is not None:
            ax.axhline(always_delayed_f1, color="#222222", linestyle=":", linewidth=1.2, label="Always Delayed")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle("Model Performance", fontsize=13, fontweight="bold", y=1.03)
    return _save_fig(fig, output_dir / "fig1_model_comparison.png")


def plot_ablation_progression(results: dict[str, Any] | None, output_dir: Path) -> Path | None:
    """Plot controlled RQ2 feature-group ablation progression (PR-AUC, ROC-AUC, F1)."""
    if not results:
        return None
    groups = results.get("ablation", {}).get("feature_groups", {})
    ordered = [g for g in ABLATION_LABELS if g in groups]
    if not ordered:
        return None

    x = np.arange(len(ordered))
    labels = [ABLATION_LABELS[g] for g in ordered]

    def _series(split_key: str, metric: str) -> list[float]:
        vals = []
        for g in ordered:
            algo_dict = next(iter(groups[g].get("algorithms", {}).values()), {})
            vals.append(float(algo_dict.get(split_key, {}).get(metric, 0.0)))
        return vals

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    test_pr = _series("test_metrics_at_selected_threshold", "pr_auc")
    test_roc = _series("test_metrics_at_selected_threshold", "roc_auc")
    test_f1 = _series("test_metrics_at_selected_threshold", "f1")
    val_pr = _series("validation_metrics_at_selected_threshold", "pr_auc")

    ax.plot(x, test_roc, marker="s", linewidth=2.0, linestyle="-", color="#111111", label="Test ROC-AUC")
    ax.plot(x, test_pr, marker="o", linewidth=2.2, linestyle="-", color="#444444", label="Test PR-AUC")
    ax.plot(x, test_f1, marker="^", linewidth=2.0, linestyle="-.", color="#666666", label="Test F1")
    ax.plot(x, val_pr, marker="o", linewidth=1.6, linestyle="--", color="#888888", fillstyle="none", label="Validation PR-AUC")

    for idx in range(len(ordered)):
        ax.annotate(f"{test_pr[idx]:.3f}", (x[idx], test_pr[idx]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")
        ax.annotate(f"{test_f1[idx]:.3f}", (x[idx], test_f1[idx]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5)
        ax.annotate(f"{test_roc[idx]:.3f}", (x[idx], test_roc[idx]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.20, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Feature Ablation")
    ax.legend(loc="lower right", frameon=True)
    return _save_fig(fig, output_dir / "fig2_ablation_progression.png")


def plot_feature_importance(results_dir: Path, output_dir: Path) -> Path | None:
    """Plot RQ3 feature-group shares (LR vs. RF) and top 10 Random Forest features."""
    group_rows = _load_csv_rows(results_dir / "feature_importance_groups.csv")
    rf_rows = _load_csv_rows(results_dir / "feature_importance_random_forest.csv")
    if not group_rows or not rf_rows:
        return None

    shares: dict[str, dict[str, float]] = {}
    for row in group_rows:
        shares.setdefault(row["model"], {})[row["feature_group"]] = float(row["share"]) * 100.0

    if "logistic_regression" not in shares or "random_forest" not in shares:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"width_ratios": [1.0, 1.35]})

    # Left: group share comparison.
    group_keys = list(GROUP_LABELS.keys())
    y = np.arange(len(group_keys))
    h = 0.36
    lr_vals = [shares["logistic_regression"].get(g, 0.0) for g in group_keys]
    rf_vals = [shares["random_forest"].get(g, 0.0) for g in group_keys]

    ax1.barh(y + h / 2, lr_vals, h, label="Logistic Regression", color="#D9D9D9", edgecolor="#222222", linewidth=0.8)
    ax1.barh(y - h / 2, rf_vals, h, label="Random Forest", color="#333333", edgecolor="#111111", linewidth=0.8)
    for idx in range(len(group_keys)):
        ax1.text(lr_vals[idx] + 1.2, y[idx] + h / 2, f"{lr_vals[idx]:.1f}%", va="center", fontsize=8.5)
        ax1.text(rf_vals[idx] + 1.2, y[idx] - h / 2, f"{rf_vals[idx]:.1f}%", va="center", fontsize=8.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([GROUP_LABELS[g] for g in group_keys])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Importance Share, %")
    ax1.set_title("Feature Group Importance")
    ax1.legend(loc="lower right", frameon=True, fontsize=8.5)

    # Right: top 10 Random Forest features.
    top_rf = sorted(rf_rows, key=lambda r: float(r["importance"]), reverse=True)[:10]
    top_rf.reverse()
    rf_names = [r["feature_name"] for r in top_rf]
    rf_imps = [float(r["importance"]) for r in top_rf]

    ax2.barh(np.arange(len(rf_names)), rf_imps, color="#4D4D4D", edgecolor="#111111", linewidth=0.7)
    for idx, val in enumerate(rf_imps):
        ax2.text(val + 0.003, idx, f"{val:.3f}", va="center", fontsize=8)
    ax2.set_yticks(np.arange(len(rf_names)))
    ax2.set_yticklabels(rf_names, fontsize=8.5)
    ax2.set_xlim(0, max(rf_imps) * 1.18 if rf_imps else 0.2)
    ax2.set_xlabel("Gini Importance")
    ax2.set_title("Top 10 Random Forest Features")

    return _save_fig(fig, output_dir / "fig3_feature_importance.png")


def plot_temporal_hour_slice(results: dict[str, Any] | None, output_dir: Path) -> Path | None:
    """Plot frozen test F1 and delay prevalence by scheduled departure hour."""
    if not results:
        return None
    lr_subgroups = results.get("flight_delay", {}).get("test", {}).get("models", {}).get("logistic_regression", {}).get("subgroups", [])
    hour_rows = [r for r in lr_subgroups if r.get("subgroup_type") == "departure_hour"]
    if not hour_rows:
        return None

    hour_rows = sorted(
        [r for r in hour_rows if int(r["subgroup_value"]) >= 5],
        key=lambda r: int(r["subgroup_value"]),
    )
    if not hour_rows:
        return None

    hours = [int(r["subgroup_value"]) for r in hour_rows]
    f1_vals = [float(r["f1"]) for r in hour_rows]
    prev_vals = [float(r["prevalence"]) for r in hour_rows]
    counts = [int(r["total_flights"]) / 1000.0 for r in hour_rows]

    fig, ax1 = plt.subplots(figsize=(9.5, 4.4))
    ax1.bar(hours, counts, color="#E5E5E5", edgecolor="#999999", linewidth=0.7, label="Test Flights", zorder=1)
    ax1.set_xlabel("Scheduled Departure Hour")
    ax1.set_ylabel("Flights in Thousands")
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{h:02d}" for h in hours])

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(hours, f1_vals, marker="o", linewidth=2.2, linestyle="-", color="#111111", label="Logistic Regression F1", zorder=3)
    ax2.plot(hours, prev_vals, marker="s", linewidth=1.8, linestyle="--", color="#555555", label="Delay Prevalence", zorder=2)
    ax2.set_ylabel("Score / Prevalence")
    ax2.set_ylim(0.0, 0.78)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    ax1.set_title("Test Performance by Departure Hour")
    return _save_fig(fig, output_dir / "fig4_temporal_hour_slice.png")


def plot_systems_scaling(benchmark: dict[str, Any] | None, output_dir: Path) -> Path | None:
    """Plot Phase 2 Spark data-size scaling and worker scaling benchmarks."""
    if not benchmark:
        return None
    size_rows = benchmark.get("data_size_scaling", [])
    worker_rows = benchmark.get("worker_scaling", [])
    if not size_rows or not worker_rows:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # Left: execution time vs data size.
    fracs = [float(r["data_fraction"]) * 100.0 for r in size_rows]
    wall_s = [float(r["wall_time_seconds"]) for r in size_rows]
    mem_gb = [float(r["peak_memory_gb"]) for r in size_rows]

    ax1.plot(fracs, wall_s, marker="o", linewidth=2.2, linestyle="-", color="#111111", label="Wall Time")
    for x_val, y_val in zip(fracs, wall_s):
        ax1.annotate(f"{y_val:.1f}s", (x_val, y_val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax1.set_xlabel("Data Fraction")
    ax1.set_ylabel("Wall Time, s")
    ax1.set_xticks(fracs)
    ax1.set_xticklabels([f"{int(f)}%" for f in fracs])
    ax1.set_ylim(0, max(wall_s) * 1.22)

    ax1_mem = ax1.twinx()
    ax1_mem.spines["top"].set_visible(False)
    ax1_mem.plot(fracs, mem_gb, marker="^", linewidth=1.8, linestyle="--", color="#666666", label="Peak Memory")
    ax1_mem.set_ylabel("Peak Memory, GB")
    ax1_mem.set_ylim(0, max(mem_gb) * 1.25)
    ax1.set_title("Execution Time vs. Data Size")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_mem.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=True, fontsize=8.5)

    # Right: workers speed up.
    workers = [_parse_worker_count(str(r["workers"])) for r in worker_rows]
    w_wall = [float(r["wall_time_seconds"]) for r in worker_rows]
    w_speedup = [float(r.get("speedup", 1.0)) for r in worker_rows]
    x_w = np.arange(len(workers))

    bars = ax2.bar(x_w, w_wall, width=0.48, color="#525252", edgecolor="#111111", linewidth=0.8)
    for bar, sp in zip(bars, w_speedup):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 8, f"{h:.1f}s\n{sp:.2f}×", ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x_w)
    ax2.set_xticklabels(workers)
    ax2.set_xlabel("Number of Spark Workers")
    ax2.set_ylabel("Wall Time, s")
    ax2.set_ylim(0, max(w_wall) * 1.22)
    ax2.set_title("Workers Speed Up")
    return _save_fig(fig, output_dir / "fig5_systems_scaling.png")


def generate_all_figures(results_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Generate all available publication figures and return paths of written PNGs."""
    _apply_style()
    figures_dir = output_dir or (results_dir / "figures")
    results = _load_json(results_dir / "results.json")
    benchmark = _load_json(results_dir / "benchmark_results.json")

    generators = [
        ("Model comparison", lambda: plot_model_comparison(results, figures_dir)),
        ("Ablation progression", lambda: plot_ablation_progression(results, figures_dir)),
        ("Feature importance", lambda: plot_feature_importance(results_dir, figures_dir)),
        ("Temporal hour slice", lambda: plot_temporal_hour_slice(results, figures_dir)),
        ("Systems scaling", lambda: plot_systems_scaling(benchmark, figures_dir)),
    ]

    written: list[Path] = []
    for name, fn in generators:
        out = fn()
        if out is not None:
            print(f"    [reporting] Wrote {out}", flush=True)
            written.append(out)
        else:
            print(f"    [reporting] Skipped {name} (missing source data).", flush=True)
    return written


def run(root: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    """Run the reporting stage against <root>/results."""
    base = root or PROJECT_ROOT
    return generate_all_figures(base / "results", output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone figure generation."""
    parser = argparse.ArgumentParser(description="Generate publication-ready figures from results/.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results", help="Directory containing results JSON/CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for PNG figures (defaults to <results-dir>/figures).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for standalone reporting."""
    args = parse_args(argv)
    written = generate_all_figures(args.results_dir, output_dir=args.output_dir)
    print(f"Generated {len(written)} figure(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
