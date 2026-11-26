"""Extract, map, and export feature weights and importances for RQ3."""

import csv
import json
from pathlib import Path
from typing import Any, Optional, Union

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.sql import Row, SparkSession

from scripts.aircraft_rotation.build_features import RECOVERY_FEATURES, ROTATION_FLAGS, ROTATION_NUMERIC
from scripts.bts.block_time import SLACK_FEATURES
from scripts.bts.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from scripts.bts.recent_delays import HOLIDAY_FEATURES, ROLLING_FEATURES
from scripts.common.config import ROOT, project_path
from scripts.common.context import PipelineContext
from scripts.common.io import write_json
from scripts.common.progress import stage
from scripts.destination_weather.build_features import DESTINATION_WEATHER_FLAGS, DESTINATION_WEATHER_NUMERIC
from scripts.enroute_weather.build_features import FEATURES as ENROUTE_FEATURES
from scripts.origin_weather.build_features import WEATHER_FLAGS, WEATHER_NUMERIC

RESULT_KEY = "flight_delay"
RUNS_DIR = Path("models") / RESULT_KEY
TOP_N_DEFAULT = 25

MODEL_DIRS: dict[str, tuple[str, str]] = {
    "logistic_regression": ("", "logistic"),
    "class_weighted_logistic_regression": ("weighted_logistic", "logistic"),
    "random_forest": ("random_forest", "random_forest"),
}

FEATURE_GROUP_MAP: dict[str, str] = {
    **{name: "bts_schedule" for name in (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, "MonthSin", "MonthCos", *HOLIDAY_FEATURES, *ROLLING_FEATURES)},
    **{name: "surface_weather" for name in (*WEATHER_NUMERIC, *WEATHER_FLAGS, *DESTINATION_WEATHER_NUMERIC, *DESTINATION_WEATHER_FLAGS)},
    **{name: "aircraft_rotation_slack" for name in (*ROTATION_NUMERIC, *ROTATION_FLAGS, *RECOVERY_FEATURES, *SLACK_FEATURES)},
    **{name: "enroute_weather" for name in ENROUTE_FEATURES},
}


def _strip_suffix(name: str) -> str:
    """Strip pipeline transformation suffixes to recover the raw feature name."""
    for suffix in ("_imputed", "_filled", "_idx"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _resolve_attribute(attr_name: str) -> tuple[str, str, str]:
    """Map a VectorAssembler attribute name to (display_name, base_column, feature_group)."""
    for cat in CATEGORICAL_FEATURES:
        prefix = f"{cat}_vec_"
        if attr_name.startswith(prefix):
            category_val = attr_name[len(prefix):]
            return f"{cat}={category_val}", cat, FEATURE_GROUP_MAP.get(cat, "unknown")

    base_col = _strip_suffix(attr_name)
    group = FEATURE_GROUP_MAP.get(base_col, "unknown")
    return base_col, base_col, group


def _dummy_row_for_pipeline(pipeline_model: PipelineModel) -> Row:
    """Construct a minimal 1-row input matching the pipeline's StringIndexer and numeric columns."""
    values: dict[str, Any] = {}
    for cat in CATEGORICAL_FEATURES:
        values[cat] = "UNK"
    for col_name in (
        "Month",
        *NUMERIC_FEATURES,
        *ROLLING_FEATURES,
        *HOLIDAY_FEATURES,
        *WEATHER_NUMERIC,
        *WEATHER_FLAGS,
        *DESTINATION_WEATHER_NUMERIC,
        *DESTINATION_WEATHER_FLAGS,
        *ROTATION_NUMERIC,
        *ROTATION_FLAGS,
        *ENROUTE_FEATURES,
        *SLACK_FEATURES,
        *RECOVERY_FEATURES,
    ):
        values[col_name] = 0.0
    return Row(**values)


def extract_feature_metadata(
    pipeline_model: PipelineModel,
    spark: SparkSession,
) -> list[dict[str, Any]]:
    """Inspect VectorAssembler metadata to map each vector index to its feature name and group."""
    preprocessor = PipelineModel(stages=list(pipeline_model.stages[:-1]))
    dummy_df = spark.createDataFrame([_dummy_row_for_pipeline(pipeline_model)])
    transformed = preprocessor.transform(dummy_df)
    attrs = transformed.schema["features"].metadata.get("ml_attr", {}).get("attrs", {})

    entries: list[dict[str, Any]] = []
    for attr_type in ("numeric", "binary", "nominal"):
        for item in attrs.get(attr_type, []):
            idx = int(item["idx"])
            raw_name = str(item.get("name", f"feature_{idx}"))
            display_name, base_col, group = _resolve_attribute(raw_name)
            entries.append({
                "index": idx,
                "feature_name": display_name,
                "base_column": base_col,
                "feature_group": group,
                "attr_type": attr_type,
            })

    entries.sort(key=lambda item: item["index"])
    return entries


def _aggregate_groups(rows: list[dict[str, Any]], metric_key: str, *, use_abs: bool = False) -> list[dict[str, Any]]:
    """Aggregate per-feature weights or importances by feature group."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        group = row["feature_group"]
        val = abs(float(row[metric_key])) if use_abs else float(row[metric_key])
        totals[group] = totals.get(group, 0.0) + val
        counts[group] = counts.get(group, 0) + 1

    grand_total = sum(totals.values()) or 1.0
    aggregated = [
        {
            "feature_group": group,
            "total": total,
            "share": total / grand_total,
            "feature_count": counts[group],
        }
        for group, total in totals.items()
    ]
    aggregated.sort(key=lambda item: item["total"], reverse=True)
    return aggregated


def extract_logistic_coefficients(
    pipeline_model: PipelineModel,
    feature_meta: list[dict[str, Any]],
    top_n: int = TOP_N_DEFAULT,
) -> dict[str, Any]:
    """Extract standardized logistic regression coefficients and group totals."""
    classifier = pipeline_model.stages[-1]
    if not isinstance(classifier, LogisticRegressionModel):
        raise TypeError(f"Expected LogisticRegressionModel, got {type(classifier).__name__}")

    coeffs = classifier.coefficients.toArray().tolist()
    if len(coeffs) != len(feature_meta):
        raise ValueError(f"Coefficient length {len(coeffs)} != metadata length {len(feature_meta)}")

    rows = [
        {
            **meta,
            "coefficient": float(coef),
            "abs_coefficient": abs(float(coef)),
        }
        for meta, coef in zip(feature_meta, coeffs)
    ]

    top_positive = sorted((r for r in rows if r["coefficient"] > 0), key=lambda r: r["coefficient"], reverse=True)[:top_n]
    top_negative = sorted((r for r in rows if r["coefficient"] < 0), key=lambda r: r["coefficient"])[:top_n]
    top_abs = sorted(rows, key=lambda r: r["abs_coefficient"], reverse=True)[:top_n]
    group_totals = _aggregate_groups(rows, "coefficient", use_abs=True)

    return {
        "intercept": float(classifier.intercept),
        "num_features": len(rows),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "top_abs": top_abs,
        "group_totals": group_totals,
        "all_features": sorted(rows, key=lambda r: r["abs_coefficient"], reverse=True),
    }


def extract_rf_importances(
    pipeline_model: PipelineModel,
    feature_meta: list[dict[str, Any]],
    top_n: int = TOP_N_DEFAULT,
) -> dict[str, Any]:
    """Extract Random Forest feature importances and group totals."""
    classifier = pipeline_model.stages[-1]
    if not isinstance(classifier, RandomForestClassificationModel):
        raise TypeError(f"Expected RandomForestClassificationModel, got {type(classifier).__name__}")

    importances = classifier.featureImportances.toArray().tolist()
    if len(importances) != len(feature_meta):
        raise ValueError(f"Importance length {len(importances)} != metadata length {len(feature_meta)}")

    rows = [
        {
            **meta,
            "importance": float(imp),
        }
        for meta, imp in zip(feature_meta, importances)
    ]
    ranked = sorted(rows, key=lambda r: r["importance"], reverse=True)
    group_totals = _aggregate_groups(rows, "importance", use_abs=False)

    return {
        "num_features": len(rows),
        "top_features": ranked[:top_n],
        "group_totals": group_totals,
        "all_features": ranked,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a list of dicts to CSV with deterministic column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_active_run_id(root: Path, run_id: Optional[str]) -> str:
    """Resolve the active flight_delay run_id from argument or results/results.json."""
    if run_id:
        return run_id
    results_path = project_path(root, "results/results.json")
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing {results_path}; cannot resolve active run_id.")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    active = payload.get(RESULT_KEY, {}).get("run_id")
    if not isinstance(active, str) or not active:
        raise ValueError("No active flight_delay run_id found in results/results.json.")
    return active


def analyze_run(
    spark: SparkSession,
    root: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    results_dir: Optional[Union[str, Path]] = None,
    top_n: int = TOP_N_DEFAULT,
) -> Path:
    """Load saved PipelineModels for run_id, extract weights/importances, and export JSON/CSV."""
    base_root = Path(root or ROOT)
    active_run_id = _resolve_active_run_id(base_root, run_id)
    run_dir = project_path(base_root, RUNS_DIR / active_run_id)
    out_dir = Path(results_dir or project_path(base_root, "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": active_run_id,
        "models": {},
    }
    group_rows: list[dict[str, Any]] = []

    for model_name, (subdir, kind) in MODEL_DIRS.items():
        model_path = (run_dir / subdir / "model") if subdir else (run_dir / "model")
        if not model_path.exists():
            continue

        pipeline_model = PipelineModel.load(str(model_path))
        feature_meta = extract_feature_metadata(pipeline_model, spark)

        if kind == "logistic":
            extracted = extract_logistic_coefficients(pipeline_model, feature_meta, top_n=top_n)
            csv_fields = ["index", "feature_name", "base_column", "feature_group", "attr_type", "coefficient", "abs_coefficient"]
            _write_csv(out_dir / f"feature_importance_{model_name}.csv", extracted["all_features"], csv_fields)
            summary["models"][model_name] = {k: v for k, v in extracted.items() if k != "all_features"}
        else:
            extracted = extract_rf_importances(pipeline_model, feature_meta, top_n=top_n)
            csv_fields = ["index", "feature_name", "base_column", "feature_group", "attr_type", "importance"]
            _write_csv(out_dir / f"feature_importance_{model_name}.csv", extracted["all_features"], csv_fields)
            summary["models"][model_name] = {k: v for k, v in extracted.items() if k != "all_features"}

        for group_entry in extracted["group_totals"]:
            group_rows.append({"model": model_name, **group_entry})

    _write_csv(
        out_dir / "feature_importance_groups.csv",
        group_rows,
        ["model", "feature_group", "total", "share", "feature_count"],
    )
    out_json = out_dir / "feature_importance.json"
    write_json(out_json, summary)
    return out_json


def run(ctx: PipelineContext) -> Path:
    """Execute feature importance and coefficient analysis for the current pipeline run."""
    with stage(
        "Feature Importance & Coefficient Analysis (RQ3)",
        timings=ctx.timings,
        key="feature_importance",
    ):
        out_json = analyze_run(ctx.spark, root=ctx.root, run_id=ctx.run_id)
        print(f"  Feature importance report: {out_json}", flush=True)
        return out_json
