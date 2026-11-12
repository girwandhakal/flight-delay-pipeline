"""Run the BTS workflow in order, stopping when a stage fails."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import platform
import re
from uuid import uuid4
from pathlib import Path
import sys
import time

# Find the project scripts even when this file is run from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.results import update_results
from scripts.bts import clean as bts_clean, download as bts_download
from scripts.bts import (
    evaluate as bts_evaluate,
    train as bts_train,
    build_features as bts_transform,
)
from scripts.origin_weather import build_features as weather_features
from scripts.origin_weather.download import ROOT, load_config, sha256, write_json
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F


def compare_predictions(spark, result_dir, split="validation", context=None):
    frames = {}
    for variant in ("bts", "bts_origin_weather"):
        frame = spark.read.parquet(str(result_dir / variant / f"{split}_predictions"))
        count = frame.count()
        if (
            frame.select("flight_id").distinct().count() != count
            or frame.filter("flight_id is null").limit(1).count()
        ):
            raise ValueError("Prediction IDs must be unique and nonnull.")
        frames[variant] = frame
    baseline, weather = frames.values()
    if (
        baseline.select("flight_id", "FlightDate", "ArrDel15")
        .exceptAll(weather.select("flight_id", "FlightDate", "ArrDel15"))
        .limit(1)
        .count()
        or weather.select("flight_id", "FlightDate", "ArrDel15")
        .exceptAll(baseline.select("flight_id", "FlightDate", "ArrDel15"))
        .limit(1)
        .count()
    ):
        raise ValueError("Prediction cohorts differ.")
    paired = baseline.select(
        "flight_id",
        "FlightDate",
        "ArrDel15",
        *[
            F.col(name).alias("bts_" + name)
            for name in ("p_delayed", "prediction", "threshold")
        ],
    ).join(
        weather.select(
            "flight_id",
            *[
                F.col(name).alias("origin_" + name)
                for name in ("p_delayed", "prediction", "threshold")
            ],
        ),
        "flight_id",
    )
    paired.write.parquet(str(result_dir / f"{split}_predictions"))
    rows = []
    for variant, frame in frames.items():
        threshold = frame.select("threshold").first().threshold
        for operating_point, value in (("selected", threshold), ("default_0_5", 0.5)):
            metrics = bts_train.evaluate_predictions(
                frame.withColumn(
                    "prediction", (F.col("p_delayed") >= value).cast("int")
                )
            )
            rows.append(
                {
                    "variant": variant,
                    "operating_point": operating_point,
                    "threshold": value,
                    "rows": frame.count(),
                    **{
                        key: val
                        for key, val in metrics.items()
                        if key != "confusion_matrix"
                    },
                    **metrics["confusion_matrix"],
                }
            )
    import pandas as pd

    table = pd.DataFrame(rows)
    slices, covered = [], []
    if context is not None:
        context = context.select(
            "flight_id",
            "Origin",
            "Month",
            "origin_weather_missing",
            F.floor(F.col("origin_wind_speed_ms") / 5).alias("origin_wind_bin"),
        )
        slices, covered = [], []
        for variant, frame in frames.items():
            scores = frame.join(context, "flight_id")
            for column in (
                "Origin",
                "Month",
                "origin_weather_missing",
                "origin_wind_bin",
            ):
                slices.extend(
                    {"variant": variant, **row}
                    for row in bts_evaluate.compute_subgroup_slice(
                        scores, column, column
                    )
                )
            subset = scores.filter("origin_weather_missing = 0")
            if subset.limit(1).count():
                covered.append(
                    {
                        "variant": variant,
                        "rows": subset.count(),
                        **bts_train.evaluate_predictions(subset),
                    }
                )
    selected = table[table.operating_point == "selected"].set_index("variant")
    deltas = {
        key: float(selected.loc["bts_origin_weather", key] - selected.loc["bts", key])
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy")
    }
    results = {}
    for variant in frames:
        results[variant] = {split: {
            row["operating_point"]: {key: value for key, value in row.items()
                                     if key not in ("variant", "operating_point")}
            for row in rows if row["variant"] == variant
        }}
        results[variant][split]["subgroups"] = [row for row in slices if row["variant"] == variant]
        results[variant][split]["covered_only"] = [row for row in covered if row["variant"] == variant]
    results["bts_origin_weather"][split]["deltas_vs_bts"] = deltas
    return results


def run_origin_weather():
    start = time.monotonic()
    if not all(
        (ROOT / "data/gold/bts" / split / "_SUCCESS").exists()
        for split in ("train", "val", "test")
    ):
        for step, title, stage in STAGES[:3]:
            if run_stage(step, title, stage):
                return 1
    gold, data_report = weather_features.main()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    )
    models = ROOT / "models/origin_weather_comparison" / run_id
    results = models
    models.mkdir(parents=True)
    spark = bts_transform.get_spark_session("origin_weather_comparison")
    try:
        timings, summaries = {}, {}
        for variant, data in (
            ("bts", ROOT / "data/gold/bts"),
            ("bts_origin_weather", gold),
        ):
            started = time.monotonic()
            summaries[variant] = bts_train.train(
                spark,
                data,
                models / variant,
                feature_set=variant,
                save_predictions=True,
                publish_results=False,
            )
            timings[variant] = time.monotonic() - started
        updates = compare_predictions(
            spark, results, context=spark.read.parquet(str(gold / "val"))
        )
        _, policy, _ = load_config()
        write_json(
            results / "run.json",
            {
                "run_id": run_id,
                "gold_dir": str(gold),
                "bts_gold_dir": str(ROOT / "data/gold/bts"),
                "models_dir": str(models),
                "data_report": str(data_report),
                "dataset_manifest_sha256": sha256(
                    data_report / "_metadata.json"
                ),
                "policy": policy,
                "python": platform.python_version(),
                "spark": spark.version,
                "code_sha256": {
                    str(path.relative_to(ROOT)): sha256(path)
                    for folder in (ROOT / "scripts", ROOT / "pipeline")
                    for path in folder.rglob("*.py")
                },
                "elapsed_seconds": time.monotonic() - start,
                "training_seconds": timings,
                "bytes": sum(
                    p.stat().st_size
                    for folder in (models, gold)
                    for p in folder.rglob("*")
                    if p.is_file()
                ),
                "holdout_status": "Potentially inspected by prior BTS pipeline; no test metrics used in this comparison.",
            },
        )
        for variant, summary in summaries.items():
            updates[variant].update({"training": summary, "run_id": run_id})
        update_results(updates)
        (results / "_SUCCESS").touch()
        print(f"Origin-weather comparison completed: {results}")
        return 0
    finally:
        spark.stop()


def evaluate_origin_weather(run_id):
    from scripts.origin_weather.download import project_path

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Use the comparison directory name as RUN_ID.")
    results = project_path(ROOT, "models/origin_weather_comparison/" + run_id)
    if not (results / "_SUCCESS").exists():
        raise ValueError("Only a completed, frozen comparison can be test-evaluated.")
    current = json.loads((ROOT / "results/results.json").read_text())
    if any(current.get(variant, {}).get("run_id") != run_id
           for variant in ("bts", "bts_origin_weather")):
        raise ValueError("Test evaluation requires the current comparison run; results have been replaced.")
    run = json.loads((results / "run.json").read_text())
    output = results / "test"
    output.mkdir()
    spark = bts_transform.get_spark_session("origin_weather_test")
    try:
        for variant, data in (
            ("bts", run["bts_gold_dir"]),
            ("bts_origin_weather", run["gold_dir"]),
        ):
            model_dir = Path(run["models_dir"]) / variant
            threshold = json.loads((model_dir / "threshold.json").read_text())[
                "threshold"
            ]
            model = PipelineModel.load(str(model_dir / "baseline_pipeline"))
            scores = bts_train.predict(
                model, spark.read.parquet(str(Path(data) / "test")), threshold
            )
            scores.select(
                "flight_id", "FlightDate", "ArrDel15", "p_delayed", "prediction"
            ).withColumn("threshold", F.lit(threshold)).write.parquet(
                str(output / variant / "test_predictions")
            )
        updates = compare_predictions(
            spark,
            output,
            "test",
            context=spark.read.parquet(str(Path(run["gold_dir"]) / "test")),
        )
        update_results(updates)
        (output / "_SUCCESS").touch()
        return 0
    finally:
        spark.stop()


STAGES = [
    ("Stage 1/5", "BTS Data Acquisition (Download)", bts_download.main),
    ("Stage 2/5", "BTS Data Cleaning", bts_clean.main),
    ("Stage 3/5", "Feature Engineering & Chronological Splitting", bts_transform.main),
    ("Stage 4/5", "Baseline Model Training & Validation", bts_train.main),
    ("Stage 5/5", "Baseline Model Evaluation", bts_evaluate.main),
]


def run_stage(step: str, title: str, func) -> int:
    bar = "=" * 80
    subbar = "-" * 80
    print(f"\n{bar}")
    print(f"[{step}] START: {title}")
    print(f"{bar}\n", flush=True)

    start_time = time.time()
    try:
        ret = func()
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"\n{'!' * 80}")
        print(
            f"[{step}] ERROR: {title} encountered an unhandled exception after {elapsed:.1f}s:"
        )
        print(f"  {type(exc).__name__}: {exc}")
        print(f"{'!' * 80}\n", flush=True)
        raise

    elapsed = time.time() - start_time
    if ret != 0:
        print(f"\n{'!' * 80}")
        print(f"[{step}] FAILED: {title} (exit code: {ret}) after {elapsed:.1f}s")
        print(f"{'!' * 80}\n", flush=True)
        return ret

    print(f"\n{subbar}")
    print(f"[{step}] STOP: {title} finished successfully in {elapsed:.1f}s")
    print(f"{bar}\n", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--origin-weather",
        action="store_true",
        help="Build origin features and compare both models on validation.",
    )
    mode.add_argument(
        "--evaluate-origin-weather",
        metavar="RUN_ID",
        help="Evaluate both frozen models on test; never tunes settings.",
    )
    args = parser.parse_args(argv)
    if args.origin_weather:
        return run_origin_weather()
    if args.evaluate_origin_weather:
        return evaluate_origin_weather(args.evaluate_origin_weather)
    total_start = time.time()
    banner = "=" * 80
    print(f"\n{banner}")
    print("STARTING COMPLETE BTS END-TO-END WORKFLOW PIPELINE")
    print(f"{banner}")

    for step, title, func in STAGES:
        if run_stage(step, title, func) != 0:
            print(f"\n{banner}")
            print(f"PIPELINE STOPPED DUE TO FAILURE IN: {step} ({title})")
            print(f"{banner}\n")
            return 1

    total_elapsed = time.time() - total_start
    print(banner)
    print(
        f"ALL 5 PIPELINE STAGES COMPLETED SUCCESSFULLY in {str(timedelta(seconds=int(total_elapsed)))}"
    )
    print(f"{banner}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
