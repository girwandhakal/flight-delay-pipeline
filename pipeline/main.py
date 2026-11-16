"""Run the BTS workflow or a paired weather experiment, stopping on failure."""

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
from scripts.progress import stage as progress_stage
from scripts.bts import clean as bts_clean, download as bts_download
from scripts.bts import (
    evaluate as bts_evaluate,
    train as bts_train,
    build_features as bts_transform,
)
from scripts.bts.recent_delays import features_current
from scripts.origin_weather import build_features as weather_features
from scripts.destination_weather import build_features as destination_features
from scripts.origin_weather.download import ROOT, load_config, sha256, write_json
from pyspark.ml import PipelineModel
from pyspark.errors import AnalysisException
from pyspark.sql import functions as F


def compare_predictions(spark, result_dir, split="validation", context=None,
                        variants=("bts", "bts_origin_weather"), training_prevalence=None):
    if tuple(variants) not in (("bts", "bts_origin_weather"), bts_train.FEATURE_SETS):
        raise ValueError("Use the two origin variants or all three destination variants.")
    frames = {}
    for variant in variants:
        frame = spark.read.parquet(str(result_dir / variant / f"{split}_predictions"))
        count = frame.count()
        if (
            frame.select("flight_id").distinct().count() != count
            or frame.filter("flight_id is null").limit(1).count()
        ):
            raise ValueError("Prediction IDs must be unique and nonnull.")
        frames[variant] = frame
    baseline = frames["bts"]
    identity = ["flight_id", "FlightDate", "ArrDel15"]
    for variant, frame in frames.items():
        if variant == "bts":
            continue
        if (baseline.select(identity).exceptAll(frame.select(identity)).limit(1).count()
                or frame.select(identity).exceptAll(baseline.select(identity)).limit(1).count()):
            raise ValueError("Prediction cohorts differ.")
    paired = baseline.select(*identity, *[F.col(name).alias("bts_" + name) for name in ("p_delayed", "prediction", "threshold")])
    for variant, frame in frames.items():
        if variant == "bts":
            continue
        prefix = {"bts": "bts", "bts_origin_weather": "origin",
                  "bts_origin_destination_weather": "destination"}[variant]
        paired = paired.join(frame.select("flight_id", *[
            F.col(name).alias(prefix + "_" + name)
            for name in ("p_delayed", "prediction", "threshold")
        ]), "flight_id")
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
        columns = ["Origin", "Month", "origin_weather_missing", "origin_wind_bin"]
        expressions = ["flight_id", "Origin", "Month", "origin_weather_missing",
                       F.floor(F.col("origin_wind_speed_ms") / 5).alias("origin_wind_bin")]
        if "bts_origin_destination_weather" in variants:
            columns += ["Dest", "destination_weather_missing", "destination_wind_bin"]
            expressions += ["Dest", "destination_weather_missing",
                            F.floor(F.col("destination_wind_speed_ms") / 5).alias("destination_wind_bin")]
        context = context.select(*expressions)
        slices, covered = [], []
        for variant, frame in frames.items():
            scores = frame.join(context, "flight_id").cache()
            try:
                for column in columns:
                    slices.extend(
                        {"variant": variant, **row}
                        for row in bts_evaluate.compute_subgroup_slice(
                            scores, column, column
                        )
                    )
                subset = scores.filter("origin_weather_missing = 0" + (" and destination_weather_missing = 0" if "bts_origin_destination_weather" in variants else ""))
                if subset.limit(1).count():
                    covered.append(
                        {
                            "variant": variant,
                            "rows": subset.count(),
                            **bts_train.evaluate_predictions(subset),
                        }
                    )
            finally:
                scores.unpersist()
    selected = table[table.operating_point == "selected"].set_index("variant")
    deltas = {
        key: (float(selected.loc["bts_origin_weather", key] - selected.loc["bts", key])
              if selected.loc[["bts_origin_weather", "bts"], key].notna().all() else None)
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy")
    }
    results = {}
    references = {}
    for variant in frames:
        results[variant] = {split: {
            row["operating_point"]: {key: value for key, value in row.items()
                                     if key not in ("variant", "operating_point")}
            for row in rows if row["variant"] == variant
        }}
        probabilities = [("always_ontime", 0.0), ("always_delayed", 1.0)]
        if training_prevalence is not None:
            probabilities.append(("train_prevalence", training_prevalence[variant]))
        results[variant][split]["reference_baselines"] = {}
        for name, probability in probabilities:
            if probability not in references:
                references[probability] = bts_train.evaluate_predictions(
                    baseline.select("ArrDel15").withColumn("p_delayed", F.lit(probability))
                    .withColumn("prediction", F.lit(int(probability >= 0.5))))
            metrics = dict(references[probability])
            if name == "train_prevalence":
                metrics["training_prevalence"] = probability
            results[variant][split]["reference_baselines"][name] = metrics
        results[variant][split]["subgroups"] = [row for row in slices if row["variant"] == variant]
        results[variant][split]["covered_only"] = [row for row in covered if row["variant"] == variant]
    results["bts_origin_weather"][split]["deltas_vs_bts"] = deltas
    if "bts_origin_destination_weather" in variants:
        for reference in ("bts", "bts_origin_weather"):
            results["bts_origin_destination_weather"][split]["deltas_vs_" + reference] = {
                key: (float(selected.loc["bts_origin_destination_weather", key] - selected.loc[reference, key])
                      if selected.loc[["bts_origin_destination_weather", reference], key].notna().all() else None)
                for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy")
            }
    return results


def inference_hashes(models):
    return {
        str(path.relative_to(models)): sha256(path)
        for variant in bts_train.FEATURE_SETS
        for path in sorted((models / variant).rglob("*"))
        if path.is_file() and ("baseline_pipeline" in path.parts or path.name == "threshold.json")
    }


def data_fingerprints(directory):
    return [{"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
             "modified_ns": path.stat().st_mtime_ns}
            for path in sorted(directory.rglob("*.parquet"))]


def run_destination_weather():
    start = time.monotonic()
    print("Starting pipeline: BTS + origin + destination weather (validation)", flush=True)
    settings, policy, config = load_config()
    print("Checking whether BTS features can be reused...", flush=True)
    if not features_current(ROOT, config):
        for step, title, stage in (
            ("Preparing BTS 1/3", "BTS Data Acquisition", bts_download.main),
            ("Preparing BTS 2/3", "BTS Data Cleaning", lambda: bts_clean.main(publish_results=False)),
            ("Preparing BTS 3/3", "BTS Features and Splits", lambda: bts_transform.main(publish_results=False)),
        ):
            status = run_stage(step, title, stage)
            if status:
                return status
    else:
        print("Reusing BTS features because they match the current inputs and policy.", flush=True)
    current_path = ROOT / "results/results.json"
    current = json.loads(current_path.read_text()) if current_path.exists() else {}
    with progress_stage("Stage: Prepare origin-weather features (validate cached data or rebuild)"):
        try:
            parent = Path(current["bts_origin_weather"]["data"]["gold_dir"])
            destination_features.validate_parent(parent, settings, policy, config)
            print(f"Checking cached origin-weather features: {parent}", flush=True)
            schema_spark = bts_transform.get_spark_session("destination_parent_schema")
            try:
                destination_features.validate_parent_data(schema_spark, parent, config)
            finally:
                schema_spark.stop()
            print(f"Reusing origin-weather features from {parent} because validation passed.", flush=True)
        except (KeyError, OSError, ValueError, AnalysisException) as error:
            print(f"Rebuilding origin-weather features because cached data could not be reused: {error}", flush=True)
            parent, _ = weather_features.main(publish_results=False)
    with progress_stage("Stage: Prepare destination-weather features"):
        gold, _ = destination_features.main(parent)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    models = ROOT / "models/destination_weather_comparison" / run_id
    models.mkdir(parents=True)
    data_dirs = dict(zip(bts_train.FEATURE_SETS, (ROOT / "data/gold/bts", parent, gold)))
    spark = bts_transform.get_spark_session("destination_weather_comparison")
    try:
        summaries, timings = {}, {}
        for variant, directory in data_dirs.items():
            started = time.monotonic()
            with progress_stage(f"Training {variant}: fixed settings and validation"):
                summaries[variant] = bts_train.train(spark, directory, models / variant,
                    feature_set=variant, save_predictions=True, publish_results=False)
            timings[variant] = time.monotonic() - started
        prevalence = {variant: summary["reference_baselines"]["train_prevalence"]["training_prevalence"] for variant, summary in summaries.items()}
        with progress_stage("Stage: Compare all three models on validation"):
            updates = compare_predictions(spark, models,
                context=spark.read.parquet(str(gold / "val")), variants=bts_train.FEATURE_SETS, training_prevalence=prevalence)
        with progress_stage("Stage: Save comparison manifest and publish results"):
            metadata = json.loads((gold / "_metadata.json").read_text())
            run = {
                "run_id": run_id, "models_dir": str(models),
                "data_dirs": {variant: str(path) for variant, path in data_dirs.items()},
                "splits": config["splits"], "policy": policy, "training_prevalence": prevalence,
                "feature_columns": {variant: summary["input_columns"] for variant, summary in summaries.items()},
                "thresholds": {variant: summary["selected_model"]["best_threshold"] for variant, summary in summaries.items()},
                "data_metadata_sha256": {variant: sha256(path / "_metadata.json") for variant, path in data_dirs.items() if variant != "bts"},
                "data_fingerprints": {variant: data_fingerprints(path) for variant, path in data_dirs.items()},
                "inference_sha256": inference_hashes(models),
                "source_identity": metadata,
                "code_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                for folder in (ROOT / "scripts", ROOT / "pipeline") for path in folder.rglob("*.py")},
                "python": platform.python_version(), "spark": spark.version,
                "training_seconds": timings, "elapsed_seconds": time.monotonic() - start,
                "bytes": sum(path.stat().st_size for folder in (models, gold) for path in folder.rglob("*") if path.is_file()),
                "holdout_status": "Potentially inspected development benchmark. No test metrics used in validation comparison.",
            }
            write_json(models / "run.json", run)
            for variant, summary in summaries.items():
                updates[variant].update(training=summary, run_id=run_id)
            updates["bts_origin_weather"]["data"] = json.loads((parent / "_metadata.json").read_text())
            updates["bts_origin_destination_weather"].update(data=metadata, cleaning={key: metadata[key] for key in ("quality", "station_quality", "source_contract")})
            updates["bts_origin_weather"]["cleaning"] = updates["bts_origin_destination_weather"]["cleaning"]
            update_results(updates)
            (models / "_SUCCESS").touch()
        print(f"Pipeline completed in {time.monotonic() - start:.1f}s. Results: {models}", flush=True)
        return 0
    finally:
        spark.stop()


def evaluate_destination_weather(run_id):
    from scripts.origin_weather.download import project_path

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Use the comparison directory name as RUN_ID.")
    models = project_path(ROOT, "models/destination_weather_comparison/" + run_id)
    if not (models / "_SUCCESS").exists():
        raise ValueError("Only a completed frozen comparison can be test-evaluated.")
    current = json.loads((ROOT / "results/results.json").read_text())
    if any(current.get(variant, {}).get("run_id") != run_id for variant in bts_train.FEATURE_SETS):
        raise ValueError("Test evaluation requires the current comparison run.")
    run = json.loads((models / "run.json").read_text())
    if run["run_id"] != run_id or Path(run["models_dir"]).resolve() != models:
        raise ValueError("Frozen run identity changed.")
    if inference_hashes(models) != run["inference_sha256"] or not run["inference_sha256"]:
        raise ValueError("Frozen model or threshold artifacts changed.")
    data_dirs = {}
    for variant in bts_train.FEATURE_SETS:
        path = Path(run["data_dirs"][variant]).resolve()
        if not path.is_relative_to((ROOT / "data/gold" / variant).resolve()):
            raise ValueError("Frozen data path escaped its feature-set directory.")
        if data_fingerprints(path) != run["data_fingerprints"][variant]:
            raise ValueError("Frozen Gold files changed.")
        if variant != "bts" and sha256(path / "_metadata.json") != run["data_metadata_sha256"][variant]:
            raise ValueError("Frozen data metadata changed.")
        data_dirs[variant] = path
    output = models / "test" / uuid4().hex
    spark = bts_transform.get_spark_session("destination_weather_test")
    try:
        for variant, data in data_dirs.items():
            directory = models / variant
            threshold = json.loads((directory / "threshold.json").read_text())["threshold"]
            if threshold != run["thresholds"][variant]:
                raise ValueError("Frozen threshold differs from run manifest.")
            model = PipelineModel.load(str(directory / "baseline_pipeline"))
            test = spark.read.parquet(str(data / "test"))
            if variant != "bts":
                bts_train.validate_weather(test, "frozen test inputs", variant, require_values=False)
            scores = bts_train.predict(model, test, threshold)
            scores.select("flight_id", "FlightDate", "ArrDel15", "p_delayed", "prediction").withColumn(
                "threshold", F.lit(threshold)).write.parquet(str(output / variant / "test_predictions"))
        updates = compare_predictions(spark, output, "test",
            context=spark.read.parquet(str(data_dirs["bts_origin_destination_weather"] / "test")), variants=bts_train.FEATURE_SETS, training_prevalence=run["training_prevalence"])
        update_results(updates)
        (output / "_SUCCESS").touch()
        return 0
    finally:
        spark.stop()


def run_origin_weather():
    start = time.monotonic()
    print("Starting pipeline: BTS + origin weather (validation)", flush=True)
    _, _, config = load_config()
    print("Checking whether BTS features can be reused...", flush=True)
    if not features_current(ROOT, config):
        for step, title, stage in STAGES[:3]:
            if run_stage(step, title, stage):
                return 1
    else:
        print("Reusing BTS features because they match the current inputs and policy.", flush=True)
    with progress_stage("Stage: Prepare origin-weather features"):
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
            with progress_stage(f"Training {variant}: fixed settings and validation"):
                summaries[variant] = bts_train.train(
                    spark,
                    data,
                    models / variant,
                    feature_set=variant,
                    save_predictions=True,
                    publish_results=False,
                )
            timings[variant] = time.monotonic() - started
        with progress_stage("Stage: Compare both models on validation"):
            updates = compare_predictions(
                spark, results, context=spark.read.parquet(str(gold / "val"))
            )
        with progress_stage("Stage: Save comparison manifest and publish results"):
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
        print(f"Pipeline completed in {time.monotonic() - start:.1f}s. Results: {results}", flush=True)
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
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] [{step}] START: {title}")
    print(f"{bar}\n", flush=True)

    start_time = time.monotonic()
    try:
        ret = func()
    except Exception as exc:
        elapsed = time.monotonic() - start_time
        print(f"\n{'!' * 80}")
        print(
            f"[{step}] ERROR: {title} encountered an unhandled exception after {elapsed:.1f}s:"
        )
        print(f"  {type(exc).__name__}: {exc}")
        print(f"{'!' * 80}\n", flush=True)
        raise

    elapsed = time.monotonic() - start_time
    if ret != 0:
        print(f"\n{'!' * 80}")
        print(f"[{step}] FAILED: {title} (exit code: {ret}) after {elapsed:.1f}s")
        print(f"{'!' * 80}\n", flush=True)
        return ret

    print(f"\n{subbar}")
    print(f"[{step}] DONE: {title} finished successfully in {elapsed:.1f}s")
    print(f"{bar}\n", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bts-only", action="store_true", help="Run the original BTS-only workflow, including test evaluation.")
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
    mode.add_argument("--destination-weather", action="store_true", help="Compare three models on validation only.")
    mode.add_argument("--evaluate-destination-weather", metavar="RUN_ID", help="Score three frozen models on test.")
    args = parser.parse_args(argv)
    if args.destination_weather:
        return run_destination_weather()
    if args.evaluate_destination_weather:
        return evaluate_destination_weather(args.evaluate_destination_weather)
    if args.origin_weather:
        return run_origin_weather()
    if args.evaluate_origin_weather:
        return evaluate_origin_weather(args.evaluate_origin_weather)
    if not args.bts_only:
        return run_destination_weather()
    total_start = time.monotonic()
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

    total_elapsed = time.monotonic() - total_start
    print(banner)
    print(
        f"ALL 5 PIPELINE STAGES COMPLETED SUCCESSFULLY in {str(timedelta(seconds=int(total_elapsed)))}"
    )
    print(f"{banner}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
