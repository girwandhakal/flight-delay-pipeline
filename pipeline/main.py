"""Run the flight-delay data pipeline sequentially from BTS ingestion to validation."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

# Support direct execution with `python pipeline/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aircraft_rotation import run as aircraft_rotation
from scripts.bts import run as bts
from scripts.common.benchmark import run_benchmark_suite
from scripts.common.cleanup import cleanup_workspace, format_cleanup_summary
from scripts.common.context import PipelineContext
from scripts.common.progress import DIVIDER, stage
from scripts.common.results import record_stage_timings
from scripts.common.spark import spark_session
from scripts.destination_weather import run as destination_weather
from scripts.enroute_weather import run as enroute_weather
from scripts.modeling import ablation, evaluate, feature_importance, run as modeling
from scripts.origin_weather import run as origin_weather


def print_pipeline_summary(ctx: PipelineContext, report: Path, total_seconds: float) -> None:
    """Print a readable stage timing breakdown and final report path."""
    print(f"\n{DIVIDER}", flush=True)
    print(f"PIPELINE SUMMARY (Run ID: {ctx.run_id})", flush=True)
    print(DIVIDER, flush=True)
    for stage_name, elapsed in ctx.timings.items():
        print(f"  {stage_name:<50} {elapsed:>7.1f}s", flush=True)
    print(f"  {'Total pipeline elapsed':<50} {total_seconds:>7.1f}s", flush=True)
    print(f"  Validation report: {report}", flush=True)
    print(DIVIDER, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the flight-delay pipeline."""
    parser = argparse.ArgumentParser(description="Run the flight-delay pipeline or controlled ablation study.")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the controlled feature-set ablation study (RQ2) on the pinned Gold cohort.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the systems performance and scalability benchmark suite (Issue #4).",
    )
    return parser.parse_args(argv)


def _run_phase1(ctx: PipelineContext, *, run_ablation: bool = False) -> Path:
    """Run Phase 1 (data prep, training, RQ3 feature importance, test eval) and optional RQ2 ablation."""
    with spark_session("flight_delay") as spark:
        ctx.spark = spark
        bts.run(ctx)
        origin_weather.run(ctx)
        destination_weather.run(ctx)
        aircraft_rotation.run(ctx)
        enroute_weather.run(ctx)
        report = modeling.run(ctx)
        feature_importance.run(ctx)
        evaluate.run(ctx)
        if run_ablation:
            ablation.run_ablation(ctx)
    ctx.spark = None
    return report


def main(argv: list[str] | None = None) -> int:
    """Execute Phase 1 and optional Phase 2 (benchmarks) / Phase 3 (ablation) by CLI flags."""
    args = parse_args(argv)
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ctx = PipelineContext.create()

    modes = ["PHASE 1 (PIPELINE & EVALUATION)"]
    if args.benchmark:
        modes.append("PHASE 2 (BENCHMARKS)")
    if args.ablation:
        modes.append("PHASE 3 (ABLATION)")

    print(DIVIDER, flush=True)
    print(f"{' -> '.join(modes)} | Run ID: {ctx.run_id}", flush=True)
    print(f"Started at: {started_at}", flush=True)
    print(DIVIDER, flush=True)

    # Run Phase 3 in the same default Spark session when --benchmark is not requested.
    report = _run_phase1(ctx, run_ablation=args.ablation and not args.benchmark)

    if args.benchmark:
        with stage("Systems Benchmark Suite (Issue #4)", timings=ctx.timings, key="benchmark"):
            run_benchmark_suite(
                config=ctx.config,
                results_dir=ctx.root / "results",
                gold_dir=ctx.gold_dir,
            )

    if args.benchmark and args.ablation:
        with spark_session("flight_delay_ablation") as spark:
            ctx.spark = spark
            ablation.run_ablation(ctx)
        ctx.spark = None

    summary = cleanup_workspace(ctx.root)
    if summary.removed_dirs:
        print(format_cleanup_summary(summary), flush=True)
    total_seconds = time.monotonic() - started
    record_stage_timings(ctx.run_id, started_at, ctx.timings, total_seconds, ctx.root / "results")
    print_pipeline_summary(ctx, report, total_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
