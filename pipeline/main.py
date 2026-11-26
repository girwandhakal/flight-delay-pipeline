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
from scripts.common.progress import DIVIDER
from scripts.common.results import record_stage_timings
from scripts.common.spark import spark_session
from scripts.destination_weather import run as destination_weather
from scripts.enroute_weather import run as enroute_weather
from scripts.modeling import ablation, feature_importance, run as modeling
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


def main(argv: list[str] | None = None) -> int:
    """Execute the flight-delay pipeline or the controlled ablation study."""
    args = parse_args(argv)
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ctx = PipelineContext.create()

    print(DIVIDER, flush=True)
    mode_label = (
        "SYSTEMS BENCHMARK SUITE (ISSUE #4)"
        if args.benchmark
        else "ABLATION STUDY (RQ2)" if args.ablation else "FLIGHT DELAY PREDICTION PIPELINE"
    )
    print(f"{mode_label} | Run ID: {ctx.run_id}", flush=True)
    print(f"Started at: {started_at}", flush=True)
    print(DIVIDER, flush=True)

    if args.benchmark:
        run_benchmark_suite()
        return 0

    if args.ablation:
        with spark_session("flight_delay_ablation") as spark:
            ctx.spark = spark
            ablation.run_ablation(ctx)
        return 0

    with spark_session("flight_delay") as spark:
        ctx.spark = spark
        bts.run(ctx)
        origin_weather.run(ctx)
        destination_weather.run(ctx)
        aircraft_rotation.run(ctx)
        enroute_weather.run(ctx)
        report = modeling.run(ctx)
        feature_importance.run(ctx)

    summary = cleanup_workspace(ctx.root)
    if summary.removed_dirs:
        print(format_cleanup_summary(summary), flush=True)
    total_seconds = time.monotonic() - started
    record_stage_timings(ctx.run_id, started_at, ctx.timings, total_seconds, ctx.root / "results")
    print_pipeline_summary(ctx, report, total_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
