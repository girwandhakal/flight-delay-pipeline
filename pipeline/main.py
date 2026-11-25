"""Run the flight-delay data pipeline sequentially from BTS ingestion to validation."""

from datetime import datetime
from pathlib import Path
import sys
import time

# Support direct execution with `python pipeline/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aircraft_rotation import run as aircraft_rotation
from scripts.bts import run as bts
from scripts.common.cleanup import cleanup_workspace, format_cleanup_summary
from scripts.common.context import PipelineContext
from scripts.common.progress import DIVIDER
from scripts.common.spark import spark_session
from scripts.destination_weather import run as destination_weather
from scripts.enroute_weather import run as enroute_weather
from scripts.modeling import run as modeling
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


def main() -> int:
    """Execute all six flight-delay pipeline stages sequentially."""
    started = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ctx = PipelineContext.create()

    print(DIVIDER, flush=True)
    print(f"FLIGHT DELAY PREDICTION PIPELINE | Run ID: {ctx.run_id}", flush=True)
    print(f"Started at: {started_at}", flush=True)
    print(DIVIDER, flush=True)

    with spark_session("flight_delay") as spark:
        ctx.spark = spark
        bts.run(ctx)
        origin_weather.run(ctx)
        destination_weather.run(ctx)
        aircraft_rotation.run(ctx)
        enroute_weather.run(ctx)
        report = modeling.run(ctx)

    summary = cleanup_workspace(ctx.root)
    if summary.removed_dirs:
        print(format_cleanup_summary(summary), flush=True)
    print_pipeline_summary(ctx, report, time.monotonic() - started)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python pipeline/main.py")
    raise SystemExit(main())
