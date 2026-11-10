"""Run the BTS workflow in order, stopping when a stage fails."""

from datetime import timedelta
from pathlib import Path
import sys
import time

# Find the project scripts even when this file is run from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bts_clean
import bts_download
import bts_train
import bts_transform

STAGES = [
    ("Stage 1/4", "BTS Data Acquisition (Download)", bts_download.main),
    ("Stage 2/4", "BTS Data Cleaning (Bronze -> Silver)", bts_clean.main),
    ("Stage 3/4", "Feature Engineering & Chronological Splitting (Silver -> Gold)", bts_transform.main),
    ("Stage 4/4", "Baseline Model Training & Validation (Gold -> Models)", bts_train.main),
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
        print(f"[{step}] ERROR: {title} encountered an unhandled exception after {elapsed:.1f}s:")
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


def main():
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
    print(f"ALL 4 PIPELINE STAGES COMPLETED SUCCESSFULLY in {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"{banner}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
