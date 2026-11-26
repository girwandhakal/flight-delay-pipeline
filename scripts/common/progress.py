"""Console stage timing and human-readable pipeline output formatting."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
import time
from typing import Any, Optional

DIVIDER = "=" * 80


@contextmanager
def stage(
    title: str,
    *,
    timings: Optional[dict[str, float]] = None,
    key: Optional[str] = None,
) -> Generator[None, Any, None]:
    """Measure and log elapsed wall-clock time for a pipeline sub-step."""
    started = time.monotonic()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"  [{timestamp}] START: {title}", flush=True)

    try:
        yield
    except (Exception, KeyboardInterrupt) as error:
        elapsed = time.monotonic() - started
        fail_time = datetime.now().astimezone().isoformat(timespec="seconds")
        print(
            f"  [{fail_time}] FAILED: {title} after {elapsed:.1f}s "
            f"({type(error).__name__}: {error})",
            flush=True,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        if timings is not None:
            timings[key or title] = elapsed
        done_time = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"  [{done_time}] DONE: {title} ({elapsed:.1f}s)", flush=True)


@contextmanager
def pipeline_stage(
    step: int,
    total: int,
    title: str,
    *,
    timings: Optional[dict[str, float]] = None,
) -> Generator[None, Any, None]:
    """Print a clean banner and elapsed summary for a top-level pipeline stage."""
    started = time.monotonic()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    label = f"Stage {step}/{total}: {title}"
    print(f"\n{DIVIDER}\n[{label}] STARTED at {timestamp}\n{DIVIDER}", flush=True)

    try:
        yield
    except (Exception, KeyboardInterrupt) as error:
        elapsed = time.monotonic() - started
        fail_time = datetime.now().astimezone().isoformat(timespec="seconds")
        print(
            f"[{label}] FAILED at {fail_time} after {elapsed:.1f}s "
            f"({type(error).__name__}: {error})",
            flush=True,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        if timings is not None:
            timings[label] = elapsed
        print(f"[{label}] COMPLETED in {elapsed:.1f}s", flush=True)
