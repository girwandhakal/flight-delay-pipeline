"""Console stage timing shared by pipeline and weather preparation."""

from contextlib import contextmanager
from datetime import datetime
import time


"""Console stage timing shared by pipeline and weather preparation."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
import time
from typing import Any, Optional


def print_progress(label: str, completed: int, total: int) -> None:
    """Print a progress snapshot that stays readable alongside stage logs."""
    filled = 24 * completed // total
    bar = "#" * filled + "-" * (24 - filled)
    print(f"  {label} [{bar}] {completed}/{total} ({completed / total:.0%})", flush=True)


@contextmanager
def stage(
    title: str,
    *,
    timings: Optional[dict[str, float]] = None,
    key: Optional[str] = None,
) -> Generator[None, Any, None]:
    """Measure and log the elapsed wall-clock time for a named pipeline stage.

    Args:
        title: Human-readable stage description printed to console.
        timings: Optional dictionary to record elapsed seconds upon success.
        key: Specific key name to store in `timings`. Defaults to `title`.

    Yields:
        None when entering the stage context block.
    """
    started = time.monotonic()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"\n[{timestamp}] START: {title}", flush=True)

    try:
        yield
    except (Exception, KeyboardInterrupt) as error:
        elapsed = time.monotonic() - started
        fail_time = datetime.now().astimezone().isoformat(timespec="seconds")
        print(
            f"[{fail_time}] FAILED: {title} after {elapsed:.1f}s "
            f"({type(error).__name__}: {error})",
            flush=True,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        if timings is not None:
            timings[key or title] = elapsed
        done_time = datetime.now().astimezone().isoformat(timespec="seconds")
        print(f"[{done_time}] DONE: {title} in {elapsed:.1f}s", flush=True)

