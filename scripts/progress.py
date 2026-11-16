"""Console stage timing shared by pipeline and weather preparation."""

from contextlib import contextmanager
from datetime import datetime
import time


@contextmanager
def stage(title):
    started = time.monotonic()
    print(f"\n[{datetime.now().astimezone().isoformat(timespec='seconds')}] START: {title}", flush=True)
    try:
        yield
    except Exception as error:
        print(f"FAILED: {title} after {time.monotonic() - started:.1f}s "
              f"({type(error).__name__}: {error})", flush=True)
        raise
    else:
        print(f"DONE: {title} in {time.monotonic() - started:.1f}s", flush=True)
