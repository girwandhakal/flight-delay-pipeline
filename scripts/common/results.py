"""Store latest model results and timestamped snapshots of replaced results."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Optional, Union
from uuid import uuid4

import pandas as pd

from .config import ROOT


def replace_file(source: Path, target: Path) -> None:
    """Atomically replace a file, retrying transient Windows permission locks."""
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Exponential backoff for transient Windows file locks (antivirus/indexers).
            time.sleep(0.1 * (2 ** attempt))


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to JSON records, preserving nulls and ISO dates."""
    return json.loads(
        frame.to_json(orient="records", date_format="iso", double_precision=15)
    )


def update_results(
    updates: dict[str, Any],
    results_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Update results.json with model metrics, archiving replaced versions."""
    # Single-writer assumption: concurrent pipeline writers are not supported.
    path = Path(results_dir or ROOT / "results") / "results.json"
    previous = path.read_bytes() if path.exists() else None
    content = json.loads(previous) if previous is not None else {}

    for model, fields in updates.items():
        entry = content.setdefault(model, {})
        if "training" in fields:
            # Clear stale test metrics whenever a model family is retrained.
            entry.pop("test", None)
            entry.pop("run_id", None)
        entry.update(fields)

    serialized = json.dumps(content, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Stage inside the target directory so atomic replace stays on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".results-", dir=path.parent) as tmp:
        staged = Path(tmp) / "results.json"
        staged.write_text(serialized, encoding="utf-8")

        if previous is not None:
            history = path.parent.parent / "results_history"
            history.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            snapshot = Path(tmp) / "previous.json"
            snapshot.write_bytes(previous)
            replace_file(snapshot, history / f"results_{timestamp}_{uuid4().hex}.json")

        replace_file(staged, path)

    return path


def record_stage_timings(
    run_id: str,
    started_at: str,
    timings: dict[str, float],
    total_seconds: float,
    results_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Append a completed run's per-stage and total durations to results/stage_timings.json."""
    path = Path(results_dir or ROOT / "results") / "stage_timings.json"
    history: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON list in {path}")
        history = loaded

    history.append({
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_seconds": round(total_seconds, 2),
        "stages": {name: round(elapsed, 2) for name, elapsed in timings.items()},
    })

    serialized = json.dumps(history, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage inside the target directory so atomic replace stays on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".timings-", dir=path.parent) as tmp:
        staged = Path(tmp) / "stage_timings.json"
        staged.write_text(serialized, encoding="utf-8")
        replace_file(staged, path)

    return path
