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
