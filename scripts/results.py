"""Store latest model results and timestamped snapshots of replaced results."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


"""Store latest model results and timestamped snapshots of replaced results."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Optional, Union
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def replace_file(source: Path, target: Path) -> None:
    """Atomically replace a file, retrying transient Windows permission locks.

    Args:
        source: Staged file path to move.
        target: Final destination path.

    Raises:
        PermissionError: If the target remains locked after 6 attempts.
    """
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            # Exponential backoff for transient Windows file locks (antivirus/indexers)
            time.sleep(0.1 * (2 ** attempt))


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to JSON records, preserving nulls and ISO dates.

    Args:
        frame: Pandas DataFrame to serialize.

    Returns:
        List of row dictionaries with ISO formatted timestamps.
    """
    return json.loads(
        frame.to_json(orient="records", date_format="iso", double_precision=15)
    )


def update_results(
    updates: dict[str, Any],
    results_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Update results.json with model metrics, archiving replaced versions.

    Preserves existing model entries while updating specified fields.
    When a model undergoes retraining, any stale test scores and run IDs
    are automatically purged.

    Args:
        updates: Nested dictionary mapping model/experiment name to updated fields.
        results_dir: Directory containing results.json. Defaults to <ROOT>/results.

    Returns:
        Path to the updated results.json file.
    """
    # Assumption: one pipeline writer at a time; add file locking if concurrent runs are needed.
    path = Path(results_dir or ROOT / "results") / "results.json"
    previous = path.read_bytes() if path.exists() else None
    content = json.loads(previous) if previous is not None else {}

    for model, fields in updates.items():
        entry = content.setdefault(model, {})
        if "training" in fields:
            # Purge test evaluation scores when retraining, as previous test scores no longer apply
            entry.pop("test", None)
            entry.pop("run_id", None)
        entry.update(fields)

    serialized = json.dumps(content, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use atomic write via temp directory in same parent to prevent partial file writes
    with tempfile.TemporaryDirectory(prefix=".results-", dir=path.parent) as tmp:
        staged = Path(tmp) / "results.json"
        staged.write_text(serialized, encoding="utf-8")

        # Snapshot replaced results into results_history
        if previous is not None:
            history = path.parent.parent / "results_history"
            history.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            snapshot = Path(tmp) / "previous.json"
            snapshot.write_bytes(previous)
            replace_file(snapshot, history / f"results_{timestamp}_{uuid4().hex}.json")

        replace_file(staged, path)

    return path

