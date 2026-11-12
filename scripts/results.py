"""Store the latest results for each model in one JSON file."""

import json
from pathlib import Path
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def replace_file(source, target):
    """Retry transient file locks without removing the previous destination."""
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)


def records(frame):
    """Convert a pandas table to JSON records, including nulls and dates."""
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def update_results(updates, results_dir=None):
    # shortcut: one pipeline writer at a time; add locking if concurrent runs are needed.
    path = Path(results_dir or ROOT / "results") / "results.json"
    content = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for model, fields in updates.items():
        entry = content.setdefault(model, {})
        if "training" in fields:
            entry.pop("test", None)  # Test scores from a previous fit no longer apply.
            entry.pop("run_id", None)
        entry.update(fields)
    serialized = json.dumps(content, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-", dir=path.parent) as tmp:
        staged = Path(tmp) / "results.json"
        staged.write_text(serialized, encoding="utf-8")
        replace_file(staged, path)
    return path
