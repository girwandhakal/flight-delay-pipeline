"""Resolve the BTS data generation separately from model and result publication."""

import json
from uuid import uuid4

from scripts.common.config import project_path
from scripts.common.io import file_fingerprints, sha256, write_json
from scripts.common.results import records, update_results
from scripts.common.stage_cache import consumed, logic_hash, resolve_stage
from scripts.common.time_utils import local_cutoff


def active(root, layer):
    """Return the active BTS Silver or Gold generation directory (or legacy path)."""
    base = root / "data" / layer / "bts_generations"
    pointer = base / "_current.json"
    if not pointer.exists():
        return root / "data" / layer / "bts"
    path = (base / json.loads(pointer.read_text())).resolve()
    if path.parent != base.resolve():
        raise ValueError("Invalid BTS generation path")
    return path


def months(config):
    """Return configured YYYY/MM month partition names for BTS."""
    from .download import iter_months
    settings = config["datasets"]["bts"]
    return [f"{year:04d}/{month:02d}" for year, month in iter_months(settings["start_month"], settings["end_month"])]


def silver_inputs(root, config):
    """Build provenance inputs for BTS Silver cleaning from Bronze CSVs."""
    bronze = project_path(root, config["datasets"]["bts"]["output_directory"])
    files = []
    for month in months(config):
        candidates = list((bronze / month).glob("*.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one BTS Bronze CSV for {month}")
        files.extend(candidates)
    return {"months": months(config), "bronze": file_fingerprints(files),
            "cleaner": sha256(root / "scripts/bts/clean.py")}


def silver_partitions(root, config, silver):
    """Verify BTS Silver matches current inputs and return month partition fingerprints."""
    contract = json.loads((silver / "_stage.json").read_text())["contract"]
    if contract["inputs"] != silver_inputs(root, config):
        raise ValueError("BTS Silver inputs changed")
    return consumed(silver, months(config))


def gold_inputs(root, config, silver):
    """Build provenance inputs for BTS Gold feature engineering."""
    from .recent_delays import feature_manifest
    return {"manifest": feature_manifest(root, config, silver_dir=silver),
            "clock": logic_hash(local_cutoff),
            "silver": silver_partitions(root, config, silver), "months": months(config),
            "code": {name: sha256(root / "scripts" / name) for name in
                     ("bts/build_features.py", "bts/recent_delays.py", "bts/block_time.py")}}


def current_partitions(root, config, gold):
    """Verify BTS Gold matches current Silver/feature inputs and return split fingerprints."""
    saved = json.loads((gold / "_stage.json").read_text())
    if saved["contract"]["inputs"] != gold_inputs(root, config, active(root, "silver")):
        raise ValueError("BTS Gold inputs changed")
    return consumed(gold, config["splits"])


def features_current(root, config):
    """Return True if the active BTS Gold generation matches current inputs."""
    try:
        current_partitions(root, config, active(root, "gold"))
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare(spark, root, layer, publish_results):
    """Resolve or build a cached BTS Silver or Gold generation and update _current.json."""
    from .build_features import build_features
    from .block_time import BLOCK_TIME_SIDECAR
    from .clean import clean_bts_data
    from .recent_delays import TAXI_BASELINE_FILE
    import pandas as pd

    config = json.loads((root / "config.json").read_text())
    base = root / "data" / layer / "bts_generations"
    if layer == "silver":
        partitions = months(config)

        def inputs():
            return silver_inputs(root, json.loads((root / "config.json").read_text()))

        def build(path):
            summary = clean_bts_data(spark, bronze_dir=project_path(root, config["datasets"]["bts"]["output_directory"]),
                                     silver_dir=path, quarantine_dir=path / "_quarantine",
                                     publish_results=False, selected_months=partitions)
            write_json(path / "summary.json", records(summary))

        sidecars = ("summary.json",)
    else:
        try:
            silver = active(root, "silver")
            silver_partitions(root, config, silver)
        except (KeyError, OSError, TypeError, ValueError):
            # The standalone feature builder may be the first command after migration.
            prepare(spark, root, "silver", publish_results=False)
            silver = active(root, "silver")
        partitions = list(config["splits"])
        history = None

        def inputs():
            nonlocal history
            identity = gold_inputs(root, json.loads((root / "config.json").read_text()), silver)
            history = identity["manifest"]["block_history"]
            return identity

        def build(path):
            summary = build_features(spark, silver_dir=silver, gold_dir=path, publish_results=False,
                                     block_history=history)
            write_json(path / "summary.json", records(summary))
            return {"silver": str(silver)}

        sidecars = ("summary.json", "_features.json", TAXI_BASELINE_FILE, BLOCK_TIME_SIDECAR)
    path = resolve_stage(base, inputs, partitions, sidecars, build)
    # Competing builders may finish together. Give each pointer update its own
    # scratch file so a reader always sees one complete generation name.
    temporary = base / f"{uuid4().hex}.json"
    try:
        write_json(temporary, path.name)
        temporary.replace(base / "_current.json")
    finally:
        temporary.unlink(missing_ok=True)
    rows = json.loads((path / "summary.json").read_text())
    if publish_results:
        update_results({"bts": {"cleaning" if layer == "silver" else "splits": rows}}, root / "results")
    return pd.DataFrame(rows)
