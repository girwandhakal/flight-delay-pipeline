"""Reuse validated weather Gold independently of model/result publication."""

import json
from pathlib import Path

from scripts.origin_weather.clean import file_fingerprints
from scripts.origin_weather.download import project_path, sha256, write_json


"""Reuse validated weather Gold independently of model/result publication."""

import json
from pathlib import Path
from typing import Any, Optional, Union

from scripts.origin_weather.clean import file_fingerprints
from scripts.origin_weather.download import project_path, sha256, write_json


def input_identity(
    root: Union[str, Path],
    settings: dict[str, Any],
    policy: dict[str, Any],
    config: dict[str, Any],
    parent: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Build a deterministic provenance fingerprint of all inputs for weather Gold.

    Captures BTS gold files, raw weather files, station metadata, airport
    references, transformation source code hashes, and parent origin gold (if destination).

    Args:
        root: Project root directory.
        settings: Weather dataset settings from config.json.
        policy: Weather matching policy parameters (max age, publication lag).
        config: Full configuration dictionary.
        parent: Optional path to parent origin Gold directory (for destination).

    Returns:
        Dictionary defining the full input identity fingerprint.
    """
    root = Path(root)
    references = [settings["reference_directory"]]
    code = [
        "origin_weather/build_features.py",
        "origin_weather/clean.py",
        "origin_weather/measurements.py",
        "origin_weather/download.py",
    ]
    if parent is not None:
        references.append(
            project_path(root, config["destination_weather"]["reference_directory"])
        )
        code.append("destination_weather/build_features.py")

    manifest = json.loads((settings["output_directory"] / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Weather acquisition is incomplete.")

    stations_summary = sorted([
        station_entry["station_id"],
        station_entry["year"],
        station_entry["time_zone"],
        "downloaded" if station_entry["status"] == "verified" else station_entry["status"],
    ] for station_entry in manifest["files"].values())

    reference_hashes = {
        str(Path(directory).resolve()): {
            name: sha256(Path(directory) / name)
            for name in ("sources.json", "airports.csv", "airport_station.csv")
        }
        for directory in references
    }

    identity = {
        "splits": config["splits"],
        "policy": policy,
        "bts": file_fingerprints((root / "data/gold/bts").rglob("*.parquet")),
        "bts_features": sha256(root / "data/gold/bts/_features.json"),
        "weather": file_fingerprints(settings["output_directory"].rglob("*.csv")),
        "stations": stations_summary,
        "references": reference_hashes,
        "code": {name: sha256(root / "scripts" / name) for name in code},
    }

    if parent is not None:
        parent_path = Path(parent).resolve()
        identity["parent"] = {
            "path": str(parent_path),
            "metadata": sha256(parent_path / "_metadata.json"),
            "files": file_fingerprints(parent_path.rglob("*.parquet")),
        }

    return identity


def save_gold(directory: Union[str, Path], inputs: dict[str, Any]) -> None:
    """Record a reuse marker sidecar (_reuse.json) in a completed Gold directory.

    Args:
        directory: Gold dataset directory to cache.
        inputs: Input identity dictionary computed before building Gold.

    Raises:
        ValueError: If split completion markers or parquet outputs are missing.
    """
    directory = Path(directory)
    if not all((directory / split / "_SUCCESS").is_file() for split in inputs["splits"]):
        raise ValueError("Cannot cache incomplete weather Gold.")

    outputs = file_fingerprints(directory.rglob("*.parquet"))
    if not outputs:
        raise ValueError("Cannot cache weather Gold without Parquet files.")

    write_json(
        directory / "_reuse.json",
        {
            "inputs": inputs,
            "outputs": outputs,
            "metadata": sha256(directory / "_metadata.json"),
        },
    )


def find_gold(
    root: Union[str, Path],
    settings: dict[str, Any],
    policy: dict[str, Any],
    config: dict[str, Any],
    parent: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Search for an existing completed Gold dataset matching current inputs.

    Checks newest to oldest completed builds. Reuses the dataset if all inputs,
    output parquet fingerprints, completion markers, and metadata match exactly.

    Args:
        root: Project root directory.
        settings: Weather dataset settings.
        policy: Weather matching policy parameters.
        config: Full configuration dictionary.
        parent: Optional parent origin Gold directory for destination searches.

    Returns:
        Resolved Path to matching Gold directory, or None if rebuild is needed.
    """
    variant = "bts_origin_weather" if parent is None else "bts_origin_destination_weather"
    base = (Path(root) / "data/gold" / variant).resolve()

    try:
        inputs = input_identity(root, settings, policy, config, parent)
    except (KeyError, OSError, ValueError):
        return None

    # Scan completed builds from newest to oldest
    for marker in sorted(base.glob("*/_reuse.json"), reverse=True):
        directory = marker.parent.resolve()
        if not directory.is_relative_to(base):
            continue

        try:
            saved = json.loads(marker.read_text())
            if not isinstance(saved, dict):
                continue

            # Verify inputs, split completion, output parquet fingerprints, and metadata
            inputs_match = saved.get("inputs") == inputs
            splits_complete = all(
                (directory / split / "_SUCCESS").is_file() for split in config["splits"]
            )
            saved_outputs = saved.get("outputs")
            outputs_match = bool(saved_outputs) and (
                saved_outputs == file_fingerprints(directory.rglob("*.parquet"))
            )
            metadata_matches = saved.get("metadata") == sha256(directory / "_metadata.json")

            if not (inputs_match and splits_complete and outputs_match and metadata_matches):
                continue

        except (OSError, ValueError):
            continue

        print(
            f"Reusing {variant} Gold from {directory}; "
            "skipping acquisition, cleaning and matching.",
            flush=True,
        )
        return directory

    return None

