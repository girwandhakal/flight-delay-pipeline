"""Reuse validated weather Gold independently of model/result publication."""

import hashlib
import inspect
import json
from time import monotonic
from uuid import uuid4
from pathlib import Path
from typing import Any, Optional, Union

from scripts.bts.cache import active, current_partitions
from scripts.origin_weather.clean import file_fingerprints
from scripts.origin_weather.download import project_path, sha256, write_json


def input_identity(
    root: Union[str, Path],
    settings: dict[str, Any],
    policy: dict[str, Any],
    config: dict[str, Any],
    parent: Optional[Union[str, Path]] = None,
    scope=None,
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

    if scope is None:
        raise ValueError("Weather station/year selection is missing.")
    manifest = scoped_manifest(manifest, scope)
    weather_files = []
    for name, entry in manifest["files"].items():
        if name != f"{entry['year']}/LCD_{entry['station_id']}_{entry['year']}.csv":
            raise ValueError("Weather ledger path differs from the file consumed by cleaning")
        path = project_path(settings["output_directory"], name)
        if entry["status"] in ("verified", "downloaded"):
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Weather file size changed: {name}")
            weather_files.append(path)
        elif entry["status"] != "unavailable" or path.exists():
            raise ValueError(f"Weather availability changed: {name}")

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

    bts = active(root, "gold")
    if bts.parent.name == "bts_generations":
        bts_files = current_partitions(root, config, bts)
    else:
        from scripts.bts.recent_delays import features_current as legacy_current
        if not legacy_current(root, config):
            raise ValueError("Legacy BTS Gold is incomplete or stale")
        bts_files = partition_files(bts, config["splits"])

    identity = {
        "splits": config["splits"],
        "policy": policy,
        "bts": bts_files,
        "weather": file_fingerprints(weather_files), "scope": sorted(scope),
        "stations": stations_summary,
        "references": reference_hashes,
        "code": {name: sha256(root / "scripts" / name) for name in code},
    }

    if parent is not None:
        parent_path = Path(parent).resolve()
        if parent_path.parent != (root / "data/gold/bts_origin_weather").resolve():
            raise ValueError("Invalid origin Gold generation path")
        identity["parent"] = consumed(parent_path, config["splits"])
        parent_inputs = json.loads((parent_path / "_reuse.json").read_text())["inputs"]
        if parent_inputs != input_identity(root, settings, policy, config, scope=parent_inputs.get("scope")):
            raise ValueError("Origin Gold inputs changed")

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
    partitions = partition_files(directory, inputs["splits"])
    outputs = file_fingerprints(directory.rglob("*.parquet"))
    if not outputs:
        raise ValueError("Cannot cache weather Gold without Parquet files.")

    write_json(
        directory / "_reuse.json",
        {
            "inputs": inputs, "key": contract_digest({"version": 1, "inputs": inputs}),
            "outputs": outputs, "partitions": partitions,
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
    started = monotonic()
    variant = "bts_origin_weather" if parent is None else "bts_origin_destination_weather"
    base = (Path(root) / "data/gold" / variant).resolve()

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
            inputs = input_identity(root, settings, policy, config, parent, saved["inputs"].get("scope"))
            if saved.get("inputs") != inputs:
                continue
            consumed(directory, config["splits"])

        except (OSError, ValueError, KeyError, TypeError):
            continue

        print(
            f"Reusing {variant} Gold from {directory} ({monotonic() - started:.2f}s); "
            "skipping acquisition, cleaning and matching.",
            flush=True,
        )
        return directory

    print(f"{variant} cache miss (no verified matching generation, {monotonic() - started:.2f}s)", flush=True)
    return None


def partition_files(directory, partitions):
    """Fingerprint each partition relative to its own root so generations can move."""
    outputs = {}
    for name in partitions:
        part = (directory / name).resolve()
        marker = part / "_SUCCESS"
        if (not part.is_relative_to(directory.resolve()) or not marker.is_file()
                or marker.resolve().parent != part):
            raise ValueError(f"Incomplete partition: {name}")
        files = file_fingerprints(part.rglob("*.parquet"))
        if not files:
            raise ValueError(f"Empty partition: {name}")
        for item in files:
            path = Path(item["path"])
            item["path"] = path.relative_to(part).as_posix()
            # The Parquet footer includes the physical schema. Reading just the
            # footer keeps validation independent of Spark and of row count.
            with path.open("rb") as stream:
                if stream.read(4) == b"PAR1":
                    stream.seek(-8, 2)
                    length = int.from_bytes(stream.read(4), "little")
                    if stream.read(4) != b"PAR1" or length > item["bytes"] - 12:
                        raise ValueError(f"Invalid Parquet footer: {path}")
                    stream.seek(-8 - length, 2)
                    item["footer_sha256"] = hashlib.sha256(stream.read(length)).hexdigest()
        outputs[name] = files
    return outputs


def output_identity(directory, partitions, sidecars):
    hashes = {}
    for name in sidecars:
        path = project_path(directory, name)
        hashes[name] = sha256(path)
    outputs = {"partitions": partition_files(directory, partitions), "sidecars": hashes}
    # A BTS month only has quarantine output when cleaning rejected rows.
    quarantine = directory / "_quarantine"
    if quarantine.exists():
        names = [path.relative_to(directory).as_posix() for path in quarantine.glob("*/*") if path.is_dir()]
        outputs["quarantine"] = partition_files(directory, names)
    return outputs


def logic_hash(*functions):
    """Hash shared helpers without coupling their callers to unrelated module code."""
    return hashlib.sha256("".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def contract_digest(contract):
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def resolve_stage(base, inputs, partitions, sidecars, build):
    """Reuse a verified generation or publish an isolated rebuild."""
    started = monotonic()
    identity = inputs()
    contract = {"version": 1, "inputs": identity, "partitions": list(partitions), "sidecars": list(sidecars)}
    key = contract_digest(contract)
    prefix = key[:16]  # Keep Spark scratch paths short enough for Windows.
    reason = "no matching inputs"
    for path in sorted(base.glob(f"{prefix}-*"), reverse=True):
        if path.resolve().parent != base.resolve():
            continue
        try:
            saved = json.loads((path / "_stage.json").read_text())
            if (saved["contract"] == contract and saved["key"] == key
                    and saved["outputs"] == output_identity(path, partitions, sidecars)):
                print(f"{base.name} cache hit ({monotonic() - started:.2f}s)", flush=True)
                return path
        except (ValueError, OSError, KeyError, TypeError):
            pass
        reason = "incomplete or changed output"
    path = base / f"{prefix}-{uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    provenance = build(path)
    outputs = output_identity(path, partitions, sidecars)
    # Only the final manifest makes this generation visible to later runs.
    if inputs() != identity:
        raise ValueError(f"{base.name} inputs changed during preparation")
    write_json(path / "_stage.json", {"key": key, "contract": contract, "outputs": outputs,
                                      "reason": reason, "seconds": monotonic() - started,
                                      "provenance": provenance})
    print(f"{base.name} cache miss ({reason}, {monotonic() - started:.2f}s)", flush=True)
    return path


def consumed(directory, partitions):
    """Check the entire producer before selecting the partitions a consumer reads."""
    if (directory / "_stage.json").is_file():
        saved = json.loads((directory / "_stage.json").read_text())
        contract = saved["contract"]
        digest = contract_digest(contract)
        if contract.get("version") != 1 or saved.get("key") != digest or not directory.name.startswith(digest[:16] + "-"):
            raise ValueError(f"Invalid stage manifest: {directory}")
        current = output_identity(directory, contract["partitions"], contract["sidecars"])
        if current != saved["outputs"]:
            raise ValueError(f"Changed stage output: {directory}")
        return {name: current["partitions"][name] for name in partitions}
    saved = json.loads((directory / "_reuse.json").read_text())
    current = partition_files(directory, saved["inputs"]["splits"])
    if ("key" in saved and saved["key"] != contract_digest({"version": 1, "inputs": saved["inputs"]})
            or ("partitions" in saved and saved["partitions"] != current)
            or saved["outputs"] != file_fingerprints(directory.rglob("*.parquet"))
            or saved["metadata"] != sha256(directory / "_metadata.json")):
        raise ValueError(f"Changed weather Gold: {directory}")
    return {name: current[name] for name in partitions}


def scoped_manifest(manifest, names):
    # NOAA tracks both airport roles in one acquisition ledger. Only pass the
    # requested station-years to cleaning, or the other role becomes a dependency.
    return {**manifest, "files": {name: manifest["files"][name] for name in sorted(set(names))}}
