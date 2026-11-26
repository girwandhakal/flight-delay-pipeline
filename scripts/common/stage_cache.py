"""Stage-level cache resolution and Parquet partition fingerprinting."""

import hashlib
import inspect
import json
from pathlib import Path
import shutil
from time import monotonic
from typing import Any, Optional, Union
from uuid import uuid4

from .config import project_path
from .io import file_fingerprints, sha256, write_json


def _default_bts_files(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Resolve active BTS Gold partition fingerprints without importing scripts.bts."""
    base = root / "data" / "gold" / "bts_generations"
    pointer = base / "_current.json"
    if pointer.exists():
        bts_dir = (base / json.loads(pointer.read_text(encoding="utf-8"))).resolve()
        if bts_dir.parent != base.resolve():
            raise ValueError("Invalid BTS generation path")
        return consumed(bts_dir, config["splits"])
    legacy = root / "data" / "gold" / "bts"
    return partition_files(legacy, config["splits"])


def input_identity(
    root: Union[str, Path],
    settings: dict[str, Any],
    policy: dict[str, Any],
    config: dict[str, Any],
    parent: Optional[Union[str, Path]] = None,
    scope=None,
    *,
    bts_files: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a deterministic provenance fingerprint of all inputs for weather Gold."""
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

    resolved_bts = bts_files if bts_files is not None else _default_bts_files(root, config)
    identity = {
        "splits": config["splits"],
        "policy": policy,
        "bts": resolved_bts,
        "weather": file_fingerprints(weather_files),
        "scope": sorted(scope),
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
        if parent_inputs != input_identity(
            root, settings, policy, config, scope=parent_inputs.get("scope"), bts_files=resolved_bts
        ):
            raise ValueError("Origin Gold inputs changed")

    return identity


def save_gold(directory: Union[str, Path], inputs: dict[str, Any]) -> None:
    """Record a reuse marker sidecar (_reuse.json) in a completed weather Gold directory."""
    directory = Path(directory)
    partitions = partition_files(directory, inputs["splits"])
    outputs = file_fingerprints(directory.rglob("*.parquet"))
    if not outputs:
        raise ValueError("Cannot cache weather Gold without Parquet files.")

    write_json(
        directory / "_reuse.json",
        {
            "inputs": inputs,
            "key": contract_digest({"version": 1, "inputs": inputs}),
            "outputs": outputs,
            "partitions": partitions,
            "metadata": sha256(directory / "_metadata.json"),
        },
    )


STAGE_LABELS = {
    ("silver", "bts_generations"): "BTS Silver (cleaned flights)",
    ("gold", "bts_generations"): "BTS Gold (schedule & rolling features)",
    ("silver", "taf_stations"): "TAF station list",
    ("silver", "taf"): "TAF Silver (forecast segments)",
    ("silver", "sigmets"): "SIGMET Silver (convective advisories)",
    ("gold", "rotation"): "Aircraft Rotation Gold",
    ("gold", "enroute"): "En-route Weather Gold",
}
WEATHER_GOLD_LABELS = {
    "bts_origin_weather": "Origin Surface Weather Gold",
    "bts_origin_destination_weather": "Destination Surface Weather Gold",
}


def _stage_label(base: Path) -> str:
    """Return a descriptive console label for a cached stage directory."""
    return STAGE_LABELS.get((base.parent.name, base.name), base.name)


def find_gold(
    root: Union[str, Path],
    settings: dict[str, Any],
    policy: dict[str, Any],
    config: dict[str, Any],
    parent: Optional[Union[str, Path]] = None,
    *,
    bts_files: Optional[dict[str, Any]] = None,
) -> Optional[Path]:
    """Return an existing completed weather Gold directory matching current inputs, or None."""
    started = monotonic()
    variant = "bts_origin_weather" if parent is None else "bts_origin_destination_weather"
    label = WEATHER_GOLD_LABELS[variant]
    base = (Path(root) / "data/gold" / variant).resolve()

    for marker in sorted(base.glob("*/_reuse.json"), reverse=True):
        directory = marker.parent.resolve()
        if not directory.is_relative_to(base):
            continue
        try:
            saved = json.loads(marker.read_text())
            if not isinstance(saved, dict):
                continue
            inputs = input_identity(
                root, settings, policy, config, parent, saved["inputs"].get("scope"),
                bts_files=bts_files,
            )
            if saved.get("inputs") != inputs:
                continue
            consumed(directory, config["splits"])
        except (OSError, ValueError, KeyError, TypeError):
            continue

        print(
            f"  [Cache HIT] Reusing {label} ({directory.name}, {monotonic() - started:.2f}s).",
            flush=True,
        )
        return directory

    print(
        f"  [Cache MISS] {label} — no matching cache ({monotonic() - started:.2f}s), building now...",
        flush=True,
    )
    return None


def partition_files(directory: Path, partitions) -> dict[str, list[dict[str, Any]]]:
    """Fingerprint each Parquet partition and validate its PAR1 footer."""
    outputs = {}
    for name in partitions:
        part = (directory / name).resolve()
        marker = part / "_SUCCESS"
        if (
            not part.is_relative_to(directory.resolve())
            or not marker.is_file()
            or marker.resolve().parent != part
        ):
            raise ValueError(f"Incomplete partition: {name}")
        files = file_fingerprints(part.rglob("*.parquet"))
        if not files:
            raise ValueError(f"Empty partition: {name}")
        for item in files:
            path = Path(item["path"])
            item["path"] = path.relative_to(part).as_posix()
            # Reading just the Parquet footer verifies the physical file without Spark.
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


def output_identity(directory: Path, partitions, sidecars) -> dict[str, Any]:
    """Fingerprint completed partitions, sidecar JSONs, and optional quarantine partitions."""
    hashes = {name: sha256(project_path(directory, name)) for name in sidecars}
    outputs: dict[str, Any] = {"partitions": partition_files(directory, partitions), "sidecars": hashes}
    quarantine = directory / "_quarantine"
    if quarantine.exists():
        names = [path.relative_to(directory).as_posix() for path in quarantine.glob("*/*") if path.is_dir()]
        outputs["quarantine"] = partition_files(directory, names)
    return outputs


def logic_hash(*functions) -> str:
    """Hash source code of helper functions to invalidate caches when logic changes."""
    return hashlib.sha256("".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def contract_digest(contract: dict[str, Any]) -> str:
    """Compute a canonical SHA-256 digest for a JSON-serializable stage contract."""
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def resolve_stage(base: Path, inputs, partitions, sidecars, build) -> Path:
    """Reuse a verified stage generation matching `inputs()` or build a fresh one."""
    started = monotonic()
    label = _stage_label(base)
    identity = inputs()
    contract = {"version": 1, "inputs": identity, "partitions": list(partitions), "sidecars": list(sidecars)}
    key = contract_digest(contract)
    prefix = key[:16]  # Keep generation directory names short for Windows paths.
    reason = "no matching inputs"
    for path in sorted(base.glob(f"{prefix}-*"), reverse=True):
        if path.resolve().parent != base.resolve():
            continue
        try:
            saved = json.loads((path / "_stage.json").read_text())
            if (
                saved["contract"] == contract
                and saved["key"] == key
                and saved["outputs"] == output_identity(path, partitions, sidecars)
            ):
                print(f"  [Cache HIT] Reusing {label} ({monotonic() - started:.2f}s)", flush=True)
                return path
        except (ValueError, OSError, KeyError, TypeError):
            pass
        reason = "incomplete or changed output"

    # Announce cache misses before starting expensive builds so logs stay clear during long runs.
    print(f"  [Cache MISS] {label} ({reason}) — building now...", flush=True)
    path = base / f"{prefix}-{uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        provenance = build(path)
        outputs = output_identity(path, partitions, sidecars)
        write_json(
            path / "_stage.json",
            {
                "key": key,
                "contract": contract,
                "outputs": outputs,
                "reason": reason,
                "seconds": monotonic() - started,
                "provenance": provenance,
            },
        )
    except BaseException:
        # Remove partial build directories immediately so failed runs do not accumulate on disk.
        shutil.rmtree(path, ignore_errors=True)
        raise
    print(f"  [Cache BUILT] {label} in {monotonic() - started:.1f}s", flush=True)
    return path


def consumed(directory: Path, partitions) -> dict[str, Any]:
    """Verify a producer directory's outputs and return fingerprints for `partitions`."""
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
    if (
        ("key" in saved and saved["key"] != contract_digest({"version": 1, "inputs": saved["inputs"]}))
        or ("partitions" in saved and saved["partitions"] != current)
        or saved["outputs"] != file_fingerprints(directory.rglob("*.parquet"))
        or saved["metadata"] != sha256(directory / "_metadata.json")
    ):
        raise ValueError(f"Changed weather Gold: {directory}")
    return {name: current[name] for name in partitions}


def scoped_manifest(manifest: dict[str, Any], names) -> dict[str, Any]:
    """Filter the NOAA acquisition ledger down to only the station-years needed by this role."""
    return {**manifest, "files": {name: manifest["files"][name] for name in sorted(set(names))}}
