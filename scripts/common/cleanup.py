"""Automatic reference-counted retention and failed-build cleanup for pipeline caches."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Optional

from .config import project_path
from .io import dir_size_bytes, rmtree_retry

STAGE_CACHE_ROOTS = (
    Path("data/silver/bts_generations"),
    Path("data/gold/bts_generations"),
    Path("data/silver/taf_stations"),
    Path("data/silver/taf"),
    Path("data/silver/sigmets"),
    Path("data/gold/rotation"),
    Path("data/gold/enroute"),
)
WEATHER_GOLD_ROOTS = (
    Path("data/gold/bts_origin_weather"),
    Path("data/gold/bts_origin_destination_weather"),
)
ALLOWED_DELETE_ROOTS = (
    Path("data/silver"),
    Path("data/gold"),
    Path("data/quarantine"),
    Path("models/flight_delay"),
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class CleanupSummary:
    """Summary of obsolete cache and run directories removed from the workspace."""

    removed_dirs: int
    reclaimed_bytes: int
    removed_paths: tuple[Path, ...]


def _read_json(path: Path) -> Any:
    """Read JSON from `path`, returning None if missing or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _safe_delete_dir(candidate: Path, allowed_root: Path, root: Path) -> int:
    """Validate containment under `allowed_root`, refuse symlinks, and delete `candidate`."""
    if candidate.is_symlink():
        raise ValueError(f"Refusing symlinked cleanup path: {candidate}")
    if not candidate.is_dir():
        return 0
    resolved_root = root.resolve()
    resolved_allowed = allowed_root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_allowed.is_relative_to(resolved_root):
        raise ValueError(f"Cleanup root escapes workspace: {allowed_root}")
    if not resolved_candidate.is_relative_to(resolved_allowed) or resolved_candidate == resolved_allowed:
        raise ValueError(f"Refusing cleanup path outside {allowed_root}: {candidate}")
    freed = dir_size_bytes(candidate)
    rmtree_retry(candidate)
    return freed


def _resolve_under(base: Path, raw_path: Any, root: Optional[Path] = None) -> Optional[Path]:
    """Resolve `raw_path` and return it only if it stays strictly inside `base`."""
    if not isinstance(raw_path, str) or not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute() and root is not None:
        candidate = root / candidate
    try:
        resolved_base = base.resolve()
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved == resolved_base or not resolved.is_relative_to(resolved_base):
        return None
    return resolved


def _normalize_stage_gen(raw_path: Any, stage_base: Path) -> Optional[Path]:
    """Normalize a stage output path (or its child subdir like `gold`/`table`) to its generation dir."""
    resolved = _resolve_under(stage_base, raw_path)
    if resolved is None:
        return None
    base_resolved = stage_base.resolve()
    if resolved.parent == base_resolved:
        return resolved
    if resolved.parent.parent == base_resolved:
        return resolved.parent
    return None


def _current_generation(root: Path, relative_base: str) -> Optional[Path]:
    """Return the active generation pointed to by `_current.json` if it is complete."""
    base = project_path(root, relative_base)
    pointer = base / "_current.json"
    target_name = _read_json(pointer)
    if not isinstance(target_name, str) or not target_name:
        return None
    resolved = _resolve_under(base, str(base / target_name))
    if resolved is None or resolved.parent != base.resolve():
        return None
    if not (resolved / "_stage.json").is_file():
        return None
    return resolved


def _newest_directory(paths: list[Path]) -> Optional[Path]:
    """Return the most recently modified directory from `paths`, or None if empty."""
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, p.name))


def _pin_weather_chain(root: Path, dest_dir: Optional[Path], origin_dir: Optional[Path], pinned: set[Path]) -> None:
    """Pin a weather Gold generation and its upstream origin/silver/quarantine directories."""
    dest_base = project_path(root, "data/gold/bts_origin_destination_weather")
    orig_base = project_path(root, "data/gold/bts_origin_weather")
    if dest_dir is not None and dest_dir.is_dir():
        pinned.add(dest_dir.resolve())
        dest_meta = _read_json(dest_dir / "_metadata.json")
        if isinstance(dest_meta, dict):
            parent = _resolve_under(orig_base, dest_meta.get("parent_gold_dir"), root=root)
            if parent is not None and parent.is_dir():
                origin_dir = parent
    if origin_dir is not None and origin_dir.is_dir():
        resolved_origin = origin_dir.resolve()
        pinned.add(resolved_origin)
        # Pin matching timestamped weather Silver and Quarantine directories when present.
        silver_weather = project_path(root, "data/silver/weather") / resolved_origin.name
        if silver_weather.is_dir():
            pinned.add(silver_weather.resolve())
        quarantine_weather = project_path(root, "data/quarantine/weather") / resolved_origin.name
        if quarantine_weather.is_dir():
            pinned.add(quarantine_weather.resolve())
    if dest_dir is not None and dest_dir.is_dir():
        quarantine_dest = project_path(root, "data/quarantine/weather") / dest_dir.name
        if quarantine_dest.is_dir():
            pinned.add(quarantine_dest.resolve())


def _pin_enroute_chain(root: Path, enroute_dir: Path, pinned: set[Path]) -> None:
    """Pin an en-route Gold generation and trace its upstream stage provenance."""
    pinned.add(enroute_dir.resolve())
    stage = _read_json(enroute_dir / "_stage.json")
    if not isinstance(stage, dict):
        return
    provenance = stage.get("provenance")
    if not isinstance(provenance, dict):
        return

    dest_base = project_path(root, "data/gold/bts_origin_destination_weather")
    rot_base = project_path(root, "data/gold/rotation")
    taf_base = project_path(root, "data/silver/taf")
    taf_stations_base = project_path(root, "data/silver/taf_stations")
    sigmets_base = project_path(root, "data/silver/sigmets")

    dest_dir = _resolve_under(dest_base, provenance.get("destination"), root=root)
    if dest_dir is not None:
        _pin_weather_chain(root, dest_dir, None, pinned)

    rot_dir = _normalize_stage_gen(provenance.get("rotation"), rot_base)
    if rot_dir is not None and rot_dir.is_dir():
        pinned.add(rot_dir.resolve())
        rot_stage = _read_json(rot_dir / "_stage.json")
        if isinstance(rot_stage, dict) and isinstance(rot_stage.get("provenance"), dict):
            rot_dest = _resolve_under(dest_base, rot_stage["provenance"].get("destination"), root=root)
            if rot_dest is not None:
                _pin_weather_chain(root, rot_dest, None, pinned)

    aviation = provenance.get("aviation")
    if isinstance(aviation, dict):
        taf_dir = _normalize_stage_gen(aviation.get("taf"), taf_base)
        if taf_dir is not None and taf_dir.is_dir():
            pinned.add(taf_dir.resolve())
            taf_stage = _read_json(taf_dir / "_stage.json")
            if isinstance(taf_stage, dict) and isinstance(taf_stage.get("provenance"), dict):
                stations_dir = _normalize_stage_gen(taf_stage["provenance"].get("stations"), taf_stations_base)
                if stations_dir is not None and stations_dir.is_dir():
                    pinned.add(stations_dir.resolve())
        sigmets_dir = _normalize_stage_gen(aviation.get("sigmets"), sigmets_base)
        if sigmets_dir is not None and sigmets_dir.is_dir():
            pinned.add(sigmets_dir.resolve())


def _collect_pinned_paths(root: Path) -> set[Path]:
    """Collect all active `_current.json` targets, retained model runs, and their Gold/Silver provenance."""
    pinned: set[Path] = set()

    silver_bts = _current_generation(root, "data/silver/bts_generations")
    if silver_bts is not None:
        pinned.add(silver_bts)

    gold_bts = _current_generation(root, "data/gold/bts_generations")
    if gold_bts is not None:
        pinned.add(gold_bts)
        bts_stage = _read_json(gold_bts / "_stage.json")
        if isinstance(bts_stage, dict) and isinstance(bts_stage.get("provenance"), dict):
            silver_src = _normalize_stage_gen(
                bts_stage["provenance"].get("silver"),
                project_path(root, "data/silver/bts_generations"),
            )
            if silver_src is not None and silver_src.is_dir():
                pinned.add(silver_src.resolve())

    runs_root = project_path(root, "models/flight_delay")
    retained_runs: list[Path] = []
    results = _read_json(project_path(root, "results/results.json"))
    if isinstance(results, dict) and isinstance(results.get("flight_delay"), dict):
        active_run_id = results["flight_delay"].get("run_id")
        if isinstance(active_run_id, str) and RUN_ID_PATTERN.fullmatch(active_run_id):
            active_run = runs_root / active_run_id
            if active_run.is_dir():
                retained_runs.append(active_run.resolve())

    if runs_root.is_dir():
        completed_runs = [
            entry.resolve()
            for entry in runs_root.iterdir()
            if entry.is_dir() and not entry.is_symlink() and (entry / "_SUCCESS").is_file()
        ]
        newest_run = _newest_directory(completed_runs)
        if newest_run is not None:
            retained_runs.append(newest_run)

    enroute_base = project_path(root, "data/gold/enroute")
    for run_dir in retained_runs:
        pinned.add(run_dir)
        manifest = _read_json(run_dir / "run.json")
        if not isinstance(manifest, dict):
            continue
        enroute_dir = _resolve_under(enroute_base, manifest.get("gold"), root=root)
        if enroute_dir is not None and enroute_dir.is_dir():
            _pin_enroute_chain(root, enroute_dir, pinned)

    return pinned


def _prune_generation_root(
    root: Path,
    base: Path,
    allowed_root: Path,
    pinned: set[Path],
    is_valid_fn,
    removed: list[Path],
) -> int:
    """Keep the newest valid generation and any pinned generations under `base`; delete the rest."""
    if not base.is_dir():
        return 0
    valid: list[Path] = []
    incomplete: list[Path] = []
    for entry in sorted(base.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        if is_valid_fn(entry):
            valid.append(entry)
        else:
            incomplete.append(entry)

    keep = set(pinned)
    newest = _newest_directory(valid)
    if newest is not None:
        keep.add(newest.resolve())

    reclaimed = 0
    for entry in [*incomplete, *valid]:
        if entry.resolve() in keep:
            continue
        reclaimed += _safe_delete_dir(entry, allowed_root, root)
        removed.append(entry)
    return reclaimed


def cleanup_workspace(root: Path) -> CleanupSummary:
    """Prune aborted builds, empty weather metadata folders, legacy BTS dirs, and unpinned caches."""
    root = Path(root).resolve()
    pinned = _collect_pinned_paths(root)
    removed: list[Path] = []
    reclaimed = 0

    for rel_base in STAGE_CACHE_ROOTS:
        allowed_root = project_path(root, rel_base.parent)
        reclaimed += _prune_generation_root(
            root,
            project_path(root, rel_base),
            allowed_root,
            pinned,
            lambda entry: (entry / "_stage.json").is_file(),
            removed,
        )

    for rel_base in WEATHER_GOLD_ROOTS:
        allowed_root = project_path(root, "data/gold")
        reclaimed += _prune_generation_root(
            root,
            project_path(root, rel_base),
            allowed_root,
            pinned,
            lambda entry: (entry / "_reuse.json").is_file(),
            removed,
        )

    silver_weather_dir = project_path(root, "data/silver/weather")
    reclaimed += _prune_generation_root(
        root,
        silver_weather_dir,
        project_path(root, "data/silver"),
        pinned,
        lambda entry: (entry / "_SUCCESS").is_file() and any(entry.rglob("*.parquet")),
        removed,
    )

    # Retain quarantine/weather directories only if they match a kept weather Silver or Gold generation.
    retained_weather_names: set[str] = set()
    for weather_parent in (
        silver_weather_dir,
        project_path(root, "data/gold/bts_origin_weather"),
        project_path(root, "data/gold/bts_origin_destination_weather"),
    ):
        if weather_parent.is_dir():
            for entry in weather_parent.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    retained_weather_names.add(entry.name)

    quarantine_weather_dir = project_path(root, "data/quarantine/weather")
    if quarantine_weather_dir.is_dir():
        for entry in sorted(quarantine_weather_dir.iterdir()):
            if entry.is_symlink() or not entry.is_dir():
                continue
            if entry.resolve() in pinned or entry.name in retained_weather_names:
                continue
            reclaimed += _safe_delete_dir(entry, project_path(root, "data/quarantine"), root)
            removed.append(entry)

    # Remove legacy pre-generation BTS directories once `_current.json` points to a valid generation.
    if _current_generation(root, "data/silver/bts_generations") is not None:
        for legacy_rel, allowed_rel in (
            ("data/silver/bts", "data/silver"),
            ("data/quarantine/bts", "data/quarantine"),
        ):
            legacy = project_path(root, legacy_rel)
            if legacy.is_dir() and not legacy.is_symlink():
                reclaimed += _safe_delete_dir(legacy, project_path(root, allowed_rel), root)
                removed.append(legacy)

    if _current_generation(root, "data/gold/bts_generations") is not None:
        legacy_gold_bts = project_path(root, "data/gold/bts")
        if legacy_gold_bts.is_dir() and not legacy_gold_bts.is_symlink():
            reclaimed += _safe_delete_dir(legacy_gold_bts, project_path(root, "data/gold"), root)
            removed.append(legacy_gold_bts)

    runs_root = project_path(root, "models/flight_delay")
    reclaimed += _prune_generation_root(
        root,
        runs_root,
        runs_root,
        pinned,
        lambda entry: (entry / "_SUCCESS").is_file(),
        removed,
    )

    return CleanupSummary(
        removed_dirs=len(removed),
        reclaimed_bytes=reclaimed,
        removed_paths=tuple(removed),
    )


def format_cleanup_summary(summary: CleanupSummary) -> str:
    """Format a concise one-line summary of reclaimed workspace disk space."""
    if summary.reclaimed_bytes >= 1024 ** 3:
        size_text = f"{summary.reclaimed_bytes / (1024 ** 3):.2f} GB"
    else:
        size_text = f"{summary.reclaimed_bytes / (1024 ** 2):.1f} MB"
    return f"  [Cleanup] Removed {summary.removed_dirs} obsolete directories ({size_text} reclaimed)."
