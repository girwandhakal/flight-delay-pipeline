"""Unit tests for failed-build rollback and reference-counted workspace cleanup."""

import os
from pathlib import Path
import stat

import pytest

from scripts.common import cleanup
from scripts.common import stage_cache as cache
from scripts.common.io import rmtree_retry, write_json

pytestmark = pytest.mark.unit


def _touch_stage(directory: Path, *, provenance: dict | None = None, mtime_ns: int | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "_stage.json", {"provenance": provenance or {}})
    if mtime_ns is not None:
        os.utime(directory, ns=(mtime_ns, mtime_ns))
    return directory


def _touch_weather_gold(directory: Path, *, metadata: dict | None = None, mtime_ns: int | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train").mkdir(exist_ok=True)
    (directory / "train/part.parquet").write_bytes(b"parquet")
    write_json(directory / "_reuse.json", {"inputs": {}})
    if metadata is not None:
        write_json(directory / "_metadata.json", metadata)
    if mtime_ns is not None:
        os.utime(directory, ns=(mtime_ns, mtime_ns))
    return directory


def test_resolve_stage_rolls_back_failed_build(tmp_path):
    base = tmp_path / "data/gold/enroute"

    def failing_build(path: Path):
        (path / "partial.txt").write_text("incomplete", encoding="utf-8")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cache.resolve_stage(base, lambda: {"k": 1}, ["train"], [], failing_build)

    assert not base.exists() or list(base.iterdir()) == []


def test_cleanup_workspace_preserves_pinned_and_newest_and_prunes_obsolete(tmp_path):
    root = tmp_path
    # Protected non-cache roots that must never be touched.
    bronze_file = root / "data/bronze/bts/2024/sample.csv"
    bronze_file.parent.mkdir(parents=True)
    bronze_file.write_text("raw", encoding="utf-8")
    ref_file = root / "data/reference/weather/airports.csv"
    ref_file.parent.mkdir(parents=True)
    ref_file.write_text("ref", encoding="utf-8")
    legacy_model = root / "models/dw/old_run/model.bin"
    legacy_model.parent.mkdir(parents=True)
    legacy_model.write_bytes(b"archive")
    (root / "data/silver/.gitkeep").parent.mkdir(parents=True, exist_ok=True)
    (root / "data/silver/.gitkeep").touch()

    # Legacy pre-generation BTS dirs + valid _current.json pointers.
    silver_curr = _touch_stage(root / "data/silver/bts_generations/gen_curr", mtime_ns=100)
    silver_old = _touch_stage(root / "data/silver/bts_generations/gen_old", mtime_ns=50)
    silver_new = _touch_stage(root / "data/silver/bts_generations/gen_new", mtime_ns=200)
    write_json(root / "data/silver/bts_generations/_current.json", "gen_curr")

    gold_curr = _touch_stage(
        root / "data/gold/bts_generations/gen_curr",
        provenance={"silver": str(silver_curr)},
        mtime_ns=100,
    )
    gold_old = _touch_stage(root / "data/gold/bts_generations/gen_old", mtime_ns=50)
    write_json(root / "data/gold/bts_generations/_current.json", "gen_curr")

    for legacy in ("data/silver/bts", "data/gold/bts", "data/quarantine/bts"):
        (root / legacy).mkdir(parents=True)
        (root / legacy / "stale.parquet").write_bytes(b"legacy")

    # Pinned weather + enroute provenance chain for an older active model run.
    ts_pinned = "20261120T000000Z"
    ts_old = "20261119T000000Z"
    ts_new = "20261125T000000Z"

    for ts, mtime, has_parquet in (
        (ts_pinned, 100, True),
        (ts_old, 50, True),
        (ts_new, 200, True),
        ("20261124T000000Z", 150, False),  # Metadata-only folder from cache hit.
    ):
        sw = root / "data/silver/weather" / ts
        sw.mkdir(parents=True)
        (sw / "_SUCCESS").touch()
        write_json(sw / "_metadata.json", {})
        if has_parquet:
            (sw / "year=2024/part.parquet").parent.mkdir(parents=True)
            (sw / "year=2024/part.parquet").write_bytes(b"weather")
        os.utime(sw, ns=(mtime, mtime))

    for ts in (ts_pinned, ts_old, ts_new):
        qw = root / "data/quarantine/weather" / ts
        qw.mkdir(parents=True)
        (qw / "bad.parquet").write_bytes(b"q")

    orig_pinned = _touch_weather_gold(root / "data/gold/bts_origin_weather" / ts_pinned, mtime_ns=100)
    orig_old = _touch_weather_gold(root / "data/gold/bts_origin_weather" / ts_old, mtime_ns=50)
    orig_new = _touch_weather_gold(root / "data/gold/bts_origin_weather" / ts_new, mtime_ns=200)

    dest_pinned = _touch_weather_gold(
        root / "data/gold/bts_origin_destination_weather" / ts_pinned,
        metadata={"parent_gold_dir": str(orig_pinned)},
        mtime_ns=100,
    )
    dest_old = _touch_weather_gold(root / "data/gold/bts_origin_destination_weather" / ts_old, mtime_ns=50)
    dest_new = _touch_weather_gold(root / "data/gold/bts_origin_destination_weather" / ts_new, mtime_ns=200)

    taf_pinned = _touch_stage(root / "data/silver/taf/taf_pinned", mtime_ns=100)
    taf_old = _touch_stage(root / "data/silver/taf/taf_old", mtime_ns=50)
    taf_new = _touch_stage(root / "data/silver/taf/taf_new", mtime_ns=200)

    sig_pinned = _touch_stage(root / "data/silver/sigmets/sig_pinned", mtime_ns=100)
    sig_new = _touch_stage(root / "data/silver/sigmets/sig_new", mtime_ns=200)

    rot_pinned = _touch_stage(
        root / "data/gold/rotation/rot_pinned",
        provenance={"destination": str(dest_pinned)},
        mtime_ns=100,
    )
    rot_old = _touch_stage(root / "data/gold/rotation/rot_old", mtime_ns=50)
    rot_new = _touch_stage(root / "data/gold/rotation/rot_new", mtime_ns=200)

    enr_pinned = _touch_stage(
        root / "data/gold/enroute/enr_pinned",
        provenance={
            "destination": str(dest_pinned),
            "rotation": str(rot_pinned / "gold"),
            "aviation": {
                "taf": str(taf_pinned / "table"),
                "sigmets": str(sig_pinned / "table"),
            },
        },
        mtime_ns=100,
    )
    enr_old = _touch_stage(root / "data/gold/enroute/enr_old", mtime_ns=50)
    enr_new = _touch_stage(root / "data/gold/enroute/enr_new", mtime_ns=200)
    aborted_enr = root / "data/gold/enroute/aborted_dir"
    aborted_enr.mkdir(parents=True)
    (aborted_enr / "junk.bin").write_bytes(b"aborted")

    # Active run in results.json (older) + newest completed run + aborted run.
    run_active = root / "models/flight_delay/run_active"
    run_active.mkdir(parents=True)
    (run_active / "_SUCCESS").touch()
    write_json(run_active / "run.json", {"gold": "data/gold/enroute/enr_pinned"})
    os.utime(run_active, ns=(100, 100))

    run_newest = root / "models/flight_delay/run_newest"
    run_newest.mkdir(parents=True)
    (run_newest / "_SUCCESS").touch()
    write_json(run_newest / "run.json", {"gold": "data/gold/enroute/enr_new"})
    os.utime(run_newest, ns=(200, 200))

    run_aborted = root / "models/flight_delay/run_aborted"
    run_aborted.mkdir(parents=True)
    (run_aborted / "partial.bin").write_bytes(b"x")

    write_json(root / "results/results.json", {"flight_delay": {"run_id": "run_active"}})

    summary = cleanup.cleanup_workspace(root)
    assert summary.removed_dirs > 0
    assert summary.reclaimed_bytes > 0

    # Preserved paths.
    assert bronze_file.is_file()
    assert ref_file.is_file()
    assert legacy_model.is_file()
    assert (root / "data/silver/.gitkeep").is_file()
    assert silver_curr.is_dir() and silver_new.is_dir()
    assert gold_curr.is_dir()
    assert orig_pinned.is_dir() and orig_new.is_dir()
    assert dest_pinned.is_dir() and dest_new.is_dir()
    assert (root / "data/silver/weather" / ts_pinned).is_dir()
    assert (root / "data/silver/weather" / ts_new).is_dir()
    assert (root / "data/quarantine/weather" / ts_pinned).is_dir()
    assert (root / "data/quarantine/weather" / ts_new).is_dir()
    assert taf_pinned.is_dir() and taf_new.is_dir()
    assert sig_pinned.is_dir() and sig_new.is_dir()
    assert rot_pinned.is_dir() and rot_new.is_dir()
    assert enr_pinned.is_dir() and enr_new.is_dir()
    assert run_active.is_dir() and run_newest.is_dir()

    # Deleted paths.
    assert not silver_old.exists()
    assert not gold_old.exists()
    assert not (root / "data/silver/bts").exists()
    assert not (root / "data/gold/bts").exists()
    assert not (root / "data/quarantine/bts").exists()
    assert not orig_old.exists()
    assert not dest_old.exists()
    assert not (root / "data/silver/weather" / ts_old).exists()
    assert not (root / "data/silver/weather/20261124T000000Z").exists()
    assert not (root / "data/quarantine/weather" / ts_old).exists()
    assert not taf_old.exists()
    assert not rot_old.exists()
    assert not enr_old.exists()
    assert not aborted_enr.exists()
    assert not run_aborted.exists()


def test_rmtree_retry_clears_readonly_files(tmp_path):
    target = tmp_path / "readonly_tree/nested"
    target.mkdir(parents=True)
    locked = target / "locked.bin"
    locked.write_bytes(b"data")
    os.chmod(locked, stat.S_IREAD)
    rmtree_retry(tmp_path / "readonly_tree")
    assert not (tmp_path / "readonly_tree").exists()
