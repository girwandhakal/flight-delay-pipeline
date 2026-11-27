"""Exercise generation integrity and the production dependency keys with tiny local files."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.aircraft_rotation import run as rotation
from scripts.common import stage_cache as cache
from scripts.common.config import read_config, weather_reference_path
from scripts.common.io import sha256, write_json
from scripts.enroute_weather import run as aviation
from scripts.enroute_weather.download import acquire
from scripts.origin_weather import clean


def partition(path, value=b"fixture"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "part.parquet").write_bytes(value)
    (path / "_SUCCESS").touch()


def test_weather_reference_paths_follow_study_dates(tmp_path):
    config = {
        "origin_weather": {"reference_directory": "refs/origin"},
        "destination_weather": {"reference_directory": "refs/destination"},
        "splits": {"train": {"start_date": "2024-01-01"}, "test": {"end_date": "2025-01-31"}},
    }
    seed = tmp_path / "refs/origin"
    seed.mkdir(parents=True)
    write_json(seed / "sources.json", {"scope": ["2024-01-01", "2025-01-31"]})
    original_destination = weather_reference_path(tmp_path, config, "destination")
    assert weather_reference_path(tmp_path, config, "origin") == seed

    config["splits"]["test"]["end_date"] = "2026-01-31"
    expanded = weather_reference_path(tmp_path, config, "origin")
    assert expanded == seed.with_name("origin-2024-01-01-to-2026-01-31")
    assert weather_reference_path(tmp_path, config, "destination") != original_destination

    config["splits"]["train"]["start_date"] = "2024-02-01"
    config["splits"]["test"]["end_date"] = "2025-01-30"
    assert weather_reference_path(tmp_path, config, "origin") not in (seed, expanded)
    assert (seed / "sources.json").is_file()


def test_incomplete_generation_and_changed_output(tmp_path):
    source = {"input": "A"}
    base = tmp_path / "gold"

    def resolve(builder):
        return cache.resolve_stage(base, source.copy, ["train"], [], builder)

    original = resolve(lambda path: partition(path / "train"))
    assert resolve(lambda _: pytest.fail("warm builder")) == original

    (original / "train/_SUCCESS").unlink()
    repaired = resolve(lambda path: partition(path / "train"))
    assert repaired != original
    (repaired / "train/part.parquet").write_bytes(b"changed output")
    with pytest.raises(ValueError, match="Changed stage output"):
        cache.consumed(repaired, ["train"])
    assert resolve(lambda path: partition(path / "train")) != repaired


def test_aviation_and_rotation_dependency_matrix(tmp_path, monkeypatch):
    root = tmp_path
    source_root = Path(rotation.__file__).resolve().parents[2]
    shutil.copytree(source_root / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    config = read_config(source_root / "config.json")
    references = weather_reference_path(root, config, "destination")
    references.mkdir(parents=True)
    (references / "airports.csv").write_text("Origin,time_zone,valid_from,valid_to,icao\nAAA,UTC,2020,2030,KAAA\n")
    (references / "sources.json").write_text("{}")
    raw = root / "flights.csv"
    raw.write_text("tail A")
    taf, sigmet = root / "taf.zip", root / "sigmet.kml"
    for archive in (taf, sigmet):
        archive.write_bytes(b"original")
        write_json(archive.with_suffix(archive.suffix + ".json"), {"url": "fixture", "sha256": sha256(archive)})

    def download(*args, **kwargs):
        return tuple(dict([acquire(root, archive.name, "fixture", None)]) for archive in (taf, sigmet))

    monkeypatch.setattr(aviation, "download", download)
    monkeypatch.setattr(aviation.clean_taf, "__file__", str(root / "scripts/enroute_weather/clean_taf.py"))
    destination = root / "destination" / "A"
    for split in config["splits"]:
        partition(destination / split)
    write_json(destination / "_metadata.json", {})
    cache.save_gold(destination, {"splits": config["splits"]})
    built = []

    # Keep the real lookup, source checks, manifests and partition fingerprints.
    # Substitute only the Spark actions behind each verified cache miss.
    def resolve(base, inputs, partitions, sidecars, transform):
        def build(path):
            built.append(base.name)
            for name in partitions:
                partition(path / name, path.name.encode())
            for name in sidecars:
                value = {"stations": ["KAAA"], "share": 1.0} if name == "stations.json" else {}
                write_json(path / name, value)
        return cache.resolve_stage(base, inputs, partitions, sidecars, build)

    monkeypatch.setattr(aviation, "resolve_stage", resolve)
    monkeypatch.setattr(rotation, "resolve_stage", resolve)
    spark = MagicMock()
    def run():
        built.clear()
        prepared = aviation.prepare(spark, root, config, destination)
        rotation_gold, _, _ = rotation.prepare_rotation(spark, root, config, references, destination, ([raw], [raw]))
        aviation.prepare_gold(spark, root, rotation_gold, destination, prepared)
        return set(built)

    assert run() == {"taf_stations", "taf", "sigmets", "rotation", "enroute"}
    config["modeling"]["random_seed"] += 1
    assert run() == set()
    with (root / "scripts/enroute_weather/clean_taf.py").open("a") as stream:
        stream.write("# changed TAF parser")
    assert run() == {"taf", "enroute"}
    taf.write_bytes(b"replacement archive")
    with pytest.raises(ValueError, match="failed verification"):
        run()
    write_json(taf.with_suffix(".zip.json"), {"url": "fixture", "sha256": sha256(taf)})
    assert run() == {"taf", "enroute"}

    sibling = destination.with_name("B")
    shutil.copytree(destination, sibling, copy_function=shutil.copy2)
    (sibling / "test/part.parquet").write_bytes(b"new test cohort")
    cache.save_gold(sibling, {"splits": config["splits"]})
    destination = sibling
    assert run() == {"rotation", "enroute"}
    rotated = sorted((root / "data/gold/rotation").glob("*/gold/test/part.parquet"), key=lambda path: path.stat().st_mtime_ns)[-1]
    rotated.write_bytes(b"changed rotation output")
    with pytest.raises(ValueError, match="Changed stage output"):
        cache.consumed(rotated.parents[2], ("gold/test",))
    raw.write_text("new tail assignment")
    assert run() == {"rotation", "enroute"}
    monkeypatch.setattr(rotation, "BTS_FEATURE_COLUMNS", list(reversed(rotation.BTS_FEATURE_COLUMNS)))
    assert run() == {"rotation", "enroute"}
    with (root / "scripts/enroute_weather/build_features.py").open("a") as stream:
        stream.write("# revised join")
    assert run() == {"enroute"}
    assert run() == set()


def test_bts_and_airport_weather_branches(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.bts import build_features, clean
    from scripts.bts import cache as bts
    from scripts.bts.block_time import BLOCK_TIME_SIDECAR
    from scripts.bts.recent_delays import TAXI_BASELINE_FILE

    root = tmp_path
    project = Path(rotation.__file__).resolve().parents[2]
    shutil.copytree(project / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    config = read_config(project / "config.json")
    config["datasets"]["bts"].update(start_month="2024-01", end_month="2024-01")
    write_json(root / "config.json", config)
    bronze = root / config["datasets"]["bts"]["output_directory"] / "2024/01"
    bronze.mkdir(parents=True)
    (bronze / "flights.csv").write_text("flight")
    settings = dict(config["datasets"]["weather"])
    settings["reference_directory"] = config["origin_weather"]["reference_directory"]
    for name in ("reference_directory", "output_directory"):
        settings[name] = root / settings[name]
    origin = config["origin_weather"]["reference_directory"]
    destination = config["destination_weather"]["reference_directory"]
    shutil.copytree(project / origin, root / origin)
    origin_manifest = root / origin / "sources.json"
    sources = json.loads(origin_manifest.read_text(encoding="utf-8"))
    sources["scope"] = [config["splits"]["train"]["start_date"], config["splits"]["test"]["end_date"]]
    write_json(origin_manifest, sources)
    shutil.copytree(project / destination, weather_reference_path(root, config, "destination"))

    weather = settings["output_directory"]
    names = [f"2024/LCD_USW0000000{i}_2024.csv" for i in (1, 2)]
    for name in names:
        (weather / name).parent.mkdir(parents=True, exist_ok=True)
        (weather / name).write_bytes(b"station")
    files = {name: {"year": 2024, "station_id": name.split("_")[1], "time_zone": "UTC",
                    "status": "downloaded", "bytes": 7} for name in names}
    write_json(weather / "manifest.json", {"complete": True, "files": files})
    calls = []

    def silver(spark, *, silver_dir, **kwargs):
        calls.append("silver")
        partition(silver_dir / "2024/01")
        return pd.DataFrame([{"retained_records": 1}])

    def gold(spark, *, gold_dir, **kwargs):
        calls.append("gold")
        for split in config["splits"]:
            partition(gold_dir / split, gold_dir.name.encode())
        write_json(gold_dir / TAXI_BASELINE_FILE, [])
        write_json(gold_dir / BLOCK_TIME_SIDECAR, {})
        write_json(gold_dir / "_features.json", {})
        return pd.DataFrame([{"total_flights": 1}])

    monkeypatch.setattr(clean, "clean_bts_data", silver)
    monkeypatch.setattr(build_features, "build_features", gold)
    def run():
        calls.clear()
        for layer in ("silver", "gold"):
            bts.prepare(None, root, layer, False)
        assert bts.features_current(root, config)
        parent = None
        for name, role in zip(names, ("origin", "destination")):
            policy = config["origin_weather"]
            found = cache.find_gold(root, settings, policy, config, parent)
            if found:
                parent = found
                continue
            calls.append(role)
            inputs = cache.input_identity(root, settings, policy, config, parent, [name])
            variant = "bts_origin_weather" if role == "origin" else "bts_origin_destination_weather"
            base = root / "data/gold" / variant
            path = base / str(len(list(base.glob("*"))))
            for split in config["splits"]:
                partition(path / split, path.name.encode())
            write_json(path / "_metadata.json", {})
            cache.save_gold(path, inputs)
            parent = path
        return calls.copy()

    assert run() == ["silver", "gold", "origin", "destination"]
    frozen = bts.active(root, "gold")
    pinned = cache.consumed(frozen, config["splits"])
    assert run() == []
    with (root / "scripts/bts/recent_delays.py").open("a") as stream:
        stream.write("# changed BTS feature policy")
    assert run() == ["gold", "origin", "destination"]
    assert run() == []
    (bts.active(root, "gold") / BLOCK_TIME_SIDECAR).write_text("changed")
    assert not bts.features_current(root, config)
    assert run() == ["gold", "origin", "destination"]
    assert run() == []
    assert bts.active(root, "gold") != frozen
    assert cache.consumed(frozen, config["splits"]) == pinned
    with (root / "scripts/destination_weather/build_features.py").open("a") as stream:
        stream.write("# changed destination matching")
    assert run() == ["destination"]
    with (root / "scripts/enroute_weather/build_features.py").open("a") as stream:
        stream.write("# changed en-route matching")
    assert run() == []
    (weather / names[1]).write_bytes(b"replacement station")
    files[names[1]]["bytes"] = (weather / names[1]).stat().st_size
    write_json(weather / "manifest.json", {"complete": True, "files": files})
    assert run() == ["destination"]
    (bts.active(root, "gold") / TAXI_BASELINE_FILE).write_text("changed")
    assert not bts.features_current(root, config)
    assert run() == ["gold", "origin", "destination"]


def test_warm_aviation_skips_spark_actions_and_preserves_splits(tmp_path, spark, monkeypatch):
    from datetime import date
    from zipfile import ZipFile

    project = Path(rotation.__file__).resolve().parents[2]
    root = tmp_path
    config = read_config(project / "config.json")
    reference = config["destination_weather"]["reference_directory"]
    shutil.copytree(project / reference, weather_reference_path(root, config, "destination"))
    shutil.copytree(project / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    destination = root / "destination"
    taf = root / "taf.zip"
    sigmet = root / "sigmet.kml"
    with ZipFile(taf, "w") as archive:
        archive.writestr("TAFJFK_202401050900.txt", "TAF KJFK 050900Z 0509/0609 22004KT P6SM SCT050=")
    sigmet.write_text('<kml xmlns="http://www.opengis.net/kml/2.2"/>')
    monkeypatch.setattr(aviation, "download", lambda *a, **kw: ({taf: sha256(taf)}, {sigmet: sha256(sigmet)}))
    splits = config["splits"]
    for split, day in zip(splits, (date(2024, 1, 5), date(2024, 10, 5), date(2025, 1, 5))):
        flight = spark.createDataFrame([(split, day, 0, "ORD", "JFK", 120.0, "1000", "1300", 200.0)],
                    "flight_id string, FlightDate date, ArrDel15 int, Origin string, Dest string, "
                    "CRSElapsedTime double, CRSDepTime string, CRSArrTime string, destination_cloud_ceiling_m double")
        flight.write.parquet(str(destination / split))
    write_json(destination / "_metadata.json", {})
    cache.save_gold(destination, {"splits": splits})

    def build_rotation_fixture(path):
        from pyspark.sql import functions as F
        from scripts.aircraft_rotation.build_features import ROTATION_NUMERIC, ROTATION_FLAGS

        for split in splits:
            rows = spark.read.parquet(str(destination / split))
            for name in ROTATION_NUMERIC:
                rows = rows.withColumn(name, F.lit(12.0 if split != "test" else None).cast("double"))
            for name in ROTATION_FLAGS:
                rows = rows.withColumn(name, F.lit(int(split == "test")))
            rows.write.parquet(str(path / "gold" / split))

    rotation_dir = cache.resolve_stage(root / "rotation", lambda: {},
                                       tuple(f"gold/{split}" for split in splits), (), build_rotation_fixture)
    cold = aviation.prepare(spark, root, config, destination)
    gold, coverage = aviation.prepare_gold(spark, root, rotation_dir / "gold", destination, cold)
    assert coverage["train"]["taf_observed"] == 1
    for split in splits:
        stored = spark.read.parquet(str(gold / split))
        assert [(row.flight_id, row.ArrDel15) for row in stored.select("flight_id", "ArrDel15").collect()] == [(split, 0)]

    test_row = spark.read.parquet(str(gold / "test")).first()
    assert test_row.rotation_prior_arrival_delay_minutes is None
    assert test_row[rotation.ROTATION_FLAGS[1]] == 1

    # Verified generations can be reused without another Spark transformation.
    warm_spark = MagicMock()
    warm_spark.read.parquet.side_effect = AssertionError("unexpected warm Spark read")
    warm_spark.createDataFrame.side_effect = AssertionError("unexpected warm Spark writer")
    warm = aviation.prepare(warm_spark, root, config, destination)
    assert warm["tables"] == cold["tables"]
    assert aviation.prepare_gold(warm_spark, root, rotation_dir / "gold", destination, warm) == (gold, coverage)


def test_weather_gold_and_silver_cache_reuse_and_invalidation(tmp_path):
    # Verify Gold stage discovery returns directory on exact match and None on policy mismatch or missing success marker
    config = {"splits": {"train": {}, "val": {}, "test": {}}}
    inputs = {"splits": config["splits"], "policy": {"max_age_minutes": 120}}
    directory = tmp_path / "data/gold/bts_origin_weather/001"
    for split in config["splits"]:
        folder = directory / split
        folder.mkdir(parents=True)
        (folder / "_SUCCESS").touch()
        (folder / "part.parquet").write_bytes(b"fixture")
    metadata = directory / "_metadata.json"
    metadata.write_text("{}")
    cache.save_gold(directory, inputs)

    with patch.object(cache, "input_identity", return_value=inputs):
        assert cache.find_gold(tmp_path, {}, {}, config, None) == directory.resolve()
        # Ensure policy change invalidates the cached Gold directory
        with patch.object(cache, "input_identity", return_value={**inputs, "policy": {"max_age_minutes": 90}}):
            assert cache.find_gold(tmp_path, {}, {}, config, None) is None
        (directory / "test/_SUCCESS").unlink()
        assert cache.find_gold(tmp_path, {}, {}, config, None) is None

    # Verify Silver cleaning reuse skips CSV reads when file fingerprints match
    raw = tmp_path / "bronze/2024/LCD_USW00094846_2024.csv"
    raw.parent.mkdir(parents=True)
    raw.write_text("fixture")
    silver = tmp_path / "silver/first"
    silver.mkdir(parents=True)
    output = silver / "part-000.parquet"
    output.write_bytes(b"fixture")
    (silver / "_SUCCESS").touch()
    manifest = {"complete": True, "files": {"fixture": {
        "status": "downloaded", "station_id": "USW00094846",
        "year": 2024, "time_zone": "America/Chicago",
    }}}
    cleaning_inputs = {
        "files": clean.file_fingerprints([raw]),
        "stations": [["USW00094846", 2024, "America/Chicago"]],
        "code": {name: clean.sha256(Path(clean.__file__).with_name(name))
                 for name in ("clean.py", "measurements.py", "download.py")},
        "spark": "3.5.7",
    }
    meta = {
        "quality": {"retained_observations": 1},
        "station_quality": [],
        "source_contract": {},
        "cleaning_cache": {
            "inputs": cleaning_inputs,
            "outputs": clean.file_fingerprints([output]),
        },
    }
    (silver / "_metadata.json").write_text(json.dumps(meta))

    spark_mock = MagicMock()
    spark_mock.version = "3.5.7"
    spark_mock.createDataFrame.side_effect = RuntimeError("cleaning started")
    result = clean.clean_weather(
        spark_mock, tmp_path / "bronze", tmp_path / "silver/next",
        tmp_path / "quarantine", manifest, tmp_path / "reports",
    )
    spark_mock.read.parquet.assert_called_once_with(str(silver), pathGlobFilter="*.parquet")
