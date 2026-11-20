"""Exercise generation integrity and the production dependency keys with tiny local files."""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import weather_cache as cache
from scripts.aircraft_rotation import run as rotation
from scripts.enroute_weather import run as aviation
from scripts.enroute_weather.download import acquire
from scripts.origin_weather.download import sha256, write_json


def partition(path, value=b"fixture"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "part.parquet").write_bytes(value)
    (path / "_SUCCESS").touch()


def test_incomplete_generation_and_source_race(tmp_path):
    source = {"input": "A"}
    base = tmp_path / "gold"
    def resolve(builder):
        return cache.resolve_stage(base, source.copy, ["train"], [], builder)

    original = resolve(lambda path: partition(path / "train"))
    assert resolve(lambda _: pytest.fail("warm builder")) == original

    def race(path):
        partition(path / "train")
        source["input"] = "C"

    source["input"] = "B"
    with pytest.raises(ValueError, match="inputs changed"):
        resolve(race)
    assert len(list(base.glob("*/_stage.json"))) == 1
    source["input"] = "A"
    assert resolve(lambda _: pytest.fail("lost prior generation")) == original
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
    config = json.loads((source_root / "config.json").read_text())
    references = root / config["destination_weather"]["reference_directory"]
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
    monkeypatch.setattr(rotation, "ROOT", root)
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
        prepared = aviation.prepare(spark, root, config, destination, "fresh-model-run")
        report = rotation.prepare_rotation(spark, config, references, destination, [raw], root / "models")
        aviation.prepare_gold(spark, root, Path(report["training_gold"]), destination, prepared)
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
    assert run() == {"enroute"}
    raw.write_text("new tail assignment")
    assert run() == {"rotation", "enroute"}
    monkeypatch.setattr(rotation.bts_train, "INPUT_FEATURES", list(reversed(rotation.bts_train.INPUT_FEATURES)))
    assert run() == {"rotation", "enroute"}
    with (root / "scripts/enroute_weather/build_features.py").open("a") as stream:
        stream.write("# revised join")
    assert run() == {"enroute"}
    assert run() == set()


def test_bts_and_airport_weather_branches(tmp_path, monkeypatch):
    import pandas as pd
    from scripts.bts import build_features, clean
    from scripts.bts import cache as bts
    from scripts.bts.recent_delays import TAXI_BASELINE_FILE

    root = tmp_path
    project = Path(rotation.__file__).resolve().parents[2]
    shutil.copytree(project / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    config = json.loads((project / "config.json").read_text())
    config["datasets"]["bts"].update(start_month="2024-01", end_month="2024-01")
    write_json(root / "config.json", config)
    bronze = root / config["datasets"]["bts"]["output_directory"] / "2024/01"
    bronze.mkdir(parents=True)
    (bronze / "flights.csv").write_text("flight")
    settings = dict(config["datasets"]["weather"])
    for name in ("reference_directory", "output_directory"):
        settings[name] = root / settings[name]
    for directory in (config["datasets"]["weather"]["reference_directory"],
                      config["destination_weather"]["reference_directory"]):
        shutil.copytree(project / directory, root / directory)

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
    with (root / "scripts/bts/upstream.py").open("a") as stream:
        stream.write("# changed Tier 1 policy")
    assert run() == ["gold", "origin", "destination"]
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
    config = json.loads((project / "config.json").read_text())
    reference = config["destination_weather"]["reference_directory"]
    shutil.copytree(project / reference, root / reference)
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

    def copy_development(path):
        for split in ("train", "val"):
            shutil.copytree(destination / split, path / "gold" / split)
    rotation_dir = cache.resolve_stage(root / "rotation", lambda: {}, ("gold/train", "gold/val"), (), copy_development)
    cold = aviation.prepare(spark, root, config, destination, "model-A")
    gold, coverage = aviation.prepare_gold(spark, root, rotation_dir / "gold", destination, cold)
    assert coverage["train"]["taf_observed"] == 1
    for split in splits:
        stored = spark.read.parquet(str(gold / split))
        assert [(row.flight_id, row.ArrDel15) for row in stored.select("flight_id", "ArrDel15").collect()] == [(split, 0)]

    # Any station aggregation, join, coverage scan or writer would need one
    # of these entry points. A new model ID should reach none of them.
    warm_spark = MagicMock()
    warm_spark.read.parquet.side_effect = AssertionError("unexpected warm Spark read")
    warm_spark.createDataFrame.side_effect = AssertionError("unexpected warm Spark writer")
    warm = aviation.prepare(warm_spark, root, config, destination, "model-B")
    assert warm["tables"] == cold["tables"]
    assert aviation.prepare_gold(warm_spark, root, rotation_dir / "gold", destination, warm) == (gold, coverage)
