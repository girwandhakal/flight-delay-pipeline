"""Completed Gold should skip preparation without hiding stale or incomplete data."""

from unittest.mock import patch

from scripts import weather_cache as cache
from scripts.origin_weather import build_features as origin
from scripts.destination_weather import build_features as destination
from pipeline import main as pipeline


def test_discovery_and_invalidation(tmp_path):
    config = {"splits": {"train": {}, "val": {}, "test": {}}}
    inputs = {"splits": config["splits"], "policy": {"lag": 0}}
    for parent, variant in ((None, "bts_origin_weather"),
                            (tmp_path / "parent", "bts_origin_destination_weather")):
        directory = tmp_path / "data/gold" / variant / "001"
        for split in config["splits"]:
            folder = directory / split
            folder.mkdir(parents=True)
            (folder / "_SUCCESS").touch()
            (folder / "part.parquet").write_bytes(b"fixture")
        metadata = directory / "_metadata.json"
        metadata.write_text("{}")
        cache.save_gold(directory, inputs)
        # A newer failed build must not hide an older complete build.
        incomplete = directory.parent / "002"
        incomplete.mkdir()
        (incomplete / "_reuse.json").write_bytes((directory / "_reuse.json").read_bytes())
        with patch.object(cache, "input_identity", return_value=inputs):
            assert cache.find_gold(tmp_path, {}, {}, config, parent) == directory.resolve()
            with patch.object(cache, "input_identity", return_value={**inputs, "policy": {"lag": 15}}):
                assert cache.find_gold(tmp_path, {}, {}, config, parent) is None
            metadata.write_text('{"changed": true}')
            assert cache.find_gold(tmp_path, {}, {}, config, parent) is None
            metadata.write_text("{}")
            output = directory / "train/part.parquet"
            output.write_bytes(b"changed")
            assert cache.find_gold(tmp_path, {}, {}, config, parent) is None
            cache.save_gold(directory, inputs)
            (directory / "test/_SUCCESS").unlink()
            assert cache.find_gold(tmp_path, {}, {}, config, parent) is None


def test_cache_hits_do_not_start_spark(tmp_path):
    for builder, args in ((origin, ()), (destination, (tmp_path / "parent",))):
        with (patch.object(builder, "load_config", return_value=({}, {}, {})),
              patch.object(cache, "find_gold", return_value=tmp_path / "cached"),
              patch.object(pipeline.bts_transform, "get_spark_session", side_effect=AssertionError("Unexpected Spark"))):
            kwargs = {"publish_results": False} if builder is origin else {}
            assert builder.main(*args, **kwargs) == (tmp_path / "cached",) * 2


def test_bts_only_reuses_gold_before_training():
    with (patch.object(pipeline, "load_config", return_value=({}, {}, {})),
          patch.object(pipeline, "features_current", return_value=True),
          patch.object(pipeline, "run_stage", return_value=0) as stage):
        assert pipeline.main(["--bts-only"]) == 0
        assert [call.args[2] for call in stage.call_args_list] == [entry[2] for entry in pipeline.STAGES[3:]]
