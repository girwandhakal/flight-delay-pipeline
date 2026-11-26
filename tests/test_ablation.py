"""Unit tests for controlled feature ablation (Issue #5)."""

import pytest

from pipeline.main import parse_args
from scripts.common.spark import spark_session
from scripts.modeling.train import ABLATION_GROUPS, build_preprocessor


def test_ablation_groups_definition() -> None:
    """Verify the four progressive ablation groups and their flags."""
    assert tuple(ABLATION_GROUPS) == (
        "bts_only",
        "bts_surface_weather",
        "bts_surface_rotation",
        "full",
    )
    assert ABLATION_GROUPS["bts_only"] == {"surface_weather": False, "rotation": False, "enroute": False}
    assert ABLATION_GROUPS["bts_surface_weather"] == {"surface_weather": True, "rotation": False, "enroute": False}
    assert ABLATION_GROUPS["bts_surface_rotation"] == {"surface_weather": True, "rotation": True, "enroute": False}
    assert ABLATION_GROUPS["full"] == {"surface_weather": True, "rotation": True, "enroute": True}


def test_build_preprocessor_feature_groups_and_invalid() -> None:
    """Verify build_preprocessor accepts all ablation groups and rejects unknown ones."""
    dims = {}
    with spark_session("test_ablation"):
        for group in ABLATION_GROUPS:
            pipeline = build_preprocessor(feature_group=group)
            assembler = pipeline.getStages()[-1]
            dims[group] = len(assembler.getInputCols())
        assert dims["bts_only"] < dims["bts_surface_weather"] < dims["bts_surface_rotation"] < dims["full"]
        with pytest.raises(ValueError, match="Unknown feature_group"):
            build_preprocessor(feature_group="invalid_group")


def test_pipeline_main_ablation_flag() -> None:
    """Verify --ablation CLI flag parsing in pipeline/main.py."""
    assert parse_args([]).ablation is False
    assert parse_args(["--ablation"]).ablation is True
