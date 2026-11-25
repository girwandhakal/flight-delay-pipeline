"""Stage 2: Match point-in-time NOAA surface weather observations at origin airports."""

from scripts.common.context import PipelineContext
from scripts.common.progress import pipeline_stage
from . import build_features


def run(ctx: PipelineContext) -> None:
    """Build or reuse origin-weather Gold and store its path in `ctx`."""
    with pipeline_stage(2, 6, "Origin Surface Weather", timings=ctx.timings):
        origin_gold, _ = build_features.main(
            publish_results=False,
            spark=ctx.spark,
            bts_files=ctx.sources.get("bts_files"),
        )
        ctx.origin_gold_dir = origin_gold
        ctx.gold_dir = origin_gold
