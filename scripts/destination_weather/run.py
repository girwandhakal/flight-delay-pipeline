"""Stage 3: Match point-in-time NOAA surface weather observations at destination airports."""

from scripts.common.context import PipelineContext
from scripts.common.progress import pipeline_stage
from . import build_features


def run(ctx: PipelineContext) -> None:
    """Build or reuse destination-weather Gold on top of origin-weather Gold."""
    with pipeline_stage(3, 6, "Destination Surface Weather", timings=ctx.timings):
        destination_gold, _ = build_features.main(
            ctx.origin_gold_dir,
            spark=ctx.spark,
            bts_files=ctx.sources.get("bts_files"),
        )
        ctx.destination_gold_dir = destination_gold
        ctx.gold_dir = destination_gold
