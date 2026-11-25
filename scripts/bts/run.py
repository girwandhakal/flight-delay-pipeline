"""Stage 1: Acquire, clean, and build schedule and baseline features for BTS flights."""

from scripts.common.context import PipelineContext
from scripts.common.progress import pipeline_stage, stage
from . import build_features, clean, download
from .cache import active, current_partitions, features_current


def run(ctx: PipelineContext) -> None:
    """Reuse verified BTS Silver/Gold generations or rebuild them sequentially."""
    with pipeline_stage(1, 6, "BTS Data & Schedule Features", timings=ctx.timings):
        if features_current(ctx.root, ctx.config):
            print("  [Cache HIT] Reusing verified BTS Silver and Gold generations.", flush=True)
        else:
            with stage("Verify or download monthly BTS CSV files"):
                status = download.main()
                if status:
                    raise RuntimeError(f"BTS download failed with exit code {status}.")
            with stage("Clean raw BTS CSVs into BTS Silver Parquet"):
                clean.clean_bts_data(ctx.spark, publish_results=False)
            with stage("Build BTS Gold features and chronological splits"):
                build_features.build_features(ctx.spark, publish_results=False)

        bts_gold = active(ctx.root, "gold")
        ctx.gold_dir = bts_gold
        # Cache BTS partition fingerprints once so downstream weather stages skip re-hashing.
        if bts_gold.parent.name == "bts_generations":
            ctx.sources["bts_files"] = current_partitions(ctx.root, ctx.config, bts_gold)
