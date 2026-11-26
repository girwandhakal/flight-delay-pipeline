"""Shared pipeline execution context passed sequentially between stages."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pyspark.sql import SparkSession

from .config import ROOT, load_config
from .io import sha256


@dataclass
class PipelineContext:
    """Mutable state carried sequentially through stages 1..6 of pipeline/main.py."""

    root: Path
    config: dict[str, Any]
    settings: dict[str, Any]
    run_id: str
    spark: Optional[SparkSession] = None
    gold_dir: Optional[Path] = None
    origin_gold_dir: Optional[Path] = None
    destination_gold_dir: Optional[Path] = None
    rotation_gold_dir: Optional[Path] = None
    coverage: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=lambda: {"sha256": {}})
    timings: dict[str, float] = field(default_factory=dict)

    @classmethod
    def create(cls, root: Path = ROOT) -> "PipelineContext":
        """Initialize configuration, root config digest, and a unique run ID."""
        settings, _, config = load_config(root)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
        return cls(
            root=root,
            config=config,
            settings=settings,
            run_id=run_id,
            sources={"sha256": {"config.json": sha256(root / "config.json")}},
        )
