"""Project root resolution and config.json loading helpers."""

import json
from pathlib import Path
from typing import Any, Union

ROOT = Path(__file__).resolve().parents[2]


def project_path(root: Path, name: Union[str, Path]) -> Path:
    """Resolve a relative path inside the project root, rejecting path traversal."""
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path must stay inside the project: {name}")
    return resolved


def load_config(
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate dataset, policy, and split configuration from config.json."""
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    settings = config["datasets"]["weather"].copy()
    if settings["product"] != "lcdv2":
        raise ValueError("Only NOAA LCDv2 is supported.")
    for key in ("output_directory", "reference_directory"):
        settings[key] = project_path(root, settings[key])
    policy = {"max_age_minutes": config["origin_weather"]["max_age_minutes"]}
    if policy["max_age_minutes"] <= 0:
        raise ValueError("Weather age must be positive.")
    return settings, policy, config
