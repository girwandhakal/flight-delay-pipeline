"""Project root resolution and config.json loading helpers."""

import json
from calendar import monthrange
from datetime import date
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


def weather_reference_path(root: Path, config: dict[str, Any], role: str) -> Path:
    """Resolve a reference bundle for the current study dates."""
    directory = project_path(root, config[f"{role}_weather"]["reference_directory"])
    splits = config["splits"]
    start = date.fromisoformat(splits["train"]["start_date"]).isoformat()
    end = date.fromisoformat(splits["test"]["end_date"]).isoformat()
    if role == "origin":
        manifest = directory / "sources.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8"))["scope"] == [start, end]:
            return directory
    # Destination coverage includes an envelope computed from the flight cohort.
    return project_path(root, directory.with_name(f"{directory.name}-{start}-to-{end}").relative_to(root.resolve()))


def read_config(path: Path) -> dict[str, Any]:
    """Read config and resolve chronological whole-month splits when configured."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in config:
        return config

    from scripts.bts.download import iter_months

    names = ("train", "val", "test")
    percentages = config["split_percentages"]
    if (not isinstance(percentages, dict) or set(percentages) != set(names)
            or any(type(value) is not int or value <= 0 for value in percentages.values())
            or sum(percentages.values()) != 100):
        raise ValueError("split_percentages must give train, val, and test positive integers totaling 100.")

    bts = config["datasets"]["bts"]
    months = list(iter_months(bts["start_month"], bts["end_month"]))
    if len(months) < len(names):
        raise ValueError("Automatic splits require at least three BTS months.")

    quotas = [len(months) * percentages[name] for name in names]
    counts = [quota // 100 for quota in quotas]
    # Give leftover months to the largest fractional shares, breaking ties in split order.
    ranking = sorted(range(len(names)), key=lambda index: quotas[index] % 100, reverse=True)
    for index in ranking[:len(months) - sum(counts)]:
        counts[index] += 1
    for index, count in enumerate(counts):
        if count == 0:
            donor = max((i for i in range(len(names)) if counts[i] > 1),
                        key=lambda i: 100 * counts[i] - quotas[i])
            counts[donor] -= 1
            counts[index] = 1

    splits = {}
    offset = 0
    for name, count in zip(names, counts):
        first_year, first_month = months[offset]
        last_year, last_month = months[offset + count - 1]
        splits[name] = {
            "start_date": date(first_year, first_month, 1).isoformat(),
            "end_date": date(last_year, last_month, monthrange(last_year, last_month)[1]).isoformat(),
        }
        offset += count
    config["splits"] = splits
    return config


def load_config(
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate dataset, policy, and split configuration from config.json."""
    config = read_config(root / "config.json")
    settings = config["datasets"]["weather"].copy()
    if settings["product"] != "lcdv2":
        raise ValueError("Only NOAA LCDv2 is supported.")
    settings["output_directory"] = project_path(root, settings["output_directory"])
    settings["reference_directory"] = weather_reference_path(root, config, "origin")
    policy = {"max_age_minutes": config["origin_weather"]["max_age_minutes"]}
    if policy["max_age_minutes"] <= 0:
        raise ValueError("Weather age must be positive.")
    return settings, policy, config
