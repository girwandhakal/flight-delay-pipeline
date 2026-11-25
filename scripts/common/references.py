"""Shared airport reference loading and date-interval time-zone lookup."""

import csv
from datetime import date
from pathlib import Path
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo


def load_airports(directory: Union[str, Path]) -> dict[str, list[dict[str, Any]]]:
    """Load and validate non-overlapping historical airport time-zone intervals."""
    with (Path(directory) / "airports.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    by_airport: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ZoneInfo(row["time_zone"])
        if row["valid_from"] >= row["valid_to"]:
            raise ValueError("Airport reference interval must be nonempty.")
        by_airport.setdefault(row["Origin"], []).append(row)

    for intervals in by_airport.values():
        ordered = sorted(intervals, key=lambda r: r["valid_from"])
        if any(a["valid_to"] > b["valid_from"] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Airport reference intervals overlap.")

    return by_airport


def airport_zone(
    airports: dict[str, list[dict[str, Any]]],
    airport: str,
    flight_date: Optional[date],
) -> Optional[str]:
    """Return the IANA time zone valid for an airport on a given flight date."""
    if flight_date is None:
        return None
    iso_date = flight_date.isoformat()
    return next(
        (
            r["time_zone"]
            for r in airports.get(airport, [])
            if r["valid_from"] <= iso_date < r["valid_to"]
        ),
        None,
    )
