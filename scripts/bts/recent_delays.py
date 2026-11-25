"""Build airport histories from completed events and federal holiday flags."""

from copy import copy
import csv
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pyspark.sql import DataFrame, Row, Window, functions as F

from scripts.bts.cache import active
from scripts.bts.download import iter_months
from scripts.common.config import project_path
from scripts.common.io import reference_matches, sha256
from scripts.common.references import airport_zone, load_airports
from scripts.common.time_utils import (
    clock_matches, event_zone, local_cutoff, scheduled_clocks, valid_delay,
)
from .block_time import BLOCK_TIME_SIDECAR, POLICY as BLOCK_TIME_POLICY, SLACK_FEATURES, resolve_block_history

ROLLING_FEATURES = [
    f"{role}_{measurement}_2h"
    for role in ("origin_departure", "destination_arrival")
    for measurement in ("delay_rate", "mean_delay_minutes")
]
TAXI_MEAN = "origin_departure_mean_taxi_out_2h"
TAXI_SURGE = "origin_departure_taxi_out_surge_2h"
TAXI_FEATURES = [TAXI_MEAN, TAXI_SURGE]
ROLLING_FEATURES += TAXI_FEATURES
TAXI_BASELINE_FILE = "_taxi_out_baselines.json"
TAXI_BASELINE_FALLBACK_MINUTES = 12.0
MAX_TAXI_MINUTES = 120.0
HOLIDAY_FEATURES = [
    "is_federal_holiday",
    "is_day_before_federal_holiday",
    "is_day_after_federal_holiday",
]
DELAY_THRESHOLD_MINUTES = 15
FEATURES = ROLLING_FEATURES + HOLIDAY_FEATURES
EVENT_COLUMNS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "CRSElapsedTime", "DepTime", "DepDelay", "ArrTime", "ArrDelay", "Diverted",
]
BTS_EVENT_COLUMNS = [*EVENT_COLUMNS, "TaxiOut", "WheelsOff", "Cancelled"]


def feature_policy(root: Union[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Validate rolling window and airport reference files against sources.json."""
    window = config["bts_features"]["rolling_window_minutes"]
    if not isinstance(window, int) or window != 120:
        raise ValueError("bts_features.rolling_window_minutes must be 120.")

    reference = project_path(root, config["datasets"]["weather"]["reference_directory"])
    sources = json.loads((reference / "sources.json").read_text(encoding="utf-8"))

    for name, checksum in sources["files"].items():
        if not reference_matches(reference / name, checksum):
            raise ValueError(f"Reference changed: {name}")

    return {
        "version": 8,
        "rolling_window_minutes": window,
        "event_availability": "immediate",
        "delay_threshold_minutes": DELAY_THRESHOLD_MINUTES,
        "holiday_calendar": "USFederalHolidayCalendar_actual_and_observed",
        "columns": [*FEATURES, *SLACK_FEATURES],
        "outbound_schedule_slack": BLOCK_TIME_POLICY,
        "splits": config["splits"],
        "reference_files": sources["files"],
        "taxi_out": {
            "availability": "verified wheels-off",
            "max_observation_minutes": MAX_TAXI_MINUTES,
            "fallback_baseline_minutes": TAXI_BASELINE_FALLBACK_MINUTES,
            "baseline_percentile": 0.15,
            "baseline": "expanding prior training months for train; frozen training baseline for validation and test",
        },
    }


def feature_months(settings):
    return {date(year, month, 1) for year, month in iter_months(settings["start_month"], settings["end_month"])}


def file_fingerprints(directory: Path, pattern: str, months) -> list[dict[str, Any]]:
    """Generate path, byte size, and mtime fingerprints for matching files."""
    return [
        {
            "path": str(p.relative_to(directory)),
            "bytes": p.stat().st_size,
            "modified_ns": p.stat().st_mtime_ns,
        }
        for month in sorted(months)
        for p in sorted((directory / month.strftime("%Y/%m")).glob(pattern))
    ]


def feature_manifest(
    root: Union[str, Path],
    config: dict[str, Any],
    bronze_dir: Optional[Union[str, Path]] = None,
    silver_dir: Optional[Union[str, Path]] = None,
    block_history: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a complete provenance dictionary covering policy and input files."""
    root = Path(root)
    bronze = Path(bronze_dir or project_path(root, config["datasets"]["bts"]["output_directory"]))
    silver = Path(silver_dir or active(root, "silver"))
    core = feature_months(config["datasets"]["bts"])
    return {
        **feature_policy(root, config),
        "block_time_code_sha256": sha256(Path(__file__).with_name("block_time.py")),
        "block_history": block_history if block_history is not None else resolve_block_history(root, config, bronze_dir),
        "bronze_files": file_fingerprints(bronze, "*.csv", core),
        "silver_files": file_fingerprints(silver, "*.parquet", core),
    }


def features_current(root: Union[str, Path], config: dict[str, Any]) -> bool:
    """Check whether BTS Gold datasets exist and match the current feature manifest."""
    gold = active(Path(root), "gold")
    if gold.parent.name == "bts_generations":
        from .cache import features_current as generation_current
        return generation_current(Path(root), config)
    try:
        recorded = json.loads((gold / "_features.json").read_text())
        return (
            isinstance(recorded, dict)
            and all((gold / split / "_SUCCESS").exists() for split in config["splits"])
            and recorded.pop("taxi_baselines_sha256") == sha256(gold / TAXI_BASELINE_FILE)
            and recorded.pop("block_time_sha256") == sha256(gold / BLOCK_TIME_SIDECAR)
            and recorded == feature_manifest(root, config)
        )
    except (KeyError, OSError, ValueError):
        return False


def event_clocks(row: Row, airports: dict[str, list[dict[str, Any]]]) -> tuple[Optional[str], Optional[str]]:
    """Reconstruct exact UTC departure and arrival timestamps for a raw BTS flight row.

    Validates that actual clock times match recorded delay offsets and civil calendar dates.
    """
    zone = airport_zone(airports, row.Origin, row.FlightDate)
    scheduled, _ = local_cutoff(row.FlightDate, row.CRSDepTime, zone)
    departure = arrival = None
    dep_delay, arr_delay = valid_delay(row.DepDelay), valid_delay(row.ArrDelay)
    elapsed = valid_delay(row.CRSElapsedTime)

    if scheduled is None:
        return None, None

    try:
        if dep_delay is not None:
            actual = scheduled + timedelta(minutes=dep_delay)
            if clock_matches(actual, row.DepTime, event_zone(airports, row.Origin, actual)):
                departure = actual.isoformat() + "Z"

        if arr_delay is not None and elapsed is not None and elapsed > 0 and valid_delay(row.Diverted) == 0:
            actual = scheduled + timedelta(minutes=elapsed + arr_delay)
            after_departure = dep_delay is None or actual > scheduled + timedelta(minutes=dep_delay)
            if after_departure and clock_matches(actual, row.ArrTime, event_zone(airports, row.Dest, actual)):
                arrival = actual.isoformat() + "Z"
    except (OverflowError, ValueError):
        return None, None

    return departure, arrival


def build_events(raw: DataFrame, airports: dict[str, list[dict[str, Any]]]) -> DataFrame:
    """Extract and validate completed departure and arrival events from raw BTS data.

    Taxi events are timed at verified wheels-off, independently of gate departure events.
    """
    missing = sorted(set(BTS_EVENT_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"BTS event columns missing: {missing}; acquire complete BTS CSVs.")

    source = raw.select(*[F.trim(F.col(c)).alias(c) for c in BTS_EVENT_COLUMNS])
    source = source.filter(F.col("CRSDepTime").rlike(r"^\d{1,4}$"))
    for identifier in ("Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"):
        source = source.filter(F.col(identifier).isNotNull() & (F.col(identifier) != ""))

    source = source.withColumn("FlightDate", F.to_date("FlightDate")).withColumn(
        "CRSDepTime", F.lpad("CRSDepTime", 4, "0")
    )
    source = source.withColumn(
        "flight_id",
        F.concat_ws("_", "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime")
    )

    def clocks(row):
        departure, arrival = event_clocks(row, airports)
        taxi = valid_delay(row.TaxiOut)
        if departure and taxi is not None and taxi > 0 and valid_delay(row.Cancelled) == 0:
            try:
                wheels_off = datetime.fromisoformat(departure[:-1]) + timedelta(minutes=taxi)
                if clock_matches(wheels_off, row.WheelsOff, event_zone(airports, row.Origin, wheels_off)):
                    return departure, arrival, wheels_off.isoformat() + "Z", min(taxi, MAX_TAXI_MINUTES)
            except (OverflowError, ValueError):
                pass
        return departure, arrival, None, None

    convert = F.udf(clocks, "departure string, arrival string, taxi string, taxi_minutes double")
    source = source.withColumn("clocks", convert(F.struct(*BTS_EVENT_COLUMNS)))

    event_columns = []
    for role, airport, minutes in (
        ("departure", "Origin", "DepDelay"), ("arrival", "Dest", "ArrDelay"), ("taxi", "Origin", "clocks.taxi_minutes"),
    ):
        event_time = F.to_timestamp(f"clocks.{role}")
        if role == "taxi":
            event_time = F.ceil(event_time.cast("double"))
        event_columns.append(
            F.struct(
                F.col(airport).alias("airport"),
                F.lit(role).alias("role"),
                event_time.cast("long").alias("event_seconds"),
                F.col(minutes).cast("double").alias("minutes"),
            )
        )
    events = (source.select("flight_id", F.explode(F.array(*event_columns)).alias("event"))
              .select("flight_id", "event.*").where("event_seconds is not null"))

    # Group duplicate records and reject any conflicting event clocks
    grouped = events.groupBy("flight_id", "role").agg(
        F.min(F.struct("airport", "event_seconds", "minutes")).alias("event"),
        F.max(F.struct("airport", "event_seconds", "minutes")).alias("last_event"),
    )
    agrees = F.col("event") == F.col("last_event")
    conflicting_departure = F.max(F.when((F.col("role") == "departure") & ~agrees, 1).otherwise(0)).over(
        Window.partitionBy("flight_id")
    )
    return (grouped.withColumn("conflicting_departure", conflicting_departure)
            .where(agrees & ((F.col("role") != "taxi") | (F.col("conflicting_departure") == 0)))
            .select("flight_id", "role", "event.*"))


def add_cutoffs(flights: DataFrame, airports: dict[str, list[dict[str, Any]]]) -> DataFrame:
    """Attach UTC scheduled departure cutoff instant to flight records."""
    def cutoff(flight_date, clock, airport):
        instant, _ = local_cutoff(flight_date, clock, airport_zone(airports, airport, flight_date))
        return instant.isoformat() + "Z" if instant else None

    convert = F.udf(cutoff, "string")
    return flights.withColumn("_cutoff", F.to_timestamp(convert("FlightDate", "CRSDepTime", "Origin")))


def add_recent_delays(flights: DataFrame, events: DataFrame, window_minutes: int = 120) -> DataFrame:
    """Compute rolling completed-delay statistics for origin departure and destination arrival.

    Merges cutoff queries with completed historical events in a single window pass.
    Self-events occurring before cutoff are safely excluded to eliminate target leakage.

    Args:
        flights: Flights DataFrame with `_cutoff` timestamp column.
        events: Historical events DataFrame from `build_events`.
        window_minutes: Lookback window duration (default 120 minutes).

    Returns:
        Enriched flights DataFrame with rolling delay rates, means, and taxi-out features.
    """
    # 1. Build query points at flight cutoff time for origin departure and destination arrival
    queries = flights.select(
        "flight_id", "Origin", "Dest", F.col("_cutoff").cast("long").alias("event_seconds")
    )
    query_rows = []
    for airport, role in (("Origin", "departure"), ("Dest", "arrival"), ("Origin", "taxi")):
        query_rows.append(
            queries.select(
                "flight_id",
                F.col(airport).alias("airport"),
                F.lit(role).alias("role"),
                "event_seconds",
                F.lit(None).cast("double").alias("minutes"),
            )
        )
    query_rows = query_rows[0].unionByName(query_rows[1]).unionByName(query_rows[2]).filter("event_seconds is not null")

    # 2. Attach flight's own event timestamp if present to enable self-exclusion
    own = events.select(
        "flight_id",
        "airport",
        "role",
        F.col("event_seconds").alias("own_seconds"),
        F.col("minutes").alias("own_minutes"),
    )
    query_rows = query_rows.join(own, ["flight_id", "airport", "role"], "left")

    # 3. Interleave query points with completed event records into a single timeline
    event_rows = events.withColumn("flight_id", F.lit(None).cast("string"))
    timeline = query_rows.unionByName(event_rows, allowMissingColumns=True)

    # 4. Compute rolling count, delayed count, and total delay over lookback window [-120min + 1s, 0s]
    ordered = Window.partitionBy("airport", "role").orderBy("event_seconds")
    window = ordered.rangeBetween(-window_minutes * 60 + 1, 0)
    taxi_window = ordered.rangeBetween(-window_minutes * 60, 0)
    taxi = F.col("role") == "taxi"
    count = F.count("minutes")
    total = F.sum(F.greatest("minutes", F.lit(0.0)))
    stats = timeline.select(
        "*",
        F.when(taxi, count.over(taxi_window)).otherwise(count.over(window)).alias("n"),
        F.sum((F.col("minutes") >= 15).cast("int")).over(window).alias("delayed"),
        F.when(taxi, total.over(taxi_window)).otherwise(total.over(window)).alias("total_minutes"),
    )
    stats = stats.filter("flight_id is not null")

    # 5. Subtract own event from window totals if it completed within the lookback window
    own_in_window = F.coalesce(
        (F.col("own_seconds") >= F.col("event_seconds") - window_minutes * 60 + F.when(taxi, 0).otherwise(1))
        & (F.col("own_seconds") <= F.col("event_seconds")),
        F.lit(False),
    )
    stats = stats.withColumn("n", F.col("n") - own_in_window.cast("int"))
    stats = stats.withColumn(
        "delayed",
        F.coalesce("delayed", F.lit(0)) - F.when(own_in_window & (F.col("own_minutes") >= 15), 1).otherwise(0),
    )
    stats = stats.withColumn(
        "total_minutes",
        F.col("total_minutes") - F.when(own_in_window, F.greatest("own_minutes", F.lit(0.0))).otherwise(0.0),
    )

    # 6. Aggregate rolling features for origin departure and destination arrival
    expressions = []
    for role, prefix in (("departure", "origin_departure"), ("arrival", "destination_arrival")):
        for value, name in (
            (F.when(F.col("n") > 0, F.col("delayed") / F.col("n")), "delay_rate"),
            (F.when(F.col("n") > 0, F.col("total_minutes") / F.col("n")), "mean_delay_minutes"),
        ):
            expressions.append(F.max(F.when(F.col("role") == role, value)).alias(f"{prefix}_{name}_2h"))

    expressions += [
        F.max(F.when(taxi & (F.col("n") > 0), F.col("total_minutes") / F.col("n"))).alias(TAXI_MEAN),
    ]
    return flights.join(stats.groupBy("flight_id").agg(*expressions), "flight_id", "left").drop("_cutoff")


def compute_unimpeded_taxi_baselines(
    events: DataFrame, flights: DataFrame, splits: dict, airports: dict,
) -> DataFrame:
    """Fit trailing-month training snapshots and a frozen evaluation reference."""
    train = splits["train"]
    months = pd.period_range(train["start_date"], train["end_date"], freq="M")
    snapshots = [(str(month), month.start_time.date()) for month in months[1:]]
    snapshots.append(("evaluation", date.fromisoformat(splits["val"]["start_date"])))
    bounds = []
    for airport in airports:
        for key, day in snapshots:
            cutoff, _ = local_cutoff(day, "0000", airport_zone(airports, airport, day))
            if cutoff is not None:
                bounds.append((airport, key, day, int(cutoff.replace(tzinfo=timezone.utc).timestamp())))
    dates = flights.sparkSession.createDataFrame(
        bounds, "airport string, taxi_snapshot string, first_day date, bound_seconds long",
    )
    training = flights.where("split = 'train'").select("flight_id", "FlightDate")
    eligible = events.where("role = 'taxi'").join(training, "flight_id")
    return (eligible.join(F.broadcast(dates), "airport")
            .where((F.col("FlightDate") < F.col("first_day"))
                   & (F.col("event_seconds") < F.col("bound_seconds")))
            .groupBy(F.col("airport").alias("Origin"), "taxi_snapshot")
            .agg(F.percentile_approx("minutes", 0.15).alias("unimpeded_taxi_minutes")))


def add_holidays(flights: DataFrame, start_date: str, end_date: str) -> DataFrame:
    """Attach binary US federal holiday indicator features to flight records.

    Adds:
    - is_federal_holiday: 1 if FlightDate is an observed or actual federal holiday
    - is_day_before_federal_holiday: 1 if FlightDate is the day before a federal holiday
    - is_day_after_federal_holiday: 1 if FlightDate is the day after a federal holiday
    """
    start = pd.Timestamp(start_date) - pd.Timedelta(days=370)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=370)
    calendar = USFederalHolidayCalendar()
    holidays = set(calendar.holidays(start, end).date)

    # Include actual unobserved holiday calendar dates
    for rule in calendar.rules:
        actual = copy(rule)
        actual.observance = None
        holidays.update(actual.dates(start, end).date)

    for name, offset in zip(HOLIDAY_FEATURES, (0, 1, -1)):
        dates = [d - timedelta(days=offset) for d in sorted(holidays)]
        flights = flights.withColumn(name, F.col("FlightDate").isin(dates).cast("int"))

    return flights
