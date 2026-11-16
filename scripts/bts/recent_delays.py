"""Build airport histories from completed events and federal holiday flags."""

from copy import copy
import csv
from datetime import timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pyspark.sql import functions as F, Window

from scripts.origin_weather.build_features import local_cutoff
from scripts.origin_weather.download import project_path, reference_matches

ROLLING_FEATURES = [
    f"{role}_{measurement}_2h"
    for role in ("origin_departure", "destination_arrival")
    for measurement in ("delay_rate", "mean_delay_minutes", "count")
]
HOLIDAY_FEATURES = ["is_federal_holiday", "is_day_before_federal_holiday", "is_day_after_federal_holiday"]
FEATURES = ROLLING_FEATURES + HOLIDAY_FEATURES
EVENT_COLUMNS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "CRSElapsedTime", "DepTime", "DepDelay", "ArrTime", "ArrDelay", "Diverted",
]


def feature_policy(root, config):
    window = config["bts_features"]["rolling_window_minutes"]
    if type(window) is not int or window != 120:
        raise ValueError("bts_features.rolling_window_minutes must be 120.")
    reference = project_path(root, config["datasets"]["weather"]["reference_directory"])
    sources = json.loads((reference / "sources.json").read_text(encoding="utf-8"))
    for name, checksum in sources["files"].items():
        if not reference_matches(reference / name, checksum):
            raise ValueError(f"Reference changed: {name}")
    return {
        "version": 1, "rolling_window_minutes": window,
        "event_availability": "immediate", "delay_threshold_minutes": 15,
        "holiday_calendar": "USFederalHolidayCalendar_actual_and_observed",
        "columns": FEATURES, "splits": config["splits"], "reference_files": sources["files"],
    }


def file_fingerprints(directory, pattern):
    return [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size,
             "modified_ns": p.stat().st_mtime_ns} for p in sorted(directory.glob(pattern))]


def feature_manifest(root, config, bronze_dir=None, silver_dir=None):
    bronze = Path(bronze_dir or project_path(root, config["datasets"]["bts"]["output_directory"]))
    silver = Path(silver_dir or root / "data/silver/bts")
    return {**feature_policy(root, config), "bronze_files": file_fingerprints(bronze, "*/*/*.csv"),
            "silver_files": file_fingerprints(silver, "*/*/*.parquet")}


def features_current(root, config):
    gold = root / "data/gold/bts"
    try:
        return (all((gold / split / "_SUCCESS").exists() for split in config["splits"])
                and json.loads((gold / "_features.json").read_text()) == feature_manifest(root, config))
    except (KeyError, OSError, ValueError):
        return False


def load_airports(directory):
    with (Path(directory) / "airports.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_airport = {}
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


def airport_zone(airports, airport, flight_date):
    if flight_date is None:
        return None
    return next((r["time_zone"] for r in airports.get(airport, [])
                 if r["valid_from"] <= flight_date.isoformat() < r["valid_to"]), None)


def valid_delay(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def clock_matches(instant, clock, zone):
    spelling = str(clock).strip()
    if not spelling.isdigit() or len(spelling) > 4 or not zone:
        return False
    hours, minutes = divmod(int(spelling), 100)
    if (hours, minutes) != (24, 0) and not (0 <= hours < 24 and 0 <= minutes < 60):
        return False
    local = instant.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(zone))
    return (local.hour, local.minute) == (hours % 24, minutes)


def event_zone(airports, airport, instant):
    zones = {r["time_zone"] for r in airports.get(airport, []) if r["valid_from"] <=
             instant.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(r["time_zone"])).date().isoformat()
             < r["valid_to"]}
    return next(iter(zones)) if len(zones) == 1 else None


def event_clocks(row, airports):
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


def build_events(raw, airports):
    missing = sorted(set(EVENT_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"BTS event columns missing: {missing}; acquire complete BTS CSVs.")
    source = raw.select(*[F.trim(F.col(c)).alias(c) for c in EVENT_COLUMNS])
    source = source.filter(F.col("CRSDepTime").rlike(r"^\d{1,4}$"))
    for identifier in ("Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"):
        source = source.filter(F.col(identifier).isNotNull() & (F.col(identifier) != ""))
    source = source.withColumn("FlightDate", F.to_date("FlightDate")).withColumn(
        "CRSDepTime", F.lpad("CRSDepTime", 4, "0"))
    source = source.withColumn("flight_id", F.concat_ws("_", "FlightDate", "Reporting_Airline",
        "Flight_Number_Reporting_Airline", "Origin", "Dest", "CRSDepTime"))
    convert = F.udf(lambda row: event_clocks(row, airports), "departure string, arrival string")
    source = source.withColumn("clocks", convert(F.struct(*EVENT_COLUMNS)))
    events = []
    for role, airport, delay in (("departure", "Origin", "DepDelay"), ("arrival", "Dest", "ArrDelay")):
        events.append(source.select("flight_id", F.col(airport).alias("airport"),
            F.lit(role).alias("role"), F.to_timestamp(f"clocks.{role}").cast("long").alias("event_seconds"),
            F.col(delay).cast("double").alias("delay_minutes")).filter("event_seconds is not null"))
    # Conflicting copies cannot establish a unique event time.
    grouped = events[0].unionByName(events[1]).groupBy("flight_id", "role").agg(
        F.min(F.struct("airport", "event_seconds", "delay_minutes")).alias("event"),
        F.max(F.struct("airport", "event_seconds", "delay_minutes")).alias("last_event"))
    return grouped.filter(F.col("event") == F.col("last_event")).select("flight_id", "role", "event.*")


def add_cutoffs(flights, airports):
    def cutoff(flight_date, clock, airport):
        instant, _ = local_cutoff(flight_date, clock, airport_zone(airports, airport, flight_date))
        return instant.isoformat() + "Z" if instant else None
    convert = F.udf(cutoff, "string")
    return flights.withColumn("_cutoff", F.to_timestamp(convert("FlightDate", "CRSDepTime", "Origin")))


def add_recent_delays(flights, events, window_minutes=120):
    queries = flights.select("flight_id", "Origin", "Dest", F.col("_cutoff").cast("long").alias("event_seconds"))
    query_rows = []
    for airport, role in (("Origin", "departure"), ("Dest", "arrival")):
        query_rows.append(queries.select("flight_id", F.col(airport).alias("airport"),
            F.lit(role).alias("role"), "event_seconds", F.lit(None).cast("double").alias("delay_minutes")))
    query_rows = query_rows[0].unionByName(query_rows[1]).filter("event_seconds is not null")
    own = events.select("flight_id", "airport", "role", F.col("event_seconds").alias("own_seconds"),
                        F.col("delay_minutes").alias("own_delay"))
    query_rows = query_rows.join(own, ["flight_id", "airport", "role"], "left")
    event_rows = events.withColumn("flight_id", F.lit(None).cast("string"))
    timeline = query_rows.unionByName(event_rows, allowMissingColumns=True)
    # Integer-second bounds implement an open lower edge and include simultaneous events.
    window = Window.partitionBy("airport", "role").orderBy("event_seconds").rangeBetween(-window_minutes * 60 + 1, 0)
    stats = timeline.select("*", F.count("delay_minutes").over(window).alias("n"),
        F.sum((F.col("delay_minutes") >= 15).cast("int")).over(window).alias("delayed"),
        F.sum(F.greatest("delay_minutes", F.lit(0.0))).over(window).alias("total_delay"))
    stats = stats.filter("flight_id is not null")
    own_in_window = F.coalesce((F.col("own_seconds") > F.col("event_seconds") - window_minutes * 60)
                               & (F.col("own_seconds") <= F.col("event_seconds")), F.lit(False))
    stats = stats.withColumn("n", F.col("n") - own_in_window.cast("int"))
    stats = stats.withColumn("delayed", F.coalesce("delayed", F.lit(0))
        - F.when(own_in_window & (F.col("own_delay") >= 15), 1).otherwise(0))
    stats = stats.withColumn("total_delay", F.col("total_delay")
        - F.when(own_in_window, F.greatest("own_delay", F.lit(0.0))).otherwise(0.0))
    expressions = []
    for role, prefix in (("departure", "origin_departure"), ("arrival", "destination_arrival")):
        for value, name in ((F.when(F.col("n") > 0, F.col("delayed") / F.col("n")), "delay_rate"),
                            (F.when(F.col("n") > 0, F.col("total_delay") / F.col("n")), "mean_delay_minutes"),
                            (F.col("n"), "count")):
            expressions.append(F.max(F.when(F.col("role") == role, value)).alias(f"{prefix}_{name}_2h"))
    return flights.join(stats.groupBy("flight_id").agg(*expressions), "flight_id", "left").drop("_cutoff")


def add_holidays(flights, start_date, end_date):
    start, end = pd.Timestamp(start_date) - pd.Timedelta(days=370), pd.Timestamp(end_date) + pd.Timedelta(days=370)
    calendar = USFederalHolidayCalendar()
    holidays = set(calendar.holidays(start, end).date)
    for rule in calendar.rules:
        actual = copy(rule)
        actual.observance = None
        holidays.update(actual.dates(start, end).date)
    for name, offset in zip(HOLIDAY_FEATURES, (0, 1, -1)):
        dates = [d - timedelta(days=offset) for d in sorted(holidays)]
        flights = flights.withColumn(name, F.col("FlightDate").isin(dates).cast("int"))
    return flights
