"""Compute incoming network delay pressure from BTS schedules and known departures."""

from datetime import date, datetime, timedelta, timezone

from pyspark.sql import DataFrame, functions as F

from .recent_delays import (
    DELAY_THRESHOLD_MINUTES, UPSTREAM_EDGE_HOURS, UPSTREAM_FEATURES, UPSTREAM_HISTORY_MINUTES,
    UPSTREAM_LAG_MINUTES, UPSTREAM_ROLES, UPSTREAM_SNAPSHOT_MINUTES, scheduled_clocks,
)

SCHEDULE_KEYS = ("Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest")
SCHEDULE_CLOCKS = ("FlightDate", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "CRSElapsedTime")
BIN_SECONDS = UPSTREAM_SNAPSHOT_MINUTES * 60


def schedule_times(flights: DataFrame, airports: dict) -> DataFrame:
    resolve = F.udf(lambda row: scheduled_clocks(row, airports), "departure string, arrival string")
    return (flights.withColumn("_clock", resolve(F.struct(*SCHEDULE_CLOCKS)))
            .withColumn("departure", F.to_timestamp("_clock.departure").cast("long"))
            .withColumn("arrival", F.to_timestamp("_clock.arrival").cast("long"))
            .drop("_clock"))


def build_inbound_schedule(raw: DataFrame, airports: dict) -> DataFrame:
    fields = ("FlightDate", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", *SCHEDULE_KEYS)
    source = (raw.select(*(F.trim(F.col(name)).alias(name) for name in fields))
              .filter(F.col("CRSDepTime").rlike(r"^\d{1,4}$"))
              .withColumn("FlightDate", F.to_date("FlightDate"))
              .withColumn("CRSDepTime", F.lpad("CRSDepTime", 4, "0"))
              .where(F.col("FlightDate").isNotNull()))
    for key in SCHEDULE_KEYS:
        source = source.where(F.col(key) != "")
    source = source.withColumn("flight_id", F.concat_ws("_", "FlightDate", *SCHEDULE_KEYS, "CRSDepTime"))
    times = schedule_times(source, airports)
    clocks = F.struct("Origin", "Dest", "departure", "arrival",
                      F.date_format("FlightDate", "yyyy-MM").alias("source_month"))
    return (times.groupBy("flight_id").agg(F.min(clocks).alias("first"), F.max(clocks).alias("last"))
            .select("flight_id", "first.*", F.when(F.col("first") != F.col("last"), "conflict")
                    .when(F.col("first.arrival").isNull(), "unresolved").otherwise("usable").alias("status")))


def uncovered_month_edges(core_months: set[date], source_months: set[date]) -> list[int]:
    edges = set()
    for month in core_months:
        previous = (month - timedelta(days=1)).replace(day=1)
        following = (month + timedelta(days=32)).replace(day=1)
        if previous not in source_months:
            edges.add(month)
        if following not in source_months:
            edges.add(following)
    return [int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())
            for day in sorted(edges)]


def add_upstream_pressure(
    flights: DataFrame, schedule: DataFrame, events: DataFrame, airports: dict, edges: list[int],
) -> DataFrame:
    """Attach one row per flight while retaining empty and unavailable windows separately."""
    schedule = schedule.where(F.col("status") == "usable")
    inbound = (schedule.groupBy("Dest", "Origin", "arrival").count()
               .select(F.col("Dest").alias("airport"), F.col("Origin").alias("feeder"), "arrival",
                       F.floor(F.col("arrival") / BIN_SECONDS).alias("bin"), "count"))
    departures = (events.where((F.col("role") == "departure") & F.col("minutes").isNotNull())
                  .select("flight_id", F.col("airport").alias("feeder"), "event_seconds",
                          (F.col("minutes") >= DELAY_THRESHOLD_MINUTES).cast("int").alias("delayed")))
    lag = UPSTREAM_LAG_MINUTES * 60
    history = UPSTREAM_HISTORY_MINUTES * 60
    first_snapshot = F.floor((F.col("event_seconds") + lag) / BIN_SECONDS) + 1
    last_snapshot = F.floor((F.col("event_seconds") + lag + history) / BIN_SECONDS)
    rates = (departures.withColumn("snapshot", F.explode(F.sequence(first_snapshot, last_snapshot)))
             .groupBy("feeder", "snapshot").agg(F.count("*").alias("observed"), F.sum("delayed").alias("late")))
    targets = schedule_times(flights.select("flight_id", *SCHEDULE_CLOCKS), airports)
    queries = []
    for role, (start, end) in UPSTREAM_ROLES.items():
        anchor = F.col("departure" if role == "origin" else "arrival")
        queries.append(F.struct(F.lit(role).alias("role"), F.col("Origin" if role == "origin" else "Dest").alias("airport"),
                                (anchor + start * 60).alias("start"), (anchor + end * 60).alias("end")))
    windows = (targets.select("flight_id", "departure", F.explode(F.array(*queries)).alias("window"))
               .select("flight_id", "departure", "window.*")
               .where("departure IS NOT NULL AND start IS NOT NULL"))
    edge_seconds = UPSTREAM_EDGE_HOURS * 3600
    for edge in edges:
        windows = windows.where(
            (~F.col("departure").between(edge - edge_seconds, edge + edge_seconds))
            & ((F.col("end") <= edge - edge_seconds) | (F.col("start") > edge + edge_seconds))
        )
    bins = windows.withColumn("bin", F.explode(F.sequence(
        F.floor(F.col("start") / BIN_SECONDS), F.floor((F.col("end") - 1) / BIN_SECONDS))))
    # Left join on the exact window so a complete, empty schedule remains distinguishable.
    matches = bins.join(inbound, (bins.airport == inbound.airport) & (bins.bin == inbound.bin)
                        & (inbound.arrival >= bins.start) & (inbound.arrival < bins.end), "left")
    own_schedule = schedule.select("flight_id", F.col("Dest").alias("own_airport"),
                                   F.col("Origin").alias("own_feeder"), F.col("arrival").alias("own_arrival"))
    matches = (matches.select("flight_id", "role", "departure", bins.airport, "feeder", "arrival", "count")
               .join(own_schedule, "flight_id", "left"))
    is_self = ((F.col("airport") == F.col("own_airport")) & (F.col("feeder") == F.col("own_feeder"))
               & (F.col("arrival") == F.col("own_arrival")))
    demand = (matches.groupBy("flight_id", "role", "departure", "feeder")
              .agg(F.sum(F.coalesce("count", F.lit(0)) - F.when(is_self, 1).otherwise(0)).alias("weight"))
              .withColumn("snapshot", F.floor(F.col("departure") / BIN_SECONDS)))
    own_departure = departures.select("flight_id", F.col("feeder").alias("own_event_airport"),
                                      "event_seconds", "delayed")
    weighted = demand.join(rates, ["feeder", "snapshot"], "left").join(own_departure, "flight_id", "left")
    snapshot_time = F.col("snapshot") * BIN_SECONDS
    own_eligible = ((F.col("feeder") == F.col("own_event_airport"))
                    & (F.col("event_seconds") >= snapshot_time - lag - history)
                    & (F.col("event_seconds") < snapshot_time - lag))
    weighted = (weighted.withColumn("observed", F.col("observed") - F.when(own_eligible, 1).otherwise(0))
                .withColumn("late", F.col("late") - F.when(own_eligible, F.col("delayed")).otherwise(0)))
    known_weight = F.when(F.col("observed") > 0, F.col("weight")).otherwise(0)
    summary = (weighted.groupBy("flight_id", "role")
               .agg(F.sum("weight").alias("scheduled_arrivals"), F.sum(known_weight).alias("known"),
                    F.sum(F.when(F.col("observed") > 0, F.col("weight") * F.col("late") / F.col("observed"))
                          .otherwise(0)).alias("weighted_late"))
               .withColumn("delay_pressure", F.when(F.col("known") > 0, F.col("weighted_late") / F.col("known")))
               .withColumn("observed_share", F.when(F.col("scheduled_arrivals") > 0,
                                                   F.col("known") / F.col("scheduled_arrivals"))))
    columns = []
    for name in UPSTREAM_FEATURES:
        role, metric = name.split("_upstream_")
        columns.append(F.max(F.when(F.col("role") == role, F.col(metric))).alias(name))
    return flights.join(summary.groupBy("flight_id").agg(*columns), "flight_id", "left")
