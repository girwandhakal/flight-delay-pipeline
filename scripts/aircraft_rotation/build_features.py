"""Link each flight to its immediate same-aircraft predecessor."""

from datetime import datetime, timedelta
from typing import Any, Optional

from pyspark.sql import DataFrame, Window, functions as F

from scripts.bts.recent_delays import (
    EVENT_COLUMNS, airport_zone, clock_matches, event_clocks, event_zone, scheduled_clocks, valid_delay,
)


TURNAROUND_SHORTFALL = "rotation_turnaround_shortfall_minutes"
TURNAROUND_MISSING = "rotation_turnaround_shortfall_missing"
ROTATION_NUMERIC = ("rotation_scheduled_turnaround_minutes", "rotation_prior_arrival_delay_minutes", TURNAROUND_SHORTFALL)
ROTATION_FLAGS = ("rotation_link_missing", "rotation_outcome_missing", TURNAROUND_MISSING)
MAX_GAP_HOURS = 24
OUTCOME_LAG_MINUTES = 15
MAX_TYPICAL_TURN_MINUTES = 180
MIN_TURN_HISTORY = 20
RAW_COLUMNS = [*EVENT_COLUMNS, "Tail_Number", "CRSArrTime", "Cancelled"]
KEY_COLUMNS = ["Reporting_Airline", "Flight_Number_Reporting_Airline", "Origin", "Dest"]


def operation_clocks(
    row: Any, airports: dict,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolve scheduled and actual departure and arrival UTC clocks."""
    scheduled, scheduled_arrival = scheduled_clocks(row, airports)
    if scheduled is None:
        return None, None, None, None
    departure, arrival = event_clocks(row, airports)
    if valid_delay(row.Cancelled) != 0:
        departure = None
    if not departure:
        arrival = None
    return scheduled, scheduled_arrival, arrival, departure


def build_operations(raw: DataFrame, airports: list[dict[str, str]]) -> DataFrame:
    """Normalize raw BTS operations, reconcile duplicate records, and resolve UTC timestamps."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing BTS rotation columns: {', '.join(sorted(missing))}")
    operations = raw.select(*[F.trim(F.col(column)).alias(column) for column in RAW_COLUMNS])
    operations = operations.withColumn("FlightDate", F.to_date("FlightDate")).withColumn(
        "tail", F.when(F.col("Tail_Number") != "", F.upper("Tail_Number"))
    ).drop("Tail_Number")
    valid_key = F.col("FlightDate").isNotNull() & F.col("CRSDepTime").rlike(r"^\d{1,4}$")
    for column in KEY_COLUMNS:
        valid_key = valid_key & F.col(column).isNotNull() & (F.col(column) != "")
    operations = operations.withColumn(
        "CRSDepTime", F.when(F.col("CRSDepTime").rlike(r"^\d{1,4}$"), F.lpad("CRSDepTime", 4, "0"))
    ).withColumn(
        "flight_id", F.when(valid_key, F.concat_ws("_", "FlightDate", *KEY_COLUMNS, "CRSDepTime"))
    )
    actual = F.struct("DepTime", "DepDelay", "ArrTime", "ArrDelay", "Cancelled", "Diverted")
    # Post-flight disagreements can invalidate the outcome, but cannot break the scheduled link.
    operations = operations.groupBy(
        "flight_id", "FlightDate", "tail", *KEY_COLUMNS, "CRSDepTime", "CRSArrTime", "CRSElapsedTime"
    ).agg(F.min(actual).alias("actual"), F.max(actual).alias("last_actual")).select(
        "flight_id", "FlightDate", "tail", *KEY_COLUMNS, "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
        (F.col("actual") == F.col("last_actual")).alias("outcome_consistent"), "actual.*",
    )
    clocks = F.udf(
        lambda row: operation_clocks(row, airports),
        "scheduled string, scheduled_arrival string, arrival string, departure string",
    )
    return operations.withColumn("clocks", clocks(F.struct(*EVENT_COLUMNS, "CRSArrTime", "Cancelled"))).select(
        "flight_id", "FlightDate", "tail", "Dest",
        F.to_timestamp("clocks.scheduled").alias("scheduled_utc"),
        F.to_timestamp("clocks.scheduled_arrival").alias("scheduled_arrival_utc"),
        F.when(F.col("outcome_consistent"), F.to_timestamp("clocks.departure")).alias("actual_departure_utc"),
        F.when(F.col("outcome_consistent"), F.to_timestamp("clocks.arrival")).alias("actual_arrival_utc"),
        F.when(F.col("outcome_consistent"), F.col("ArrDelay").cast("double")).alias("arrival_delay"),
        F.when(F.col("outcome_consistent"), F.coalesce(
            (F.col("Cancelled").cast("double") == 0) & (F.col("Diverted").cast("double") == 0), F.lit(False)
        )).alias("completed"),
    )


def build_rotation_links(
    parent: DataFrame,
    operations: DataFrame,
    first_reliable_date: str,
) -> DataFrame:
    """Identify each aircraft's immediate predecessor leg and compute turnaround features."""
    by_id = Window.partitionBy("flight_id")
    candidates = operations.withColumn(
        "conflict", F.col("flight_id").isNull() | (F.count("*").over(by_id) > 1)
    )
    operation = F.struct(
        "flight_id", "Dest", "scheduled_utc", "scheduled_arrival_utc", "actual_arrival_utc",
        "arrival_delay", "completed", "conflict",
    )
    timeline = (
        candidates.where("tail is not null and scheduled_utc is not null")
        .groupBy("tail", "scheduled_utc").agg(F.count("*").alias("group_size"), F.first(operation).alias("leg"))
        .withColumn("leg", F.when(F.col("group_size") == 1, F.col("leg")))
    )
    by_tail = Window.partitionBy("tail").orderBy("scheduled_utc")
    timeline = timeline.select(
        "tail", "scheduled_utc", "group_size",
        F.lag("group_size").over(by_tail).alias("prior_group_size"),
        F.lag("leg").over(by_tail).alias("prior"),
    )
    target_operations = operations.where("flight_id is not null").groupBy("flight_id").agg(
        F.count("*").alias("target_copies"),
        F.first(F.struct("tail", "scheduled_utc", "actual_arrival_utc", "arrival_delay",
                         "actual_departure_utc", "completed")).alias("target"),
    ).select(
        "flight_id", "target_copies",
        F.when(F.col("target_copies") == 1, F.col("target")).alias("target"),
    )
    unresolved = operations.where("tail is not null and scheduled_utc is null").groupBy("tail").agg(
        F.collect_set("FlightDate").alias("unresolved_dates")
    )
    rows = parent.join(target_operations, "flight_id", "left").select(
        *parent.columns, "target_copies", "target",
        F.col("target.tail").alias("tail"), F.col("target.scheduled_utc").alias("scheduled_utc"),
    ).join(timeline, ["tail", "scheduled_utc"], "left").join(unresolved, "tail", "left")
    cutoff = F.col("scheduled_departure_utc")
    gap = cutoff.cast("double") - F.col("prior.scheduled_utc").cast("double")
    # Outcome requires 15-minute publication buffer before departure cutoff.
    prior_available = F.col("prior.actual_arrival_utc") + F.expr(f"INTERVAL {OUTCOME_LAG_MINUTES} MINUTES")
    # Bound history inspection: maximum turnaround gap is 24 hours.
    earliest = F.greatest(F.col("prior.scheduled_utc"), cutoff - F.expr(f"INTERVAL {MAX_GAP_HOURS} HOURS"))
    # Check for unresolvable dates within the turnaround window (-14h / +36h timezone envelope).
    ambiguous_history = F.exists(
        "unresolved_dates",
        lambda day: ((day.cast("timestamp") - F.expr("INTERVAL 14 HOURS") <= cutoff)
                     & (day.cast("timestamp") + F.expr("INTERVAL 36 HOURS") >= earliest)),
    )
    link_failure = (
        F.when(cutoff.isNull(), "invalid_target_cutoff")
        .when(F.col("FlightDate") < F.lit(first_reliable_date), "left_censored")
        .when(F.col("target_copies").isNull(), "target_unresolved")
        .when(F.col("target_copies") > 1, "ambiguous_target_or_timeline")
        .when(~F.col("scheduled_utc").eqNullSafe(cutoff), "target_unresolved")
        .when(F.col("tail").isNull(), "missing_tail")
        .when((F.col("group_size") != 1) | ambiguous_history, "ambiguous_target_or_timeline")
        .when(F.col("prior_group_size").isNull(), "no_predecessor")
        .when((F.col("prior_group_size") != 1) | F.col("prior.conflict"), "ambiguous_predecessor")
        .when(~F.col("prior.Dest").eqNullSafe(F.col("Origin")), "airport_discontinuity")
        .when((gap <= 0) | (gap > MAX_GAP_HOURS * 3600), "gap_exceeded")
        .when(
            F.col("prior.scheduled_arrival_utc").isNull()
            | (F.col("prior.scheduled_arrival_utc") > cutoff),
            "invalid_prior_schedule",
        )
    )
    outcome_failure = (
        F.when(~F.col("prior.completed"), "prior_operation_ineligible")
        .when(prior_available.isNull(), "invalid_prior_outcome")
        .when(prior_available > cutoff, "prior_outcome_unavailable")
    )
    linked = link_failure.isNull()
    usable = linked & outcome_failure.isNull()
    prior_arrival = F.col("prior.actual_arrival_utc").cast("long")
    actual_departure = F.col("target.actual_departure_utc")
    historical_turn = linked & F.col("prior.completed") & F.col("target.completed")
    # An early departure cannot contribute to the reference used on its own flight.
    turn_available = F.greatest(actual_departure, cutoff) + F.expr(f"INTERVAL {OUTCOME_LAG_MINUTES} MINUTES")
    return rows.select(
        *parent.columns,
        F.when(linked, (cutoff.cast("double") - F.col("prior.scheduled_arrival_utc").cast("double")) / 60)
        .alias(ROTATION_NUMERIC[0]),
        F.when(usable, F.col("prior.arrival_delay")).alias(ROTATION_NUMERIC[1]),
        (~linked).cast("int").alias(ROTATION_FLAGS[0]),
        (~usable).cast("int").alias(ROTATION_FLAGS[1]),
        F.when(usable, (cutoff.cast("long") - prior_arrival) / 60).alias("_remaining_turn_minutes"),
        F.when(historical_turn, (actual_departure.cast("long") - prior_arrival) / 60).alias("_observed_turn_minutes"),
        F.when(historical_turn & actual_departure.isNotNull(), turn_available).alias("_turn_available_utc"),
        F.coalesce(link_failure, outcome_failure, F.lit("usable")).alias("rotation_reason"),
        F.col("prior.flight_id").alias("prior_flight_id"),
        F.col("prior.scheduled_utc").alias("prior_scheduled_departure_utc"),
        F.col("prior.scheduled_arrival_utc").alias("prior_scheduled_arrival_utc"),
        F.col("prior.actual_arrival_utc").alias("prior_actual_arrival_utc"),
        prior_available.alias("prior_arrival_available_utc"),
        (gap / 60).alias("prior_departure_gap_minutes"),
        (F.col("target.actual_arrival_utc") + F.expr(f"INTERVAL {OUTCOME_LAG_MINUTES} MINUTES"))
        .alias("label_available_utc"),
        ((F.col("target.arrival_delay") >= 15).cast("int") == F.col("ArrDel15")).alias("label_matches"),
    )


def add_turnaround_shortfall(linked: DataFrame, validation_bound: datetime) -> DataFrame:
    """Compare observed time left on the ground with a historical carrier-airport median."""
    bound = F.lit(validation_bound).cast("timestamp")
    cutoff = F.col("scheduled_departure_utc")
    history_month = "_turn_history_month"
    keys = ["Origin", "Reporting_Airline", history_month]
    reference = (
        linked.where((F.col("split") == "train") & (F.col("_turn_available_utc") < bound)
                     & (F.col("_observed_turn_minutes") > 0)
                     & (F.col("_observed_turn_minutes") <= MAX_TYPICAL_TURN_MINUTES))
        .withColumn(history_month, F.trunc("_turn_available_utc", "month"))
        .groupBy(*keys).agg(F.count("*").alias("_turn_samples"),
                           F.percentile_approx("_observed_turn_minutes", 0.5).alias("_typical_turn_minutes"))
        .where(F.col("_turn_samples") >= MIN_TURN_HISTORY).drop("_turn_samples")
    )
    # Training uses the previous UTC month. Validation freezes the last month at the training boundary.
    month = F.when((F.col("split") == "val") & (cutoff >= bound),
                   F.trunc(bound - F.expr("INTERVAL 1 SECOND"), "month")).otherwise(
        F.add_months(F.trunc(cutoff, "month"), -1)
    )
    difference = F.col("_typical_turn_minutes") - F.col("_remaining_turn_minutes")
    return (linked.withColumn(history_month, month).join(reference, keys, "left")
            .withColumn(TURNAROUND_SHORTFALL, F.when(difference.isNotNull(), F.greatest(F.lit(0.0), difference)))
            .withColumn(TURNAROUND_MISSING, F.col(TURNAROUND_SHORTFALL).isNull().cast("int"))
            .drop(history_month, "_typical_turn_minutes", "_remaining_turn_minutes",
                  "_observed_turn_minutes", "_turn_available_utc"))
