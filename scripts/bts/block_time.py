"""Build cumulative historical gate-to-gate references from completed training flights."""

from pyspark.sql import functions as F

BLOCK_TIME_SIDECAR = "_block_time.json"
SLACK_FEATURES = ("outbound_schedule_slack_minutes", "outbound_schedule_slack_missing")
PERCENTILE = 0.5
ACCURACY = 10_000
MIN_SAMPLES = 100
MIN_DURATION = 10
MAX_DURATION = 1440
ROUTE = ["Origin", "Dest"]
CARRIER_ROUTE = ["Reporting_Airline", *ROUTE]
KEY = ["FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", *ROUTE, "CRSDepTime"]
POLICY = {
    "version": 3,
    "percentile": PERCENTILE,
    "accuracy": ACCURACY,
    "min_samples": MIN_SAMPLES,
    "duration_bounds_minutes": [MIN_DURATION, MAX_DURATION],
    "schedule_scope": "Retrospective BTS schedule. Strict replay requires as-of schedule evidence.",
    "history_scope": "Completed prior months are assumed available at the next month boundary. The training reference is frozen for evaluation.",
}


def build_block_duration_history(raw, airports, train_dates):
    """Resolve historical gate clocks, discarding conflicting source rows."""
    from .recent_delays import EVENT_COLUMNS, event_clocks, scheduled_clocks

    fields = [*EVENT_COLUMNS, "CRSArrTime", "Cancelled"]
    rows = raw.select(*[F.trim(name).alias(name) for name in fields], "_source_month")
    valid_key = F.col("CRSDepTime").rlike(r"^\d{1,4}$")
    for name in KEY[1:-1]:
        valid_key = valid_key & (F.length(name) > 0)
    rows = rows.withColumn("FlightDate", F.expr("try_cast(FlightDate as date)"))
    valid_date = (
        (F.date_format("FlightDate", "yyyy-MM") == F.col("_source_month"))
        & F.col("FlightDate").between(train_dates["start_date"], train_dates["end_date"])
    )
    rows = rows.withColumn("_eligible", F.coalesce(valid_key & valid_date, F.lit(False)))
    ineligible = (rows.where("not _eligible").groupBy("_source_month").count()
                  .withColumn("reason", F.lit("invalid_identity_or_outside_training")))
    rows = (rows.where("_eligible").withColumn("CRSDepTime", F.lpad("CRSDepTime", 4, "0"))
            .withColumn("flight_id", F.concat_ws("_", *KEY)))
    payload = F.struct(*[name for name in fields if name not in KEY])
    grouped = (rows.groupBy("_source_month", "flight_id", *KEY)
               .agg(F.min(payload).alias("_first"), F.max(payload).alias("_last"))
               .select("*", "_first.*"))

    def gate_clocks(row):
        if not scheduled_clocks(row, airports)[1]:
            return None, None
        return event_clocks(row, airports)

    clocks = F.udf(gate_clocks, "departure string, arrival string")
    verified = F.when(
        (F.col("_first") == F.col("_last")) & (F.expr("try_cast(Cancelled as double)") == 0)
        & (F.expr("try_cast(Diverted as double)") == 0), clocks(F.struct(*EVENT_COLUMNS, "CRSArrTime")),
    )
    grouped = (grouped.withColumn("_clocks", verified)
               .withColumn("_arrived", F.to_timestamp("_clocks.arrival"))
               .withColumn("_duration", (F.col("_arrived").cast("long")
                                         - F.to_timestamp("_clocks.departure").cast("long")) / 60))
    reason = (F.when(F.col("_first") != F.col("_last"), "conflicting_identity")
              .when(F.col("_duration").isNull(), "invalid_or_ineligible_operation")
              .when(~F.col("_duration").between(MIN_DURATION, MAX_DURATION), "invalid_duration")
              .otherwise("usable"))
    return (grouped.select("flight_id", "FlightDate", *CARRIER_ROUTE, "_source_month", "_arrived", "_duration",
                           reason.alias("reason")), ineligible.select("_source_month", "reason", "count"))


def build_block_time_reference(history, snapshots):
    eligible = (history.where("reason = 'usable'").join(F.broadcast(snapshots), "Origin")
                .where((F.col("FlightDate") < F.col("_snapshot"))
                       & (F.col("_arrived") < F.col("_snapshot_cutoff"))))
    references = [
        eligible.groupBy("_snapshot", *keys).agg(
            F.countDistinct("flight_id").alias(f"{prefix}_samples"),
            F.percentile_approx("_duration", PERCENTILE, ACCURACY).alias(prefix),
        ).where(F.col(f"{prefix}_samples") >= MIN_SAMPLES)
        for keys, prefix in ((CARRIER_ROUTE, "_carrier_block"), (ROUTE, "_route_block"))
    ]
    return eligible, references


def add_outbound_schedule_slack(flights, history, splits, airports):
    """Use completed prior months in training and one frozen reference for evaluation."""
    from .recent_delays import add_cutoffs

    first_validation_day = F.lit(splits["val"]["start_date"]).cast("date")
    first_training_month = F.trunc(F.lit(splits["train"]["start_date"]).cast("date"), "month")
    snapshot = F.when(F.col("split") == "train", F.trunc("FlightDate", "month")).otherwise(
        first_validation_day
    )
    # Convert the calendar boundary at each origin so an overnight flight must actually have arrived.
    dates = flights.select("Origin", snapshot.alias("FlightDate")).distinct().withColumn("CRSDepTime", F.lit("0000"))
    bounds = add_cutoffs(dates, airports).select(
        "Origin", F.col("FlightDate").alias("_snapshot"), F.col("_cutoff").alias("_snapshot_cutoff"),
    )
    rows = (flights.withColumn("_snapshot", snapshot).join(F.broadcast(bounds), ["Origin", "_snapshot"], "left")
            .withColumn("_snapshot", F.when(F.col("_cutoff") >= F.col("_snapshot_cutoff"), F.col("_snapshot"))))
    eligible, references = build_block_time_reference(history, bounds)
    if eligible.select("flight_id", "_snapshot").join(
        rows.select("flight_id", "_snapshot"), ["flight_id", "_snapshot"], "left_semi",
    ).limit(1).count():
        raise ValueError("A flight contributed to its own block-time reference.")
    for frame, keys in zip(references, (CARRIER_ROUTE, ROUTE)):
        rows = rows.join(frame, ["_snapshot", *keys], "left")
    scheduled = F.expr("try_cast(CRSElapsedTime as double)")
    reason = (F.when(F.col("_snapshot").isNull(), "invalid_cutoff")
              .when(~F.coalesce(scheduled.between(MIN_DURATION, MAX_DURATION), F.lit(False)), "invalid_scheduled_duration")
              .when(F.col("_snapshot") <= first_training_month, "no_prior_history")
              .when(F.col("_carrier_block").isNotNull(), "carrier_route")
              .when(F.col("_route_block").isNotNull(), "route")
              .otherwise("insufficient_history"))
    value, missing = SLACK_FEATURES
    return (rows.withColumn("_block_reason", reason)
            .withColumn(value, F.when(F.col("_block_reason").isin("carrier_route", "route"),
                                      scheduled - F.coalesce("_carrier_block", "_route_block")))
            .withColumn(missing, F.col(value).isNull().cast("int"))
            .withColumn("_block_samples", F.when(F.col(value).isNotNull(),
                        F.coalesce("_carrier_block_samples", "_route_block_samples")))
            .drop("_snapshot_cutoff", "_carrier_block", "_route_block", "_carrier_block_samples", "_route_block_samples"))
