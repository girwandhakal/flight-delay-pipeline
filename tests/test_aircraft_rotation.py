"""Synthetic coverage for rotation linkage and split-safe turnaround history."""

from datetime import date, datetime, timezone

import pytest
from pyspark.sql import functions as F

from scripts.aircraft_rotation import run
from scripts.aircraft_rotation.build_features import (
    MIN_TURN_HISTORY, RAW_COLUMNS, ROTATION_FLAGS, ROTATION_NUMERIC, TURNAROUND_MISSING,
    TURNAROUND_SHORTFALL, add_turnaround_shortfall, build_operations, build_rotation_links,
)


AIRPORTS = {
    airport: [{"time_zone": zone, "valid_from": "2024-01-01", "valid_to": "2025-01-01"}]
    for airport, zone in (("ORD", "America/Chicago"), ("JFK", "America/New_York"))
}


def leg(day, flight, tail, origin, dest, departure, arrival, *, actual=None, delay=0, cancelled=0, diverted=0):
    return dict(
        FlightDate=day, Reporting_Airline="AA", Flight_Number_Reporting_Airline=str(flight),
        Origin=origin, Dest=dest, CRSDepTime=departure, CRSArrTime=arrival, CRSElapsedTime="120",
        DepTime=departure, DepDelay="0", ArrTime=actual or arrival, ArrDelay=str(delay),
        Diverted=str(diverted), Cancelled=str(cancelled), Tail_Number=tail,
    )


def operations(spark, legs):
    raw = spark.createDataFrame([tuple(row[key] for key in RAW_COLUMNS) for row in legs], RAW_COLUMNS)
    return build_operations(raw, AIRPORTS)


def targets(spark, entries):
    return spark.createDataFrame(
        entries, "flight_id string, FlightDate date, Origin string, ArrDel15 int, cutoff string, split string"
    ).withColumn("scheduled_departure_utc", F.to_timestamp("cutoff")).drop("cutoff")


def test_cutoff_ambiguity_and_direct_predecessor(spark):
    legs = [
        leg("2024-09-30", 1, " N1 ", "JFK", "ORD", "2100", "2200", actual="2215", delay=15),
        leg("2024-09-30", 2, "N1", "ORD", "JFK", "2230", "0130"),
        leg("2024-11-01", 3, "N1", "ORD", "JFK", "invalid", "0130"),
        leg("2024-09-30", 4, "N2", "JFK", "ORD", "2100", "2200", actual="2216", delay=16),
        leg("2024-09-30", 5, "N2", "ORD", "JFK", "2230", "0130"),
        leg("2024-09-30", 6, "N3", "JFK", "ORD", "1000", "1100"),
        leg("2024-09-30", 7, "N3", "JFK", "ORD", "2100", "2200", cancelled=1),
        leg("2024-09-30", 8, "N3", "ORD", "JFK", "2230", "0130"),
        leg("2024-09-30", 9, "N4", "JFK", "ORD", "2100", "2200"),
        leg("2024-09-30", 10, "N4", "JFK", "ORD", "2100", "2200"),
        leg("2024-09-30", 11, "N4", "ORD", "JFK", "2230", "0130"),
        leg("2024-09-30", 12, "N5", "JFK", "ORD", "2100", "2200"),
        leg("2024-09-30", 13, "N5", "ORD", "JFK", "2230", "0130"),
        leg("2024-10-01", 14, "N5", "ORD", "JFK", "invalid", "0130"),
        leg("2024-09-30", 15, "N6", "JFK", "ORD", "2100", "2200", actual="2216", delay=16),
        leg("2024-09-30", 15, "N6", "JFK", "ORD", "2100", "2200", actual="2230", delay=30),
        leg("2024-09-30", 16, "N6", "ORD", "JFK", "2230", "0130"),
        leg("2024-09-29", 17, "N7", "JFK", "ORD", "2330", "0030"),
        leg("2024-09-30", 18, "N7", "ORD", "JFK", "2230", "0130"),
        leg("2024-09-30", 19, "N8", "JFK", "ORD", "2100", "2200", actual="2231", delay=31),
        leg("2024-09-30", 20, "N8", "ORD", "JFK", "2230", "0130"),
    ]
    legs[1].update(DepTime="2225", DepDelay="-5")
    parent = targets(spark, [
        (f"2024-09-30_AA_{flight}_ORD_JFK_2230", date(2024, 9, 30), "ORD", 0, "2024-10-01T03:30:00Z", "train")
        for flight in (2, 5, 8, 11, 13, 16, 18, 20)
    ])
    result = {int(row.flight_id.split("_")[2]): row for row in build_rotation_links(
        parent, operations(spark, [*legs, legs[0]]), date(2024, 1, 3)
    ).withColumn("turn_available_epoch", F.col("_turn_available_utc").cast("long")).collect()}
    assert [(result[i].rotation_reason, result[i][ROTATION_NUMERIC[1]]) for i in (2, 5, 8, 11, 13)] == [
        ("usable", 15.0), ("usable", 16.0), ("prior_operation_ineligible", None),
        ("ambiguous_predecessor", None), ("ambiguous_target_or_timeline", None),
    ]
    assert result[2][ROTATION_NUMERIC[0]] == 30.0
    assert result[2]._remaining_turn_minutes == 15.0
    assert result[2]._observed_turn_minutes == 10.0
    assert result[2].turn_available_epoch == datetime(2024, 10, 1, 3, 30, tzinfo=timezone.utc).timestamp()
    assert result[5][ROTATION_FLAGS[0]] == 0
    assert result[5][ROTATION_FLAGS[1]] == 0
    assert result[5]._remaining_turn_minutes == 14.0
    assert result[20].rotation_reason == "prior_outcome_unavailable"
    assert result[20][ROTATION_NUMERIC[1]] is None
    assert result[8].prior_flight_id == "2024-09-30_AA_7_JFK_ORD_2100"
    assert result[16].rotation_reason == "invalid_prior_outcome"
    assert result[16][ROTATION_NUMERIC[0]] == 30.0
    assert result[16][ROTATION_NUMERIC[1]] is None
    assert result[18].rotation_reason == "usable"
    assert result[18].prior_departure_gap_minutes == 1440.0


def test_turnaround_shortfall_uses_only_available_training_history(spark):
    bound = datetime(2024, 9, 30, 10, tzinfo=timezone.utc)
    history = [
        ("aug", "ORD", 35.0, "2024-08-14T12:00:00Z", "2024-08-14T10:00:00Z", MIN_TURN_HISTORY),
        ("sep", "ORD", 50.0, "2024-09-14T12:00:00Z", "2024-09-14T10:00:00Z", MIN_TURN_HISTORY),
        ("late", "ORD", 100.0, "2024-09-30T10:00:00Z", "2024-09-30T08:00:00Z", MIN_TURN_HISTORY + 1),
        ("sparse", "JFK", 25.0, "2024-09-14T12:00:00Z", "2024-09-14T10:00:00Z", MIN_TURN_HISTORY - 1),
    ]
    rows = [(f"{prefix}{i}", "train", origin, duration, available, 20.0, cutoff)
            for prefix, origin, duration, available, cutoff, count in history for i in range(count)]
    rows.extend([
        ("october", "val", "ORD", None, None, 20.0, "2024-10-01T10:00:00Z"),
        ("november", "val", "ORD", None, None, 60.0, "2024-11-01T10:00:00Z"),
        ("no_arrival", "val", "ORD", None, None, None, "2024-10-01T10:00:00Z"),
        ("january", "test", "ORD", None, None, 20.0, "2025-01-01T10:00:00Z"),
        ("sparse_airport", "val", "JFK", None, None, 20.0, "2024-10-01T10:00:00Z"),
    ])
    linked = spark.createDataFrame(
        rows, "flight_id string, split string, Origin string, _observed_turn_minutes double, "
        "available string, _remaining_turn_minutes double, cutoff string",
    ).withColumn("Reporting_Airline", F.lit("AA")).withColumn(
        "_turn_available_utc", F.to_timestamp("available"),
    ).withColumn("scheduled_departure_utc", F.to_timestamp("cutoff"))
    result = {row.flight_id: (row[TURNAROUND_SHORTFALL], row[TURNAROUND_MISSING]) for row in
              add_turnaround_shortfall(linked, bound).select("flight_id", TURNAROUND_SHORTFALL, TURNAROUND_MISSING).collect()}
    assert len(result) == len(rows)
    assert {name: result[name] for name in ("aug0", "sep0", "october", "november", "january", "no_arrival", "sparse_airport")} == {
        "aug0": (None, 1), "sep0": (15.0, 0), "october": (30.0, 0),
        "november": (0.0, 0), "january": (30.0, 0), "no_arrival": (None, 1), "sparse_airport": (None, 1),
    }


def test_rotation_keeps_development_history_and_links_test(spark, tmp_path):
    development = [
        leg("2024-09-27", 1, "N1", "JFK", "ORD", "2100", "2200", actual="2215", delay=15),
        leg("2024-09-27", 2, "N1", "ORD", "JFK", "2230", "0130"),
        leg("2024-11-30", 3, "N2", "JFK", "ORD", "2100", "2200", actual="2216", delay=16),
        leg("2024-11-30", 4, "N2", "ORD", "JFK", "2230", "0130"),
    ]
    extended = development + [
        leg("2024-12-01", 5, "N2", "ORD", "JFK", "invalid", "0130"),
        leg("2024-12-02", 6, "N3", "JFK", "ORD", "2100", "2200", actual="2217", delay=17),
        leg("2024-12-02", 7, "N3", "ORD", "JFK", "2230", "0130"),
        leg("2024-12-02", 8, "N4", "JFK", "ORD", "2100", "2200", actual="2231", delay=31),
        leg("2024-12-02", 9, "N4", "ORD", "JFK", "2230", "0130"),
    ]
    parent = targets(spark, [
        ("2024-09-27_AA_2_ORD_JFK_2230", date(2024, 9, 27), "ORD", 0, "2024-09-28T03:30:00Z", "train"),
        ("2024-11-30_AA_4_ORD_JFK_2230", date(2024, 11, 30), "ORD", 0, "2024-12-01T04:30:00Z", "val"),
        ("2024-12-02_AA_7_ORD_JFK_2230", date(2024, 12, 2), "ORD", 0, "2024-12-03T04:30:00Z", "test"),
        ("2024-12-02_AA_9_ORD_JFK_2230", date(2024, 12, 2), "ORD", 1, "2024-12-03T04:30:00Z", "test"),
    ])
    missing = set(run.BTS_FEATURE_COLUMNS + run.SURFACE_WEATHER_COLUMNS) - set(parent.columns)
    parent = parent.select("*", *(F.lit("AA" if name == "Reporting_Airline" else 1).alias(name)
                                  for name in sorted(missing)))
    first = date(2024, 1, 3)
    bound = datetime(2024, 9, 30, 10, tzinfo=timezone.utc)
    dev_operations = operations(spark, development)
    extended_operations = operations(spark, extended)
    comparison = [*run.IDENTITY, *ROTATION_NUMERIC, *ROTATION_FLAGS]
    expected = build_rotation_links(parent.where("split != 'test'"), dev_operations, first)
    expected = add_turnaround_shortfall(expected, bound).select(*comparison).collect()

    report = run.build_rotation_gold(parent, dev_operations, extended_operations, tmp_path, first, bound)
    assert report["training_exclusions"] == {"unresolved": 0, "after_bound": 0}
    stored = spark.read.parquet(str(tmp_path / "gold/train"), str(tmp_path / "gold/val"))
    assert sorted(stored.select(*comparison).collect()) == sorted(expected)

    test_rows = spark.read.parquet(str(tmp_path / "gold/test")).orderBy("flight_id").collect()
    assert [(row.flight_id, row.ArrDel15, row[ROTATION_NUMERIC[1]], row[ROTATION_FLAGS[1]]) for row in test_rows] == [
        ("2024-12-02_AA_7_ORD_JFK_2230", 0, 17.0, 0),
        ("2024-12-02_AA_9_ORD_JFK_2230", 1, None, 1),
    ]
