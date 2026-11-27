"""Synthetic airport event, calendar and dataset freshness contracts."""

from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
from pyspark.sql import Row, functions as F

from scripts.bts import recent_delays as recent
from scripts.common.io import sha256, write_json


def airports():
    return {code: [{"Origin": code, "time_zone": zone,
                    "valid_from": "2023-01-01", "valid_to": "2026-01-01"}]
            for code, zone in (("JFK", "America/New_York"), ("LAX", "America/Los_Angeles"))}


def test_rolling_boundaries_roles_self_exclusion_and_future_invariance(spark):
    cutoff = datetime(2024, 10, 1, 0, 30, tzinfo=timezone.utc)
    epoch = int(cutoff.timestamp())
    flights = spark.createDataFrame([
        ("target", "JFK", "LAX", cutoff),
        ("empty", "XXX", "YYY", cutoff),
        ("invalid", "JFK", "LAX", None),
    ], "flight_id string, Origin string, Dest string, _cutoff timestamp")
    events = spark.createDataFrame([
        ("lower", "JFK", "departure", epoch - 7200, 200.0),
        ("old", "JFK", "departure", epoch - 7201, 200.0),
        ("a", "JFK", "departure", epoch - 7199, -10.0),
        ("b", "JFK", "departure", epoch, 15.0),
        ("target", "JFK", "departure", epoch - 60, -1.0),
        ("future", "JFK", "departure", epoch + 1, 300.0),
        ("wrong-role", "JFK", "arrival", epoch, 100.0),
        ("c", "LAX", "arrival", epoch - 60, 30.0),
        ("d", "LAX", "arrival", epoch, 14.0),
        ("taxi-lower", "JFK", "taxi", epoch - 7200, 45.0),
        ("taxi-near", "JFK", "taxi", epoch - 7199, 10.0),
        ("taxi-at", "JFK", "taxi", epoch, 30.0),
        ("taxi-late", "JFK", "taxi", epoch - 60, 100.0),
        ("target", "JFK", "taxi", epoch - 960, 90.0),
        ("future", "JFK", "taxi", epoch + 1, 100.0),
    ], "flight_id string, airport string, role string, event_seconds long, minutes double")
    rows = {r.flight_id: r for r in recent.add_recent_delays(flights, events).collect()}
    target = rows["target"]
    assert {"origin_departure_count_2h", "destination_arrival_count_2h"}.isdisjoint(target.asDict())
    assert target.origin_departure_delay_rate_2h == .5
    assert target.origin_departure_mean_delay_minutes_2h == 7.5
    assert target.destination_arrival_delay_rate_2h == .5
    assert target.destination_arrival_mean_delay_minutes_2h == 22.0
    assert "origin_departure_taxi_out_count_2h" not in target.asDict()
    assert target[recent.TAXI_MEAN] == pytest.approx(185 / 4)
    assert rows["empty"][recent.TAXI_MEAN] is None
    assert rows["empty"].origin_departure_delay_rate_2h is None
    assert rows["empty"].destination_arrival_mean_delay_minutes_2h is None
    assert all(rows["invalid"][c] is None for c in recent.ROLLING_FEATURES if c != recent.TAXI_SURGE)
    without_future = recent.add_recent_delays(flights, events.filter("flight_id != 'future'")).collect()
    assert {r.flight_id: r.asDict() for r in without_future} == {key: value.asDict() for key, value in rows.items()}


def raw_event(**changes):
    row = dict(FlightDate=date(2024, 1, 31), Reporting_Airline="AA", Flight_Number_Reporting_Airline="1",
               Origin="JFK", Dest="LAX", CRSDepTime="2300", CRSElapsedTime="360",
               DepTime="2330", DepDelay="30", ArrTime="0230", ArrDelay="30", Diverted="0",
               TaxiOut="40", WheelsOff="0010", Cancelled="0")
    row.update(changes)
    return row


def test_raw_events_validate_clocks_and_preserve_departures_without_arrival_labels(spark):
    raw_rows = [raw_event(), raw_event(), raw_event(Flight_Number_Reporting_Airline="2", Diverted="1"),
                raw_event(Flight_Number_Reporting_Airline="3", ArrDelay=None),
                raw_event(Flight_Number_Reporting_Airline="4", DepTime="9999", ArrTime="9999"),
                raw_event(Flight_Number_Reporting_Airline="5", DepDelay="NaN", ArrDelay="Infinity"),
                raw_event(Flight_Number_Reporting_Airline="6", DepTime="2331", ArrTime="0231"),
                raw_event(Flight_Number_Reporting_Airline="7", CRSDepTime="23000"),
                raw_event(Flight_Number_Reporting_Airline="8", ArrDelay=None),
                raw_event(Flight_Number_Reporting_Airline="8", DepTime="0000", DepDelay="60", ArrDelay=None),
                raw_event(Reporting_Airline=None), raw_event(Flight_Number_Reporting_Airline=""),
                raw_event(Flight_Number_Reporting_Airline="9", WheelsOff="0009", ArrDelay=None),
                raw_event(Flight_Number_Reporting_Airline="10", Cancelled="1", ArrDelay=None)]
    raw = spark.createDataFrame([{k: str(v) if v is not None else None for k, v in row.items()} for row in raw_rows])
    result = recent.build_events(raw, airports()).collect()
    departures = [r for r in result if r.role == "departure"]
    arrivals = [r for r in result if r.role == "arrival"]
    assert len(departures) == 5
    assert len(arrivals) == 1
    assert all(r.minutes == 30 for r in departures + arrivals)
    taxis = [r for r in result if r.role == "taxi"]
    assert len(taxis) == 3
    assert all(r.minutes == 40 for r in taxis)
    assert all(r.event_seconds == int(datetime(2024, 2, 1, 5, 10, tzinfo=timezone.utc).timestamp()) for r in taxis)
    actual_arrival = datetime(2024, 2, 1, 10, 30)
    assert arrivals[0].event_seconds == int((actual_arrival - datetime(1970, 1, 1)).total_seconds())
    with pytest.raises(ValueError, match="BTS event columns missing"):
        recent.build_events(raw.drop("DepDelay"), airports())


def test_taxi_baselines_use_prior_training_snapshots(spark):
    data = [
        ("jan", date(2024, 1, 3), "train", "2024-01-03T20:00:00+00:00", 25.0),
        ("feb", date(2024, 2, 3), "train", "2024-02-03T20:00:00+00:00", 12.0),
        ("late", date(2024, 2, 29), "train", "2024-03-01T06:00:00+00:00", 2.0),
        ("pending", date(2024, 2, 29), "train", "2024-03-01T04:50:00+00:00", 3.0),
        ("val", date(2024, 3, 3), "val", "2024-03-03T20:00:00+00:00", 1.0),
    ]
    flights = spark.createDataFrame(data, "flight_id string, FlightDate date, split string, taken_off string, minutes double")
    events = flights.select("flight_id", "minutes", F.lit("JFK").alias("airport"), F.lit("taxi").alias("role"),
                            F.to_timestamp("taken_off").cast("long").alias("event_seconds"))
    splits = {"train": {"start_date": "2024-01-01", "end_date": "2024-02-29"},
              "val": {"start_date": "2024-03-01"}}
    rows = recent.compute_taxi_baselines(events, flights, splits, airports()).collect()
    assert {row.taxi_snapshot: row.median_taxi_minutes for row in rows} == {"2024-02": 25.0, "evaluation": 12.0}


def test_dst_2400_missing_zones_and_historical_event_intervals(spark):
    rows = [("gap", date(2024, 3, 10), "0230", "JFK"),
            ("repeat", date(2024, 11, 3), "0130", "JFK"),
            ("midnight", date(2024, 1, 1), "2400", "JFK"),
            ("unknown", date(2024, 1, 1), "1200", "XXX")]
    flights = spark.createDataFrame(rows, "flight_id string, FlightDate date, CRSDepTime string, Origin string")
    results = {r.flight_id: r._cutoff.timestamp() if r._cutoff else None
               for r in recent.add_cutoffs(flights, airports()).collect()}
    assert results == {"gap": None, "repeat": None,
                       "midnight": datetime(2024, 1, 2, 5, tzinfo=timezone.utc).timestamp(), "unknown": None}
    refs = airports()
    refs["LAX"] = [{"time_zone": "America/Los_Angeles", "valid_from": "2024-02-01", "valid_to": "2025-01-01"}]
    dep, arr = recent.event_clocks(Row(**raw_event()), refs)
    assert dep == "2024-02-01T04:30:00Z"
    assert arr == "2024-02-01T10:30:00Z"
    assert recent.event_clocks(Row(**raw_event(DepTime="0000", DepDelay="60")), airports())[0] == "2024-02-01T05:00:00Z"
    assert recent.event_clocks(Row(**raw_event(CRSElapsedTime="10", ArrTime="2010", ArrDelay="0")), airports()) == (
        "2024-02-01T04:30:00Z", None)


def test_federal_holidays_actual_observed_adjacent_days_and_year_edges(spark):
    dates = [date(2024, 6, 19), date(2024, 7, 3), date(2024, 7, 4), date(2024, 7, 5),
             date(2024, 11, 28), date(2024, 2, 6), date(2021, 12, 31), date(2022, 1, 1),
             date(2021, 7, 4), date(2021, 7, 5)]
    flights = spark.createDataFrame([(d,) for d in dates], "FlightDate date")
    rows = {r.FlightDate: r for r in recent.add_holidays(flights, "2021-01-01", "2024-12-31").collect()}
    for d in (dates[0], dates[2], dates[4], *dates[6:]):
        assert rows[d].is_federal_holiday == 1
    assert rows[dates[1]].is_day_before_federal_holiday == 1
    assert rows[dates[3]].is_day_after_federal_holiday == 1
    assert all(rows[dates[5]][c] == 0 for c in recent.HOLIDAY_FEATURES)
    assert rows[date(2021, 12, 31)].is_day_before_federal_holiday == 1


def test_feature_freshness_checks_sources_references_policy_and_splits(tmp_path):
    reference = tmp_path / "refs"
    reference.mkdir()
    (reference / "airports.csv").write_text("Origin,time_zone\nJFK,America/New_York\n")
    sources = {"scope": ["2024-01-01", "2024-01-31"], "files": {"airports.csv": sha256(reference / "airports.csv")}}
    write_json(reference / "sources.json", sources)
    config = {"bts_features": {"rolling_window_minutes": 120}, "splits": {"train": {"start_date": "2024-01-01", "end_date": "2024-01-31"}, "test": {"end_date": "2024-01-31"}},
              "datasets": {"bts": {"output_directory": "bronze", "start_month": "2024-01", "end_month": "2024-01"}}, "origin_weather": {"reference_directory": "refs"}}
    bronze = tmp_path / "bronze/2024/01/events.csv"
    bronze.parent.mkdir(parents=True)
    bronze.write_text("events")
    gold = tmp_path / "data/gold/bts"
    for split in config["splits"]:
        (gold / split).mkdir(parents=True)
        (gold / split / "_SUCCESS").touch()
    assert not recent.features_current(tmp_path, config)
    lookup = gold / recent.TAXI_BASELINE_FILE
    write_json(lookup, [])
    block = gold / recent.BLOCK_TIME_SIDECAR
    write_json(block, {})

    def save_manifest():
        write_json(gold / "_features.json", {**recent.feature_manifest(tmp_path, config),
                                             "block_time_sha256": sha256(block),
                                             "taxi_baselines_sha256": sha256(lookup)})

    save_manifest()
    assert recent.features_current(tmp_path, config)
    bronze.write_text("changed events")
    assert not recent.features_current(tmp_path, config)
    save_manifest()
    silver = tmp_path / "data/silver/bts/2024/01/part.parquet"
    silver.parent.mkdir(parents=True)
    silver.write_bytes(b"synthetic fingerprint only")
    assert not recent.features_current(tmp_path, config)
    save_manifest()
    assert recent.features_current(tmp_path, config)
    write_json(lookup, [{"changed": True}])
    assert not recent.features_current(tmp_path, config)
    config["splits"]["val"] = {}
    assert not recent.features_current(tmp_path, config)
    config["bts_features"]["rolling_window_minutes"] = 60
    with pytest.raises(ValueError, match="must be 120"):
        recent.feature_policy(tmp_path, config)
    config["bts_features"]["rolling_window_minutes"] = 120
    (reference / "airports.csv").write_text("modified")
    with pytest.raises(ValueError, match="Reference changed"):
        recent.feature_manifest(tmp_path, config)
