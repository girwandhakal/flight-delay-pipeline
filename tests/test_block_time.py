from datetime import date, datetime, timezone


from scripts.aircraft_rotation.build_features import RECOVERY_FEATURES, TURNAROUND_SHORTFALL, add_recovery_shortfall
from scripts.bts import block_time as block
from scripts.bts.recent_delays import EVENT_COLUMNS


SPLITS = {
    "train": {"start_date": "2024-01-01", "end_date": "2024-09-30"},
    "val": {"start_date": "2024-10-01", "end_date": "2024-11-30"},
}


def utc(*parts):
    return datetime(*parts, tzinfo=timezone.utc)


def test_block_duration_requires_consistent_historical_operations(spark):
    airports = {
        "JFK": [{"time_zone": "America/New_York", "valid_from": "2000-01-01", "valid_to": "2100-01-01"}],
        "ORD": [{"time_zone": "America/Chicago", "valid_from": "2000-01-01", "valid_to": "2100-01-01"}],
    }
    fields = [*EVENT_COLUMNS, "CRSArrTime", "Cancelled", "_source_month"]
    base = dict(zip(fields, (
        "2024-01-05", "AA", "1", "JFK", "ORD", "2330", "120",
        "2330", "0", "0030", "0", "0", "0030", "0", "2024-01",
    )))
    cases = [
        {}, {}, {"Flight_Number_Reporting_Airline": "2"},
        {"Flight_Number_Reporting_Airline": "2", "ArrTime": "BAD"},
        {"Flight_Number_Reporting_Airline": "3", "Cancelled": "1"},
        {"Flight_Number_Reporting_Airline": "4", "Diverted": "1"},
        {"Flight_Number_Reporting_Airline": "6", "FlightDate": "2024-02-01"},
        {"Flight_Number_Reporting_Airline": "7", "FlightDate": "2024-03-10", "_source_month": "2024-03",
         "CRSDepTime": "0230", "DepTime": "0230"},
        {"Flight_Number_Reporting_Airline": "8", "ArrDelay": "-115", "ArrTime": "2235"},
        {"Flight_Number_Reporting_Airline": "9", "ArrDelay": "-110", "ArrTime": "2240"},
        {"Flight_Number_Reporting_Airline": "10", "ArrDelay": "1320", "ArrTime": "2230"},
        {"Flight_Number_Reporting_Airline": "11", "ArrDelay": "1321", "ArrTime": "2231"},
    ]
    raw = spark.createDataFrame([tuple((base | changes)[name] for name in fields) for changes in cases], fields)
    history, rejected = block.build_block_duration_history(raw, airports, SPLITS["train"])
    result = {r.flight_id.split("_")[2]: (r.reason, r._duration) for r in history.collect()}
    assert result == {
        "1": ("usable", 120.0), "2": ("conflicting_identity", None),
        "3": ("invalid_or_ineligible_operation", None), "4": ("invalid_or_ineligible_operation", None),
        "8": ("invalid_duration", 5.0),
        "7": ("invalid_or_ineligible_operation", None),
        "9": ("usable", 10.0), "10": ("usable", 1440.0), "11": ("invalid_duration", 1441.0),
    }
    assert sum(r["count"] for r in rejected.collect()) == 1


def test_slack_uses_cumulative_prior_month_medians_and_freezes_at_validation(spark):
    airports = {
        origin: [{"time_zone": "America/New_York", "valid_from": "2000-01-01", "valid_to": "2100-01-01"}]
        for origin in ("JFK", "ORD")
    }
    history = spark.createDataFrame(
        [(f"jan-{i}", "AA", "JFK", "ORD", date(2024, 1, 5), utc(2024, 1, 5, 12),
          80.0 if i < 40 else 100.0, "usable") for i in range(100)]
        + [(f"apr-{i}", "AA", "JFK", "ORD", date(2024, 4, 5), utc(2024, 4, 5, 12), 160.0, "usable")
           for i in range(101)]
        + [(f"may-{i}", "AA", "JFK", "ORD", date(2024, 5, 5), utc(2024, 5, 5, 12), 200.0, "usable")
           for i in range(203)]
        + [(f"sep-{i}", "AA", "JFK", "ORD", date(2024, 9, 29), utc(2024, 9, 29, 12), 250.0, "usable")
           for i in range(405)]
        + [(f"overnight-{i}", "AA", "JFK", "ORD", date(2024, 9, 30), utc(2024, 10, 1, 5), 300.0, "usable")
           for i in range(810)],
        "flight_id string, Reporting_Airline string, Origin string, Dest string, FlightDate date, "
        "_arrived timestamp, _duration double, reason string",
    )
    targets = spark.createDataFrame([
        ("first", "AA", "JFK", "ORD", "train", date(2024, 1, 15), utc(2024, 1, 15), 120.0, 0),
        ("feb", "AA", "JFK", "ORD", "train", date(2024, 2, 15), utc(2024, 2, 15), 100.0, 1),
        ("may", "AA", "JFK", "ORD", "train", date(2024, 5, 15), utc(2024, 5, 15), 170.0, 0),
        ("jun", "DL", "JFK", "ORD", "train", date(2024, 6, 15), utc(2024, 6, 15), 195.0, 1),
        ("val", "AA", "JFK", "ORD", "val", date(2024, 10, 1), utc(2024, 10, 1, 12), 260.0, 1),
        ("frozen", "AA", "JFK", "ORD", "test", date(2024, 12, 1), utc(2024, 12, 1, 12), 260.0, 0),
        ("reverse", "AA", "ORD", "JFK", "val", date(2024, 10, 1), utc(2024, 10, 1, 12), 100.0, 0),
        ("invalid", "AA", "JFK", "ORD", "val", date(2024, 10, 1), utc(2024, 10, 1, 12), float("nan"), 0),
        ("null", "AA", "JFK", "ORD", "val", date(2024, 10, 1), None, 100.0, 0),
    ], "flight_id string, Reporting_Airline string, Origin string, Dest string, split string, FlightDate date, "
       "_cutoff timestamp, CRSElapsedTime double, ArrDel15 int")
    value, flag = block.SLACK_FEATURES
    rows = block.add_outbound_schedule_slack(targets, history, SPLITS, airports).collect()
    assert {row.flight_id: (row[value], row[flag], row._block_reason, row.ArrDel15) for row in rows} == {
        "first": (None, 1, "no_prior_history", 0),
        "feb": (0.0, 0, "carrier_route", 1),
        "may": (10.0, 0, "carrier_route", 0),
        "jun": (-5.0, 0, "route", 1),
        "val": (10.0, 0, "carrier_route", 1),
        "frozen": (10.0, 0, "carrier_route", 0),
        "reverse": (None, 1, "insufficient_history", 0),
        "invalid": (None, 1, "invalid_scheduled_duration", 0),
        "null": (None, 1, "invalid_cutoff", 0),
    }
    assert {row.flight_id: row._block_samples for row in rows if row._block_samples is not None} == {
        "feb": 100, "may": 201, "jun": 404, "val": 809, "frozen": 809,
    }


def test_net_turnaround_shortfall_preserves_signed_values_and_missingness(spark):
    slack = block.SLACK_FEATURES[0]
    frame = spark.createDataFrame([(25., 18.), (25., 4.), (0., 7.), (0., -7.), (None, 0.), (10., None)],
                                  f"{TURNAROUND_SHORTFALL} double, {slack} double")
    result = add_recovery_shortfall(frame)
    assert RECOVERY_FEATURES == ("net_turnaround_shortfall_minutes", "net_turnaround_shortfall_missing")
    rows = result.select(*RECOVERY_FEATURES).collect()
    assert [tuple(row) for row in rows] == [(7., 0), (21., 0), (-7., 0), (7., 0), (None, 1), (None, 1)]
