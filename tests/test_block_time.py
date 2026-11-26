import json
from datetime import date, datetime, timezone

import pytest
from pyspark.sql import functions as F

from scripts.aircraft_rotation.build_features import RECOVERY_FEATURES, TURNAROUND_SHORTFALL, add_recovery_shortfall
from scripts.bts import block_time as block
from scripts.bts.download import build_download_url
from scripts.bts.recent_delays import EVENT_COLUMNS
from scripts.common.io import sha256


SPLITS = {
    "train": {"start_date": "2024-01-01", "end_date": "2024-09-30"},
    "val": {"start_date": "2024-10-01", "end_date": "2024-11-30"},
}


def utc(*parts):
    return datetime(*parts, tzinfo=timezone.utc)


@pytest.mark.unit
def test_release_manifest_validation(tmp_path):
    config = {"bts_features": {}}
    assert block.load_block_history_manifest(tmp_path, config)["status"] == "absent"
    manifest = tmp_path / block.DEFAULT_MANIFEST
    manifest.parent.mkdir(parents=True)
    source = manifest.parent / "january.csv"
    source.write_text("synthetic")
    entry = {
        "source_month": "2024-01", "csv_path": source.name, "sha256": sha256(source),
        "available_utc": "2024-03-02T05:00:00Z", "source_url": "https://transtats.bts.gov/PREZIP/january.zip",
        "evidence_url": "https://web.archive.org/web/20240301/https://transtats.bts.gov/PREZIP/january.zip",
        "evidence_note": "Synthetic fixture treating March 1 as a local Eastern release day.",
    }

    def load(entries, version=1):
        manifest.write_text(json.dumps({"schema_version": version, "months": entries}))
        return block.load_block_history_manifest(tmp_path, config)

    parsed = load([entry])
    assert parsed["months"][0]["csv_path"] == f"{block.HISTORY_ROOT}/january.csv"
    assert parsed["manifest_sha256"] == sha256(manifest)
    bad_entries = (
        [entry, entry], [{**entry, "csv_path": "../outside.csv"}],
        [{**entry, "csv_path": str(source)}], [{**entry, "sha256": "0" * 64}],
        [{**entry, "evidence_url": "https://bts.gov.evil.invalid/file"}],
        [{**entry, "source_url": "https://user@bts.gov/file"}],
        [{**entry, "source_url": "https://bts.gov:444/file"}],
        [{**entry, "available_utc": "2024-02-01T11:59:59Z"}],
        [{**entry, "available_utc": "2024-03-02T05:00:00+00:00"}],
        [{**entry, "extra": False}],
    )
    for records in bad_entries:
        with pytest.raises(ValueError):
            load(records)
    with pytest.raises(ValueError):
        load([], version=True)
    with pytest.raises(OSError):
        load([{**entry, "csv_path": "missing.csv"}])


@pytest.mark.unit
def test_default_history_reuses_bts_downloads_with_archive_cutoffs(tmp_path):
    config = {
        "splits": SPLITS,
        "datasets": {"bts": {"output_directory": "data/bronze/bts"}},
        "bts_features": {"outbound_schedule_slack": {"bts_archive_dates": {
            "2024-01": "2024-04-11", "2024-07": "2024-09-27", "2024-08": "2024-10-21",
        }}},
    }
    for month in (1, 7, 8, 10):
        name = build_download_url(2024, month).rsplit("/", 1)[1].replace(".zip", ".csv")
        path = tmp_path / "data/bronze/bts/2024" / f"{month:02d}" / name
        path.parent.mkdir(parents=True)
        path.write_text(f"fixture {month}")
    history = block.resolve_block_history(tmp_path, config)
    assert history["status"] == "retrospective_bts_archive"
    assert history["manifest_sha256"] is None
    assert [(entry["source_month"], entry["available_utc"]) for entry in history["months"]] == [
        ("2024-01", "2024-04-12T12:00:00Z"), ("2024-07", "2024-09-28T12:00:00Z"),
    ]
    for entry in history["months"]:
        assert entry["sha256"] == sha256(tmp_path / entry["csv_path"])
    manifest = tmp_path / block.DEFAULT_MANIFEST
    manifest.parent.mkdir()
    manifest.write_text('{"schema_version": 1, "months": []}')
    assert block.resolve_block_history(tmp_path, config)["status"] == "verified"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="Invalid block history manifest"):
        block.resolve_block_history(tmp_path, config)


def test_block_duration_requires_consistent_historical_operations(spark):
    airports = {
        "JFK": [{"time_zone": "America/New_York", "valid_from": "2000-01-01", "valid_to": "2100-01-01"}],
        "ORD": [{"time_zone": "America/Chicago", "valid_from": "2000-01-01", "valid_to": "2100-01-01"}],
    }
    fields = [*EVENT_COLUMNS, "CRSArrTime", "Cancelled", "_source_month", "_release"]
    base = dict(zip(fields, (
        "2024-01-05", "AA", "1", "JFK", "ORD", "2330", "120",
        "2330", "0", "0030", "0", "0", "0030", "0", "2024-01", utc(2024, 3, 2),
    )))
    cases = [
        {}, {}, {"Flight_Number_Reporting_Airline": "2"},
        {"Flight_Number_Reporting_Airline": "2", "ArrTime": "BAD"},
        {"Flight_Number_Reporting_Airline": "3", "Cancelled": "1"},
        {"Flight_Number_Reporting_Airline": "4", "Diverted": "1"},
        {"Flight_Number_Reporting_Airline": "5", "_release": utc(2024, 1, 6, 6, 30)},
        {"Flight_Number_Reporting_Airline": "6", "FlightDate": "2024-02-01"},
        {"Flight_Number_Reporting_Airline": "7", "FlightDate": "2024-03-10", "_source_month": "2024-03",
         "CRSDepTime": "0230", "DepTime": "0230", "_release": utc(2024, 5, 1)},
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
        "5": ("completion_not_before_release", 120.0), "8": ("invalid_duration", 5.0),
        "7": ("invalid_or_ineligible_operation", None),
        "9": ("usable", 10.0), "10": ("usable", 1440.0), "11": ("invalid_duration", 1441.0),
    }
    assert sum(r["count"] for r in rejected.collect()) == 1


def test_reference_uses_only_published_training_history(spark):
    release = utc(2024, 7, 1)
    history = spark.createDataFrame(
        [(f"history-{i}", "AA", "JFK", "ORD", release, 100.0, "usable") for i in range(100)]
        + [(f"fallback-{i}", "DL", "JFK", "ORD", release, 200.0, "usable") for i in range(99)]
        + [(f"late-{i}", "AA", "JFK", "ORD", utc(2024, 9, 30, 10), 10.0, "usable") for i in range(100)],
        "flight_id string, Reporting_Airline string, Origin string, Dest string, _release timestamp, "
        "_duration double, reason string",
    )
    targets = spark.createDataFrame([
        ("equal", "AA", "JFK", "ORD", "train", utc(2024, 7, 15), 120.0, 0),
        ("training", "AA", "JFK", "ORD", "train", utc(2024, 8, 15), 100.0, 1),
        ("negative", "AA", "JFK", "ORD", "val", utc(2024, 10, 1), 95.0, 1),
        ("fallback", "DL", "JFK", "ORD", "val", utc(2024, 10, 1), 110.0, 0),
        ("frozen", "AA", "JFK", "ORD", "test", utc(2024, 12, 1), 120.0, 0),
        ("reverse", "AA", "ORD", "JFK", "val", utc(2024, 10, 1), 100.0, 0),
        ("invalid", "AA", "JFK", "ORD", "val", utc(2024, 10, 1), float("nan"), 0),
        ("early", "AA", "JFK", "ORD", "val", utc(2024, 9, 30, 9), 100.0, 1),
        ("null", "AA", "JFK", "ORD", "val", None, 100.0, 0),
    ], "flight_id string, Reporting_Airline string, Origin string, Dest string, split string, "
       "_cutoff timestamp, CRSElapsedTime double, ArrDel15 int").withColumn("FlightDate", F.lit(date(2024, 1, 1)))
    releases = [{"available_utc": "2024-07-01T00:00:00Z"}, {"available_utc": "2024-09-30T10:00:00Z"}]
    value, flag = block.SLACK_FEATURES
    rows = block.add_outbound_schedule_slack(targets, history, SPLITS, releases).collect()
    observed = {row.flight_id: (row[value], row[flag], row._block_reason, row.ArrDel15) for row in rows}
    assert observed == {
        "equal": (None, 1, "no_released_history", 0), "training": (0.0, 0, "carrier_route", 1),
        "negative": (-5.0, 0, "carrier_route", 1), "fallback": (10.0, 0, "route", 0),
        "frozen": (20.0, 0, "carrier_route", 0), "reverse": (None, 1, "insufficient_history", 0),
        "invalid": (None, 1, "invalid_scheduled_duration", 0), "early": (None, 1, "invalid_cutoff", 1),
        "null": (None, 1, "invalid_cutoff", 0),
    }
    assert {row._block_samples for row in rows if row._block_samples is not None} == {100, 199}
    missing = block.add_outbound_schedule_slack(targets, None, SPLITS, [])
    assert missing.where(F.col(value).isNotNull() | (F.col(flag) != 1)).count() == 0


def test_shortfall_after_recovery_preserves_missingness(spark):
    slack = block.SLACK_FEATURES[0]
    frame = spark.createDataFrame([(25., 18.), (25., 4.), (0., 7.), (0., -7.), (None, 0.), (10., None)],
                                  f"{TURNAROUND_SHORTFALL} double, {slack} double")
    rows = add_recovery_shortfall(frame).select(*RECOVERY_FEATURES).collect()
    assert [tuple(row) for row in rows] == [(7., 0), (21., 0), (-7., 0), (7., 0), (None, 1), (None, 1)]
