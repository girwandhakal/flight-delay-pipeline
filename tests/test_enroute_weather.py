"""Test terminal forecast causality and convective corridor matching."""

from datetime import date, datetime, timezone

import pytest
from pyspark.sql import functions as F

from scripts.enroute_weather import build_features as features, clean_sigmets, clean_taf


@pytest.mark.unit
def test_taf_segments_decode_groups_and_stop_at_expiry():
    published = datetime(2024, 1, 31, 23, 40, tzinfo=timezone.utc)
    rows = clean_taf.parse_taf(
        "TAF AMD KORD 312340Z 0100/0206 22004KT P6SM SCT050 "
        "TEMPO 0101/0103 1 1/2SM +TSRA BKN005CB "
        "FM010400 18012KT 5SM OVC020 WS020/23040KT "
        "PROB30 0105/0108 2SM FZRA OVC008=", published, "TAFORD_202401312340.txt",
    )
    at_hour = {datetime.fromtimestamp(row[3], timezone.utc).hour: row[5:] for row in rows}
    assert at_hour[0][:4] == (0, 0, 0, 0)
    assert at_hour[1][:4] == (1, 1, 0, 0)
    assert at_hour[1][4:] == pytest.approx((152.4, 2.414016))
    assert at_hour[4][:4] == (0, 0, 0, 1)
    assert at_hour[5][:4] == (1, 0, 1, 1)
    assert datetime.fromtimestamp(rows[-1][4], timezone.utc).isoformat() == "2024-02-02T06:00:00+00:00"


@pytest.mark.unit
def test_sigmet_geometry_is_not_an_unrelated_corridor():
    assert clean_sigmets.polygon_bounds("POLYGON ((-92 28,-88 30,-89 32,-92 28))") == (-92, 28, -88, 32)
    with pytest.raises(ValueError):
        clean_sigmets.polygon_bounds("POLYGON ((-92 28,-88 30,-89 32,-99 28))")
    boxes = features.buffered_route((41.97, -87.91), (40.64, -73.78))
    assert min(box[1] for box in boxes) > 35
    assert max(box[2] for box in boxes) > -73.78


def test_enrichment_preserves_cohort_and_filters_issue_expiry_and_corridor(spark):
    flights = spark.createDataFrame([
        ("flight", 41.97, -87.91, 40.64, -73.78, True, 7200, 11000, "KJFK"),
        ("missing", 19.73, -155.04, 21.3, -157.9, False, 7200, None, "PHNL"),
    ], "flight_id string, lat1 double, lon1 double, lat2 double, lon2 double, conus boolean, "
       "cutoff long, arrival long, station string")
    taf = spark.createDataFrame([
        ("KJFK", "old", 100, 9000, 12000, 0, 0, 0, 0, 500.0, 10.0),
        ("KJFK", "known", 7000, 10800, 12000, 1, 1, 0, 0, 150.0, 3.0),
        ("KJFK", "future amendment", 7201, 10800, 12000, 1, 0, 0, 0, 100.0, 2.0),
        ("KJFK", "expires at arrival", 7199, 10800, 11000, 1, 0, 0, 0, 80.0, 2.0),
    ], clean_taf.SCHEMA)
    sigmets = spark.createDataFrame([
        ("hit", 7200, 8000, -82.0, 40.0, -80.0, 44.0, 420.0),
        ("hit", 7200, 8000, -81.0, 41.0, -80.0, 43.0, 420.0),
        ("expired", 200, 7199, -82.0, 40.0, -80.0, 44.0, 510.0),
        ("future", 7201, 8000, -82.0, 40.0, -80.0, 44.0, 530.0),
        ("texas", 5000, 8000, -103.0, 29.0, -93.0, 34.0, 600.0),
    ], clean_sigmets.SCHEMA)
    parent = spark.createDataFrame([("flight", 200.0), ("missing", 300.0)],
                                   "flight_id string, destination_cloud_ceiling_m double")
    actual = features.enrich(parent, flights, taf, sigmets, ["KJFK"]).collect()
    assert len(actual) == 2
    rows = {row.flight_id: row for row in actual}
    assert set(rows) == {"flight", "missing"}
    assert rows["flight"].taf_issue_utc_seconds == 7000
    assert rows["flight"].dest_weather_deterioration_delta_m == -50.0
    assert rows["flight"].enroute_sigmet_convective_count == 1
    assert rows["flight"].enroute_sigmet_max_top_fl == 420.0
    assert rows["flight"].taf_sigmet_missing == 0
    assert rows["missing"].taf_sigmet_missing == 1
    assert rows["missing"].enroute_sigmet_convective_count is None


def test_schedule_uses_destination_local_arrival_and_rolls_service_date(spark):
    def reference(icao, latitude, longitude, zone):
        return [{"icao": icao, "latitude": latitude, "longitude": longitude, "time_zone": zone,
                 "valid_from": "2024-01-01", "valid_to": "2025-01-01"}]

    airports = {"ORD": reference("KORD", 41.97, -87.91, "America/Chicago"),
                "JFK": reference("KJFK", 40.64, -73.78, "America/New_York")}
    parent = spark.createDataFrame([("flight", date(2024, 11, 1), "ORD", "JFK", 120.0)],
                                   "flight_id string, FlightDate date, Origin string, Dest string, CRSElapsedTime double")
    schedule = spark.createDataFrame([("flight", "2300", "0200")], "flight_id string, CRSDepTime string, CRSArrTime string")
    row = features.attach_schedule(parent, schedule, airports).first()
    assert row.station == "KJFK" and row.conus
    assert row.arrival - row.cutoff == 7200
    assert datetime.fromtimestamp(row.arrival, timezone.utc).isoformat() == "2024-11-02T06:00:00+00:00"
