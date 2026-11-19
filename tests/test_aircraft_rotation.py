"""Synthetic coverage for rotation linkage and the paired validation runner."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from scripts.aircraft_rotation import run
from scripts.aircraft_rotation.build_features import (
    RAW_COLUMNS, ROTATION_FLAGS, ROTATION_NUMERIC, build_operations, build_rotation_links,
)
from scripts.bts import train


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
    ]
    parent = targets(spark, [
        (f"2024-09-30_AA_{flight}_ORD_JFK_2230", date(2024, 9, 30), "ORD", 0, "2024-10-01T03:30:00Z", "train")
        for flight in (2, 5, 8, 11, 13, 16, 18)
    ])
    result = {int(row.flight_id.split("_")[2]): row for row in build_rotation_links(
        parent, operations(spark, [*legs, legs[0]]), date(2024, 1, 3)
    ).collect()}
    assert [(result[i].rotation_reason, result[i][ROTATION_NUMERIC[1]]) for i in (2, 5, 8, 11, 13)] == [
        ("usable", 15.0), ("prior_outcome_unavailable", None), ("prior_operation_ineligible", None),
        ("ambiguous_predecessor", None), ("ambiguous_target_or_timeline", None),
    ]
    assert result[2][ROTATION_NUMERIC[0]] == 30.0
    assert result[5][ROTATION_FLAGS[0]] == 0
    assert result[5][ROTATION_FLAGS[1]] == 1
    assert result[8].prior_flight_id == "2024-09-30_AA_7_JFK_ORD_2100"
    assert result[16].rotation_reason == "invalid_prior_outcome"
    assert result[16][ROTATION_NUMERIC[0]] == 30.0
    assert result[16][ROTATION_NUMERIC[1]] is None
    assert result[18].rotation_reason == "usable"
    assert result[18].prior_departure_gap_minutes == 1440.0


def test_paired_rotation_comparison(spark, tmp_path):
    legs, entries = [], []
    for index in range(8):
        day = date(2024, 9 if index < 4 else 10, 28 if index == 3 else 5)
        flight = index * 2 + 2
        delay = 15 * (index % 2)
        legs += [
            leg(day.isoformat(), flight - 1, f"N{index}", "JFK", "ORD", "1100", "1200", diverted=int(index == 4)),
            leg(day.isoformat(), flight, f"N{index}", "ORD", "JFK", "1300", "1600",
                actual="1615" if delay else "1600", delay=delay),
        ]
        entries.append((f"{day}_AA_{flight}_ORD_JFK_1300", day, "ORD", int(bool(delay)),
                        f"{day}T18:00:00Z", "train" if index < 4 else "val"))
    parent = targets(spark, entries)
    numeric, flags = train.weather_columns(run.FEATURE_SET)
    values = {name: 0.0 for name in (*train.INPUT_FEATURES, *numeric)}
    values.update({name: 0 for name in flags})
    values.update(Reporting_Airline="AA", Dest="JFK", Month=9, DayOfWeek=1, DepHour=13, ArrHour=16)
    parent = parent.select("*", *[F.lit(value).alias(name) for name, value in values.items() if name not in parent.columns])
    prep = run.prepare_experiment(
        parent, operations(spark, legs), tmp_path, date(2024, 1, 3),
        datetime(2024, 9, 28, 20, tzinfo=timezone.utc),
    )
    assert prep["training_exclusions"] == {"unresolved": 0, "after_bound": 1}
    with patch.object(train, "LOGISTIC_PARAMS", {**train.LOGISTIC_PARAMS, "maxIter": 2}):
        summary = run.run_comparison(spark, tmp_path)
    w2, h1 = summary["arms"]["w2"], summary["arms"]["h1"]
    assert w2["training_rows"] == h1["training_rows"] == 3
    assert w2["validation_rows"] == h1["validation_rows"] == 4
    assert h1["vector_dimension"] == w2["vector_dimension"] + 4
    assert h1["input_columns"] == w2["input_columns"] + [*ROTATION_NUMERIC, *ROTATION_FLAGS]
    assert summary["completed_predecessor_validation_rows"] == 3
    assert set(summary["h1_minus_w2"]) == set(run.METRICS)
    model = PipelineModel.load(str(tmp_path / "h1/baseline_pipeline"))
    gold = spark.read.parquet(str(tmp_path / "gold/val"))
    missing = model.transform(gold.where(F.col(ROTATION_FLAGS[1]) == 1)).first().features
    assert missing.size == h1["vector_dimension"]
    assert list(missing)[-4:] == [60.0, 0.0, 0.0, 1.0]


def test_publication_requires_unchanged_inputs(tmp_path):
    manifest = {"inputs": {"parent": "original"}}
    with pytest.raises(ValueError, match="inputs changed"):
        run.publish_run(tmp_path, manifest, lambda: {"parent": "changed"})
    with patch.object(run, "write_json", side_effect=OSError("locked")):
        with pytest.raises(OSError, match="locked"):
            run.publish_run(tmp_path, manifest, lambda: manifest["inputs"])
    assert not (tmp_path / "_SUCCESS").exists()
    run.publish_run(tmp_path, manifest, lambda: manifest["inputs"])
    assert (tmp_path / "run.json").is_file()
    assert (tmp_path / "_SUCCESS").is_file()


def test_failed_second_fit_propagates(tmp_path):
    with patch.object(train, "train", side_effect=[{}, RuntimeError("second fit failed")]) as fit:
        with pytest.raises(RuntimeError, match="second fit failed"):
            run.run_comparison(None, tmp_path)
    assert [call.kwargs["feature_set"] for call in fit.call_args_list] == [run.FEATURE_SET, train.ROTATION_FEATURE_SET]
    assert all(not call.kwargs["publish_results"] for call in fit.call_args_list)
    assert not (tmp_path / "_SUCCESS").exists()


def test_prediction_cohort_rejects_duplicate_and_invalid_scores(spark, tmp_path):
    expected = spark.createDataFrame(
        [("a", date(2024, 10, 1), 0), ("b", date(2024, 10, 1), 1)], run.IDENTITY
    )
    valid = expected.withColumn("p_delayed", F.lit(0.5))
    duplicate = valid.where("flight_id = 'a'")
    for bad in (duplicate.unionByName(duplicate),
                valid.withColumn("p_delayed", F.lit(float("nan")))):
        reader = MagicMock()
        reader.read.parquet.return_value = bad
        with pytest.raises(ValueError, match="predictions have invalid probabilities"):
            run.validation_predictions(reader, tmp_path, "w2", expected)


def test_missing_parent_fails_before_spark():
    with (patch.object(run, "load_config", return_value=({}, {}, {})),
          patch.object(run, "features_current", return_value=True),
          patch.object(run, "find_gold", side_effect=["origin", None]),
          patch.object(run, "get_spark_session") as start_spark):
        with pytest.raises(ValueError, match="completed origin and destination weather Gold"):
            run.main([])
    start_spark.assert_not_called()


def test_entry_point_reads_only_development_splits(tmp_path):
    spark = MagicMock()
    parent = tmp_path / "parent"
    raw_file = tmp_path / "november.csv"
    config = {"datasets": {"bts": {"start_month": "2024-01"}},
              "splits": {"val": {"start_date": "2024-10-01"}}}
    with (patch.object(run, "ROOT", tmp_path),
          patch.object(run, "resolve_inputs", return_value=(config, tmp_path, parent, [raw_file])),
          patch.object(run, "source_identity", return_value={}),
          patch.object(run, "get_spark_session", return_value=spark),
          patch.object(run, "load_airports"),
          patch.object(run, "build_operations"),
          patch.object(run, "prepare_experiment", return_value={}),
          patch.object(run, "run_comparison", return_value={"timing_seconds": {}})):
        assert run.main([]) == 0
    spark.read.parquet.assert_called_once_with(str(parent / "train"), str(parent / "val"))
    spark.stop.assert_called_once_with()
    completed, = (tmp_path / "models/aircraft_rotation_comparison").glob("*/_SUCCESS")
    assert (completed.parent / "run.json").is_file()
