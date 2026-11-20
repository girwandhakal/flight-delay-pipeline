"""Synthetic destination contracts; real test-period labels are never read."""

from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from tests.conftest import get_test_spark_session
from pipeline import main as pipeline
from scripts.bts import train
from scripts.destination_weather import build_features as destination
from scripts.origin_weather import build_features as origin, download
from scripts.origin_weather.measurements import FEATURE_COLUMNS, NUMERIC_FEATURES, SOURCE_FIELDS, decode_weather
from scripts.results import update_results


class DestinationTests(unittest.TestCase):
    def test_inference_hashes_include_tree_challenger_models_and_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "bts/gbt/baseline_pipeline/model"
            model.parent.mkdir(parents=True)
            model.write_text("tree")
            threshold = root / "bts/gbt/threshold.json"
            threshold.write_text('{"threshold": 0.3}')
            original = pipeline.inference_hashes(root)
            self.assertEqual(set(original), {str(model.relative_to(root)), str(threshold.relative_to(root))})
            model.write_text("changed tree")
            self.assertNotEqual(original, pipeline.inference_hashes(root))
            threshold.write_text('{"threshold": 0.4}')
            self.assertNotEqual(original[str(threshold.relative_to(root))],
                                pipeline.inference_hashes(root)[str(threshold.relative_to(root))])

    def test_parent_reference_accepts_zip_line_endings_and_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "data/gold/bts_origin_weather/fixture"
            for split in ("train", "val", "test"):
                (parent / split).mkdir(parents=True)
                (parent / split / "_SUCCESS").touch()
            reference = root / "references"
            reference.mkdir()
            target = reference / "airports.csv"
            target.write_bytes(b"Origin\r\nORD\r\n")
            sources = {"files": {"airports.csv": download.sha256(target)}}
            download.write_json(reference / "sources.json", sources)
            metadata = {
                "policy": {}, "splits": dict.fromkeys(("train", "val", "test")),
                "weather_columns": origin.WEATHER_NUMERIC + origin.WEATHER_FLAGS,
                "bts_input_files": [], "reference_sources": sources,
                "source_contract": {"timestamp_basis": "metar_verified_utc_or_local_standard"},
            }
            download.write_json(parent / "_metadata.json", metadata)
            target.write_bytes(b"Origin\nORD\n")
            with patch.object(destination, "ROOT", root), patch.object(destination, "bts_fingerprints", return_value=[]):
                self.assertEqual(destination.validate_parent(parent, {"reference_directory": reference}, {}, metadata), metadata)
                target.write_bytes(b"Origin\nLAX\n")
                with self.assertRaisesRegex(ValueError, "Origin reference changed: airports.csv"):
                    destination.validate_parent(parent, {"reference_directory": reference}, {}, metadata)

    def test_default_driver_runs_destination_workflow(self):
        with patch.object(pipeline, "run_destination_weather", return_value=0) as run, patch.object(pipeline, "run_stage") as legacy:
            self.assertEqual(pipeline.main([]), 0)
            run.assert_called_once_with(
                include_gbt=False, include_rotation=True, include_enroute=True,
            )
            legacy.assert_not_called()
        with (patch.object(pipeline, "run_destination_weather") as run,
              patch.object(pipeline, "features_current", return_value=False),
              patch.object(pipeline, "run_stage", return_value=0) as legacy):
            self.assertEqual(pipeline.main(["--bts-only"]), 0)
            run.assert_not_called()
            self.assertEqual(legacy.call_count, 5)

    def test_destination_bootstraps_missing_bts_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (patch.object(pipeline, "ROOT", root),
                  patch.object(pipeline.bts_download, "main", return_value=0) as download_bts,
                  patch.object(pipeline.bts_clean, "main", return_value=0) as clean_bts,
                  patch.object(pipeline.bts_transform, "main", return_value=0) as build_bts,
                  patch.object(pipeline.weather_features, "main", return_value=(root / "parent", root / "parent")) as origin_build,
                  patch.object(destination, "main", side_effect=RuntimeError("after bootstrap"))):
                with self.assertRaisesRegex(RuntimeError, "after bootstrap"):
                    pipeline.run_destination_weather()
                download_bts.assert_called_once_with()
                clean_bts.assert_called_once_with(publish_results=False)
                build_bts.assert_called_once_with(publish_results=False)
                origin_build.assert_called_once_with(publish_results=False)
            self.assertFalse((root / "results/results.json").exists())

    def test_destination_bootstrap_stops_on_each_failed_stage(self):
        for index in range(3):
            with self.subTest(stage=index), tempfile.TemporaryDirectory() as tmp:
                with (patch.object(pipeline, "ROOT", Path(tmp)),
                      patch.object(pipeline.bts_download, "main", return_value=7 if index == 0 else 0) as download_bts,
                      patch.object(pipeline.bts_clean, "main", return_value=7 if index == 1 else 0) as clean_bts,
                      patch.object(pipeline.bts_transform, "main", return_value=7 if index == 2 else 0) as build_bts,
                      patch.object(pipeline.weather_features, "main") as origin_build,
                      patch.object(destination, "main") as destination_build):
                    self.assertEqual(pipeline.run_destination_weather(), 7)
                    self.assertEqual([download_bts.call_count, clean_bts.call_count, build_bts.call_count], [int(i <= index) for i in range(3)])
                    origin_build.assert_not_called()
                    destination_build.assert_not_called()

    def test_modes_and_feature_contract(self):
        with patch.object(pipeline, "run_destination_weather", return_value=0) as run, patch.object(pipeline, "evaluate_destination_weather", return_value=0) as evaluate:
            self.assertEqual(pipeline.main(["--destination-weather", "--include-gbt"]), 0)
            run.assert_called_once_with(include_gbt=True, include_rotation=False)
            evaluate.assert_not_called()
            self.assertEqual(pipeline.main(["--evaluate-destination-weather", "frozen"]), 0)
            evaluate.assert_called_once_with("frozen")
        with patch.object(pipeline, "run_origin_weather", return_value=0) as run:
            self.assertEqual(pipeline.main(["--origin-weather", "--include-gbt"]), 0)
            run.assert_called_once_with(include_gbt=True)
        with (patch.object(pipeline, "features_current", return_value=True),
              patch.object(pipeline, "run_stage", return_value=0) as stage):
            self.assertEqual(pipeline.main(["--bts-only", "--include-gbt"]), 0)
            training = stage.call_args_list[0].args[2]
            self.assertIs(training.func, train.main)
            self.assertEqual(training.keywords, {"include_gbt": True})
        self.assertEqual(len(destination.DESTINATION_WEATHER_NUMERIC), 27)
        self.assertEqual(len(destination.DESTINATION_WEATHER_FLAGS), 18)
        self.assertEqual(destination.DESTINATION_WEATHER_NUMERIC, [name.replace("origin_", "destination_", 1) for name in origin.WEATHER_NUMERIC])
        with self.assertRaises(ValueError):
            origin.validate_prefix("arrival")
        with self.assertRaisesRegex(ValueError, "isolated model paths"):
            train.train(None, feature_set="bts_origin_destination_weather")

    def test_failed_third_fit_does_not_publish_or_mark_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                path = root / "data/gold/bts" / split
                path.mkdir(parents=True)
                (path / "_SUCCESS").touch()
            parent = root / "parent"
            current = root / "results/results.json"
            current.parent.mkdir()
            current.write_text(json.dumps({"bts_origin_weather": {"data": {"gold_dir": str(parent)}}}))
            before = current.read_bytes()
            with (patch.object(pipeline, "ROOT", root),
                  patch.object(pipeline, "features_current", return_value=True),
                  patch.object(pipeline.weather_features, "main", return_value=(parent, parent)),
                  patch.object(destination, "main", return_value=(root / "combined", root / "combined")),
                  patch.object(pipeline.bts_transform, "get_spark_session") as session,
                  patch.object(train, "train", side_effect=[{}, {}, RuntimeError("third fit")]) as fit,
                  patch.object(pipeline, "update_results") as publish):
                with self.assertRaisesRegex(RuntimeError, "third fit"):
                    pipeline.run_destination_weather()
                self.assertEqual(fit.call_count, 3)
                self.assertTrue(all(call.kwargs["publish_results"] is False for call in fit.call_args_list))
                publish.assert_not_called()
                self.assertTrue(session.return_value.stop.called)
            self.assertEqual(current.read_bytes(), before)
            self.assertFalse(list((root / "models").rglob("_SUCCESS")))

    def test_default_pipeline_trains_and_reports_four_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin_gold, combined_gold = root / "origin", root / "combined"
            for directory in (origin_gold, combined_gold):
                directory.mkdir()
                (directory / "_metadata.json").write_text(
                    json.dumps(dict.fromkeys(("quality", "station_quality", "source_contract"), {}))
                )
            variants = train.ALL_FEATURE_SETS
            metrics = {key: 0.5 for key in pipeline.ABLATION_METRICS}
            summary = {
                "reference_baselines": {"train_prevalence": {"training_prevalence": 0.5}},
                "selected_model": {"best_threshold": 0.5}, "training_rows": 2, "input_columns": [],
            }
            updates = {variant: {"validation": {"selected": {**metrics, "threshold": 0.5, "rows": 2}}}
                       for variant in variants}
            with (patch.object(pipeline, "ROOT", root),
                  patch.object(pipeline, "load_config", return_value=({"reference_directory": root}, {}, {"splits": {}})),
                  patch.object(pipeline, "features_current", return_value=True),
                  patch.object(pipeline.weather_features, "main", return_value=(origin_gold, origin_gold)),
                  patch.object(destination, "main", return_value=(combined_gold, combined_gold)),
                  patch.object(pipeline.bts_transform, "get_spark_session") as session,
                  patch.object(pipeline.rotation, "rotation_raw_files", return_value=[]),
                  patch.object(pipeline.rotation, "source_identity", return_value={}),
                  patch.object(pipeline.rotation, "prepare_rotation", return_value={"coverage": []}) as prepare,
                  patch.object(train, "train", side_effect=lambda *args, **kwargs: dict(summary)) as fit,
                  patch.object(pipeline, "compare_predictions", return_value=updates) as compare,
                  patch.object(pipeline, "update_results") as publish,
                  patch.object(pipeline.enroute, "prepare", return_value="prepared forecasts") as prepare_enroute,
                  patch.object(pipeline.enroute, "fit") as fit_enroute):
                session.return_value.version = "3.5.7"
                self.assertEqual(pipeline.main([]), 0)
                prepared_dir = prepare.call_args.args[-1]
                self.assertEqual(prepare.call_args.args[-3], combined_gold)
                self.assertEqual([call.kwargs["feature_set"] for call in fit.call_args_list], list(variants))
                self.assertEqual({call.args[1] for call in fit.call_args_list}, {prepared_dir / "gold"})
                self.assertEqual([call.args[2].name for call in fit.call_args_list],
                                 ["bts", "origin", "both", "rot"])
                self.assertTrue(all(call.kwargs["model_name"] == "model" for call in fit.call_args_list))
                self.assertTrue(all(not call.kwargs["include_gbt"] for call in fit.call_args_list))
                self.assertEqual(compare.call_args.kwargs["variants"], variants)
                self.assertEqual(compare.call_args.kwargs["variant_dirs"], pipeline.DESTINATION_MODEL_DIRS)
                self.assertEqual(set(publish.call_args.args[0]), set(variants))
                prepare_enroute.assert_called_once_with(
                    session.return_value, root, {"splits": {}}, combined_gold, prepared_dir.name,
                )
                fit_enroute.assert_called_once_with(
                    session.return_value, root, prepared_dir, combined_gold, "prepared forecasts", summary,
                )
                session.return_value.stop.assert_called_once_with()
                fit.side_effect = [summary, summary, summary, RuntimeError("fourth fit failed")]
                compare.reset_mock()
                publish.reset_mock()
                fit_enroute.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "fourth fit failed"):
                    pipeline.main([])
                compare.assert_not_called()
                publish.assert_not_called()
                fit_enroute.assert_not_called()
            report = json.loads((prepared_dir / "validation_comparison.json").read_text())
            self.assertEqual([row["feature_set"] for row in report["models"]], list(variants))
            self.assertEqual(
                list((root / "models/dw").glob("*/_SUCCESS")),
                [prepared_dir / "_SUCCESS"],
            )

    def test_frozen_evaluation_rejects_stale_or_changed_artifacts_before_spark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models/destination_weather_comparison/frozen"
            models.mkdir(parents=True)
            (models / "_SUCCESS").touch()
            latest = root / "results/results.json"
            latest.parent.mkdir()
            latest.write_text(json.dumps({variant: {"run_id": "newer"} for variant in train.FEATURE_SETS}))
            with patch.object(pipeline, "ROOT", root), patch.object(pipeline.bts_transform, "get_spark_session") as session:
                with self.assertRaisesRegex(ValueError, "current comparison"):
                    pipeline.evaluate_destination_weather("frozen")
                latest.write_text(json.dumps({variant: {"run_id": "frozen"} for variant in train.FEATURE_SETS}))
                (models / "run.json").write_text(json.dumps({"run_id": "frozen", "models_dir": str(models), "inference_sha256": {"changed": "hash"}}))
                with self.assertRaisesRegex(ValueError, "artifacts changed"):
                    pipeline.evaluate_destination_weather("frozen")
                with self.assertRaises(ValueError):
                    pipeline.evaluate_destination_weather("../escape")
                session.assert_not_called()

    def test_frozen_evaluation_only_scores_saved_test_inputs(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self.check_frozen_evaluation(legacy)

    def check_frozen_evaluation(self, legacy):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_name = "baseline_pipeline" if legacy else "model"
            models = root / ("models/destination_weather_comparison" if legacy else "models/dw") / "frozen"
            variant_dirs = None if legacy else pipeline.DESTINATION_MODEL_DIRS
            expected_models = []
            models.mkdir(parents=True)
            (models / "_SUCCESS").touch()
            dirs = {variant: root / "data/gold" / variant for variant in train.FEATURE_SETS}
            for variant, path in dirs.items():
                path.mkdir(parents=True)
                (path / "_metadata.json").write_text("{}")
                model = models / (variant if legacy else variant_dirs[variant]) / model_name
                expected_models.append(model)
                model.mkdir(parents=True)
                (model / "part").write_text("frozen")
                (model.parent / "threshold.json").write_text('{"threshold": 0.6}')
            hashes = pipeline.inference_hashes(models, variant_dirs, model_name)
            self.assertEqual(len(hashes), 6)
            run = {"run_id": "frozen", "models_dir": str(models),
                   "data_dirs": {v: str(p) for v, p in dirs.items()},
                   "inference_sha256": hashes,
                   "data_fingerprints": {v: [] for v in dirs},
                   "data_metadata_sha256": {v: download.sha256(p / "_metadata.json") for v, p in dirs.items() if v != "bts"},
                   "thresholds": {v: .6 for v in dirs}, "training_prevalence": {v: .25 for v in dirs}}
            (models / "run.json").write_text(json.dumps(run))
            latest = root / "results/results.json"
            latest.parent.mkdir()
            latest.write_text(json.dumps({v: {"run_id": "frozen"} for v in dirs}))
            scores = MagicMock()
            scores.select.return_value.withColumn.return_value = scores
            scores.write.parquet.side_effect = lambda path: Path(path).mkdir(parents=True)
            with (patch.object(pipeline, "ROOT", root),
                  patch.object(pipeline.bts_transform, "get_spark_session") as session,
                  patch.object(pipeline.PipelineModel, "load") as load,
                  patch.object(pipeline.F, "lit", return_value=.6),
                  patch.object(train, "predict", return_value=scores),
                  patch.object(train, "validate_weather"),
                  patch.object(train, "train") as fit,
                  patch.object(download, "download_weather") as acquire,
                  patch.object(destination, "build_destination_features") as rematch,
                  patch.object(pipeline, "compare_predictions", return_value={v: {"test": {}} for v in dirs}),
                  patch.object(pipeline, "update_results") as publish):
                self.assertEqual(pipeline.evaluate_destination_weather("frozen"), 0)
                self.assertEqual([Path(call.args[0]) for call in load.call_args_list], expected_models)
                self.assertEqual(
                    [Path(call.args[0]).parent.name for call in scores.write.parquet.call_args_list],
                    [model.parent.name for model in expected_models],
                )
                fit.assert_not_called()
                acquire.assert_not_called()
                rematch.assert_not_called()
                publish.assert_called_once()
                self.assertTrue(all(Path(call.args[0]).name == "test" for call in session.return_value.read.parquet.call_args_list))
                self.assertEqual(len(list((models / "test").rglob("_SUCCESS"))), 1)

class DestinationSparkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = get_test_spark_session("destination_contract")

    def references(self):
        airports = [{"Origin": "LAX", "time_zone": "America/Los_Angeles", "valid_from": "2023-01-01", "valid_to": "2026-01-01"}]
        mappings = [{"Origin": "LAX", "station_id": "USW00023174", "method": "exact", "valid_from": "2023-01-01", "valid_to": "2026-01-01"}]
        return airports, mappings

    def observations(self, instants):
        tokens = {name: "10" for name in SOURCE_FIELDS}
        tokens.update(HourlyPresentWeatherType="AU:RA SN", HourlySkyConditions="BKN:02 10", HourlyPrecipitation="T")
        decoded = decode_weather(tokens)
        schema = "station_id string, observation_utc timestamp, source_file string, REM string, " + ", ".join(f"{name} {'double' if name in NUMERIC_FEATURES else 'int'}" for name in FEATURE_COLUMNS)
        return self.spark.createDataFrame([("USW00023174", instant, source, "raw report", *[decoded[name] for name in FEATURE_COLUMNS]) for instant, source in instants], schema)

    def test_effective_local_dates_and_unresolved_mappings(self):
        flights = self.spark.createDataFrame([
            ("west", "LAX", date(2024, 1, 1), datetime(2024, 1, 1, 1, tzinfo=timezone.utc)),
            ("transition", "LAX", date(2024, 1, 1), datetime(2024, 1, 1, 8, 30, tzinfo=timezone.utc)),
            ("east", "TKO", date(2023, 12, 31), datetime(2023, 12, 31, 20, tzinfo=timezone.utc)),
            ("invalid", "LAX", date(2024, 1, 1), None),
            ("missing", "XXX", date(2024, 1, 1), datetime(2024, 1, 1, 1)),
            ("badzone", "BAD", date(2024, 1, 1), datetime(2024, 1, 1, 1)),
        ], "flight_id string, Dest string, FlightDate date, scheduled_departure_utc timestamp")
        airports, mappings = self.references()
        mappings[0]["valid_to"] = "2024-01-01"
        for code, zone in (("TKO", "Asia/Tokyo"), ("BAD", "Invalid/Zone")):
            airports.append({**airports[0], "Origin": code, "time_zone": zone})
            mappings.append({**mappings[0], "Origin": code, "valid_from": "2024-01-01", "valid_to": "2026-01-01"})
        rows = {row.flight_id: row for row in destination.build_destination_cutoffs(self.spark, flights, (airports, mappings)).collect()}
        self.assertEqual(rows["west"].destination_station_id, "USW00023174")
        self.assertEqual(rows["east"].destination_time_zone, "Asia/Tokyo")
        for key in ("invalid", "missing", "badzone", "transition"):
            self.assertIsNone(rows[key].destination_station_id)
        self.assertEqual(rows["west"].scheduled_departure_utc, flights.filter("flight_id = 'west'").first().scheduled_departure_utc)
        ambiguous = [*airports, {**airports[0], "time_zone": "Asia/Tokyo"}]
        row = destination.build_destination_cutoffs(self.spark, flights.filter("flight_id = 'west'"), (ambiguous, mappings)).first()
        self.assertIsNone(row.destination_station_id)

    def test_bounds_missing_flags_and_parent_equality(self):
        instant = datetime(2024, 1, 1, 18)
        ages = (0, 14, 15, 120, 121, -1)
        flight_rows = [
            (f"ok_{age}", "LAX", instant + timedelta(minutes=age), "keep origin", 123.0)
            for age in ages
        ] + [
            ("unmapped", "XXX", instant, "keep missing", 456.0),
            ("null_destination", None, instant, "keep null destination", 456.0),
            ("invalid", "LAX", None, "keep invalid", 789.0),
        ]
        flights = self.spark.createDataFrame(
            flight_rows,
            "flight_id string, Dest string, scheduled_departure_utc timestamp, source_file string, origin_temperature_c double"
        )
        cutoffs = destination.build_destination_cutoffs(self.spark, flights, self.references())
        observations = self.observations([(instant, "selected")])
        for lag in (0, 15):
            enriched = destination.build_destination_features(cutoffs, observations, {"files": {}}, publication_lag_minutes=lag).cache()
            try:
                plan = enriched._jdf.queryExecution().executedPlan().toString()
                self.assertIn("SortMergeJoin", plan)
                self.assertNotIn("BroadcastHashJoin", plan)
                self.assertNotIn("BroadcastNestedLoopJoin", plan)
                rows = {row.flight_id: row for row in enriched.collect()}
                for age in ages:
                    row = rows[f"ok_{age}"]
                    matched = lag <= age <= 120
                    self.assertEqual(row.destination_weather_missing, int(not matched))
                    self.assertEqual(row.source_file, "keep origin")
                    self.assertEqual(row.origin_temperature_c, 123)
                    if matched:
                        self.assertEqual(row.destination_weather_age_minutes, age)
                        self.assertLessEqual(row.destination_available_utc, instant + timedelta(minutes=age))
                    else:
                        self.assertTrue(all(row[name] is None for name in destination.DESTINATION_WEATHER_NUMERIC))
                        self.assertTrue(all(row[name] == 1 for name in destination.DESTINATION_WEATHER_FLAGS[:13]))
                        self.assertTrue(all(row[name] == 0 for name in destination.DESTINATION_WEATHER_FLAGS[13:]))
                self.assertEqual(rows["invalid"].destination_weather_match_status, "invalid_cutoff")
                self.assertEqual(rows["unmapped"].destination_weather_match_status, "unmapped_airport")
                self.assertEqual(rows["null_destination"].destination_weather_match_status, "unmapped_airport")
                if lag == 15:
                    destination.validate_destination_cohort(flights, enriched, publication_lag_minutes=15)
                    for changed in (
                        enriched.filter("flight_id != 'invalid'"),
                        enriched.withColumn("flight_id", F.concat(F.lit("other_"), F.col("flight_id"))),
                        enriched.withColumn("source_file", F.lit(None).cast("string")),
                    ):
                        with self.subTest(change=changed.columns), self.assertRaisesRegex(ValueError, "parent columns"):
                            destination.validate_destination_cohort(flights, changed, publication_lag_minutes=15)
                    with self.assertRaisesRegex(ValueError, "nonnull and unique"):
                        destination.validate_destination_cohort(flights, enriched.unionByName(enriched))
                    with self.assertRaisesRegex(ValueError, "parent columns"):
                        destination.validate_destination_cohort(flights, enriched.withColumn("origin_temperature_c", F.lit(999.0)))
                    with self.assertRaisesRegex(ValueError, "observation"):
                        destination.validate_destination_cohort(flights, enriched.withColumn(
                            "destination_observation_utc", F.col("scheduled_departure_utc") + F.expr("INTERVAL 1 MINUTE")
                        ), publication_lag_minutes=15)
            finally:
                enriched.unpersist()

    def test_combined_vector_prefix_means_and_reload(self):
        columns = train.INPUT_FEATURES
        fitting = self.spark.createDataFrame([
            ("AA", "ORD", "LAX", 1, 1, 12, 15, 120.0, 800.0,
             0.25, 8.0, 4.0, 0.5, 10.0, 6.0, 18.0, 5.0, 3, 0, 0, 0),
        ], columns)
        for name in origin.WEATHER_NUMERIC:
            fitting = fitting.withColumn(name, F.lit(10.0))
        for name in destination.DESTINATION_WEATHER_NUMERIC:
            fitting = fitting.withColumn(name, F.lit(20.0))
        for name in origin.WEATHER_FLAGS + destination.DESTINATION_WEATHER_FLAGS:
            fitting = fitting.withColumn(name, F.lit(0))
        train.validate_weather(fitting, "fixture fold", "bts_origin_destination_weather")
        origin_model = train.build_preprocessor("bts_origin_weather").fit(fitting)
        model = train.build_preprocessor("bts_origin_destination_weather").fit(fitting)
        validation = fitting.withColumn("destination_temperature_c", F.lit(None).cast("double")).withColumn("Reporting_Airline", F.lit("UNSEEN"))
        validation = validation.withColumn("origin_temperature_c", F.lit(None).cast("double")).withColumn("Month", F.lit(12))
        for name in train.ROLLING_FEATURES:
            validation = validation.withColumn(name, F.lit(None).cast("double"))
        row = model.transform(validation).first()
        vector = row.features.toArray().tolist()
        self.assertEqual(row.Reporting_Airline_idx, len(model.stages[0].labelsArray[0]))
        self.assertNotIn("Month", model.stages[0].getInputCols())
        self.assertAlmostEqual(row.MonthSin, -0.5)
        self.assertAlmostEqual(row.MonthCos, math.sqrt(3) / 2)
        self.assertTrue(all(row[name + "_filled"] == 0 for name in train.ROLLING_FEATURES))
        prefix = origin_model.transform(validation).first().features.toArray().tolist()
        self.assertEqual(vector[:-45], prefix)
        self.assertEqual(vector[-45:-18], [20.0] * 27)
        self.assertEqual(model.stages[2].surrogateDF.first().destination_temperature_c, 20)
        self.assertEqual(row.origin_temperature_c_imputed, 10)
        self.assertEqual(row.destination_temperature_c_imputed, 20)
        with self.assertRaisesRegex(ValueError, "destination_temperature_c.*fixture fold"):
            train.validate_weather(fitting.withColumn("destination_temperature_c", F.lit(None).cast("double")), "fixture fold", "bts_origin_destination_weather")
        with self.assertRaisesRegex(ValueError, "origin_temperature_c.*fixture fold"):
            train.validate_weather(validation, "fixture fold", "bts_origin_destination_weather")
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "pipeline")
            model.write().save(path)
            self.assertEqual(PipelineModel.load(path).transform(validation).first().features, model.transform(validation).first().features)

    def test_four_way_metrics_and_cohort_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variants = train.ALL_FEATURE_SETS
            variant_dirs = pipeline.DESTINATION_MODEL_DIRS
            for variant, probabilities in zip(variants, ([.9, .8, .2, .1], [.9, .1, .9, .1], [.9, .1, .9, .1], [.8, .2, .1, .1])):
                scores = self.spark.createDataFrame([(str(i), date(2024, 10, 1), int(i % 2 == 0), p, int(p >= .5), .5) for i, p in enumerate(probabilities)], "flight_id string, FlightDate date, ArrDel15 int, p_delayed double, prediction int, threshold double")
                scores.write.parquet(str(root / variant_dirs[variant] / "validation_predictions"))
            context = self.spark.createDataFrame([(str(i), "ORD", "LAX", 10, 0, int(i > 1), 2.0, None if i > 1 else 7.0) for i in range(4)],
                "flight_id string, Origin string, Dest string, Month int, origin_weather_missing int, destination_weather_missing int, origin_wind_speed_ms double, destination_wind_speed_ms double")
            context = context.withColumn("rotation_outcome_missing", F.col("destination_weather_missing"))
            result = pipeline.compare_predictions(self.spark, root, context=context, variants=variants,
                                                  training_prevalence={v: .25 for v in variants}, variant_dirs=variant_dirs)
            combined = result["bts_origin_destination_weather"]["validation"]
            self.assertEqual(combined["deltas_vs_bts"]["f1"], .5)
            self.assertEqual(combined["deltas_vs_bts_origin_weather"]["f1"], 0)
            self.assertEqual(combined["selected"]["tp"], 2)
            self.assertEqual(combined["reference_baselines"]["train_prevalence"]["training_prevalence"], .25)
            self.assertEqual({row["subgroup_type"] for row in combined["subgroups"]}, {"Origin", "Dest", "Month", "origin_weather_missing", "destination_weather_missing", "origin_wind_bin", "destination_wind_bin", "rotation_outcome_missing"})
            self.assertTrue(all(row["total_flights"] > 0 for row in combined["subgroups"]))
            paired = self.spark.read.parquet(str(root / "validation_predictions"))
            self.assertEqual(paired.count(), 4)
            self.assertIn("rotation_p_delayed", paired.columns)
            rotation = result[train.ROTATION_FEATURE_SET]["validation"]
            self.assertLess(rotation["deltas_vs_bts_origin_destination_weather"]["f1"], 0)
            scores.withColumn("ArrDel15", F.lit(0)).write.mode("overwrite").parquet(str(root / "rot/validation_predictions"))
            with self.assertRaisesRegex(ValueError, "cohorts differ"):
                pipeline.compare_predictions(self.spark, root, variants=variants, variant_dirs=variant_dirs)
