"""Shared pytest configuration and session-scoped Spark management."""

import pytest
from pyspark.sql import SparkSession

_SHARED_SPARK = None


def get_test_spark_session(app_name="pipeline_unit_tests") -> SparkSession:
    """Return a single persistent local SparkSession for all tests."""
    from scripts.common.spark import configure_spark_env

    global _SHARED_SPARK
    if (
        _SHARED_SPARK is None
        or _SHARED_SPARK._jsc is None
        or _SHARED_SPARK._jsc.sc().isStopped()
    ):
        configure_spark_env()
        _SHARED_SPARK = (
            SparkSession.builder.master("local[2]")
            .appName(app_name)
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.default.parallelism", "2")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        _SHARED_SPARK.sparkContext.setLogLevel("ERROR")
    return _SHARED_SPARK


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Session-scoped Spark fixture for pytest test functions."""
    return get_test_spark_session()


def pytest_sessionfinish(session, exitstatus):
    """Clean up the shared Spark session at the very end of the test run."""
    global _SHARED_SPARK
    if _SHARED_SPARK is not None:
        try:
            if not _SHARED_SPARK._jsc.sc().isStopped():
                _SHARED_SPARK.stop()
        except Exception:
            pass
        _SHARED_SPARK = None


def pytest_collection_modifyitems(items):
    """Automatically mark tests without an explicit 'unit' marker as 'integration'."""
    for item in items:
        if "unit" not in item.keywords:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def full_feature_rows(spark):
    from pyspark.sql import functions as F

    from scripts.modeling import train

    def rows(split, day):
        frame = spark.range(9).select(
            F.concat(F.lit(split + "-"), F.col("id")).alias("flight_id"), F.lit(day).alias("FlightDate"),
            F.when(F.col("id") % 2 == 0, "AA").otherwise("DL").alias("Reporting_Airline"),
            F.lit("ORD").alias("Origin"), F.lit("JFK").alias("Dest"),
            F.lit(day.month).alias("Month"), F.lit(5).alias("DayOfWeek"),
            F.lit(10).alias("DepHour"), F.lit(13).alias("ArrHour"),
            F.col("id").cast("double").alias("CRSElapsedTime"), F.lit(600.0).alias("Distance"),
            (F.col("id") % 3 == 0).cast("int").alias(train.TARGET_COL),
        )
        values = sorted(set(train.INPUT_COLUMNS) - set(frame.columns))
        flags = {*train.SURFACE_FLAGS, *train.ROTATION_FLAGS, *train.ENROUTE_FLAGS,
                 *train.HOLIDAY_FEATURES, train.SLACK_FEATURES[1], train.RECOVERY_FEATURES[1]}
        return frame.select("*", *(F.lit(0 if name in flags else 1.0).alias(name) for name in values))

    return rows
