"""Shared pytest configuration and session-scoped Spark management."""

import pytest
from pyspark.sql import SparkSession

_SHARED_SPARK = None


def get_test_spark_session(app_name="pipeline_unit_tests") -> SparkSession:
    """Return a single persistent local SparkSession for all tests."""
    global _SHARED_SPARK
    if (
        _SHARED_SPARK is None
        or _SHARED_SPARK._jsc is None
        or _SHARED_SPARK._jsc.sc().isStopped()
    ):
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
