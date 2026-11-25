"""Spark session configuration and lifecycle management."""

from collections.abc import Generator
from contextlib import contextmanager
import os
import sys

from pyspark.sql import SparkSession

from .config import ROOT


def configure_spark_env() -> None:
    """Configure Windows Hadoop binaries and pin PySpark workers to this Python."""
    if os.name == "nt" and "HADOOP_HOME" not in os.environ:
        hadoop_dir = ROOT / "envs" / "hadoop"
        if (hadoop_dir / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(hadoop_dir)
            os.environ["PATH"] = (
                str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")
            )
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


configure_spark_env()


def get_spark_session(app_name: str = "flight_delay") -> SparkSession:
    """Create or retrieve a tuned local SparkSession bounded for laptop execution."""
    configure_spark_env()
    python_path = os.environ.get("PYTHONPATH", "")
    if str(ROOT) not in python_path.split(os.pathsep):
        # Ensure Spark workers can import the project scripts package.
        os.environ["PYTHONPATH"] = str(ROOT) + (os.pathsep + python_path if python_path else "")
    return (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.files.maxPartitionBytes", "32m")
        .config("spark.sql.parquet.columnarReaderBatchSize", "1024")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "16m")
        .config("spark.sql.adaptive.coalescePartitions.parallelismFirst", "false")
        .getOrCreate()
    )


@contextmanager
def spark_session(app_name: str = "flight_delay") -> Generator[SparkSession, None, None]:
    """Yield a configured SparkSession and stop it cleanly when leaving the block."""
    spark = get_spark_session(app_name)
    try:
        yield spark
    finally:
        spark.stop()
