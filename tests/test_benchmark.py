"""Tests for the systems performance and scalability benchmark suite (Issue #4)."""

import json
from pathlib import Path
import sys

from pipeline.main import parse_args
from scripts.common.benchmark import (
    _resolve_gold_train_dir,
    collect_hardware_metadata,
    compute_speedup,
    run_benchmark_suite,
    run_with_memory_monitor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_contains_benchmark_defaults() -> None:
    """Verify config.json defines the required default benchmark parameters."""
    cfg = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    assert "benchmark" in cfg
    bench = cfg["benchmark"]
    assert bench["workers"] == ["local[2]", "local[4]", "local[8]"]
    assert bench["data_fractions"] == [0.1, 0.5, 1.0]
    assert bench["fixed_worker_for_size_scaling"] == "local[4]"
    assert isinstance(bench["seed"], int)


def test_parse_args_benchmark_flag() -> None:
    """Verify --benchmark CLI flag is parsed cleanly and defaults to False."""
    assert parse_args([]).benchmark is False
    assert parse_args(["--benchmark"]).benchmark is True


def test_compute_speedup() -> None:
    """Verify speedup calculation and invalid-input handling."""
    assert compute_speedup(100.0, 25.0) == 4.0
    assert compute_speedup(100.0, 100.0) == 1.0
    assert compute_speedup(0.0, 25.0) is None
    assert compute_speedup(100.0, 0.0) is None


def test_collect_hardware_metadata() -> None:
    """Verify hardware metadata includes CPU, RAM, Spark, Python, and OS."""
    meta = collect_hardware_metadata()
    assert meta["cpu_cores_logical"] >= 1
    assert meta["total_ram_gb"] > 0
    assert "spark_version" in meta
    assert "python_version" in meta
    assert "os" in meta


def test_run_with_memory_monitor_captures_time_and_rss() -> None:
    """Verify subprocess wrapper records positive wall time and non-negative RSS."""
    res = run_with_memory_monitor(
        [sys.executable, "-c", "import time; time.sleep(0.3); print('BENCHMARK_ROW_COUNT=42')"],
        poll_interval=0.05,
    )
    assert res["wall_time_seconds"] >= 0.25
    assert res["peak_memory_gb"] >= 0.0
    assert "BENCHMARK_ROW_COUNT=42" in res["stdout"]


def test_run_with_memory_monitor_handles_large_stdout_and_stderr() -> None:
    """Verify large stdout/stderr output does not deadlock the polling loop."""
    script = (
        "import sys; "
        "sys.stderr.write('E' * 262144 + '\\n'); "
        "sys.stdout.write('O' * 262144 + '\\nBENCHMARK_ROW_COUNT=7\\n')"
    )
    res = run_with_memory_monitor([sys.executable, "-c", script], poll_interval=0.05)
    assert "BENCHMARK_ROW_COUNT=7" in res["stdout"]


def test_run_benchmark_suite_writes_results_json(tmp_path: Path, monkeypatch) -> None:
    """Verify run_benchmark_suite writes valid JSON with data-size and worker scaling."""
    def fake_monitor(cmd, poll_interval=0.25, cwd=None):
        return {
            "wall_time_seconds": 10.0 if "local[2]" in cmd else 5.0,
            "peak_memory_gb": 1.25,
            "stdout": "BENCHMARK_ROW_COUNT=1000\n",
        }

    monkeypatch.setattr("scripts.common.benchmark.run_with_memory_monitor", fake_monitor)
    cfg = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    out = run_benchmark_suite(config=cfg, results_dir=tmp_path)
    assert out == tmp_path / "benchmark_results.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "hardware" in data
    assert len(data["data_size_scaling"]) == 3
    assert len(data["worker_scaling"]) == 3
    assert data["worker_scaling"][0]["speedup"] == 1.0
    assert data["worker_scaling"][1]["speedup"] == 2.0
    assert data["workload"] == "preprocessing+logistic_regression"


def test_parse_args_combined_flags() -> None:
    """Verify --benchmark and --ablation can be combined independently."""
    args = parse_args(["--benchmark", "--ablation"])
    assert args.benchmark is True
    assert args.ablation is True


def test_resolve_gold_train_dir_prefers_explicit_then_active_enroute(tmp_path: Path) -> None:
    """Verify _resolve_gold_train_dir resolves explicit gold_dir and latest enroute Gold."""
    explicit = tmp_path / "explicit_gold"
    (explicit / "train").mkdir(parents=True)
    assert _resolve_gold_train_dir(gold_dir=explicit, root=tmp_path) == explicit / "train"

    enroute_gen = tmp_path / "data" / "gold" / "enroute" / "gen_abc"
    (enroute_gen / "train").mkdir(parents=True)
    (enroute_gen / "_SUCCESS").touch()
    assert _resolve_gold_train_dir(gold_dir=None, root=tmp_path) == enroute_gen / "train"


def test_run_worker_task_executes_ml_workload_on_gold_schema(tmp_path: Path, full_feature_rows) -> None:
    """Verify _run_worker_task runs real preprocessing and LogisticRegression fitting in a subprocess."""
    import datetime
    import subprocess

    gold_dir = tmp_path / "gold"
    train_df = full_feature_rows("train", datetime.date(2024, 1, 5))
    train_df.coalesce(1).write.parquet(str(gold_dir / "train"))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.common.benchmark",
            "--master",
            "local[2]",
            "--fraction",
            "1.0",
            "--seed",
            "42",
            "--gold-dir",
            str(gold_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "BENCHMARK_ROW_COUNT=" in proc.stdout
    assert "BENCHMARK_PREPROCESS_SECONDS=" in proc.stdout
    assert "BENCHMARK_FIT_SECONDS=" in proc.stdout


def test_run_benchmark_suite_deduplicates_overlapping_case(tmp_path: Path, monkeypatch) -> None:
    """Verify run_benchmark_suite executes overlapping (local[4], 1.0) once and isolates speedup."""
    calls: list[list[str]] = []

    def fake_monitor(cmd, poll_interval=0.25, cwd=None):
        calls.append(cmd)
        return {
            "wall_time_seconds": 10.0 if "local[2]" in cmd else 5.0,
            "peak_memory_gb": 1.25,
            "stdout": "BENCHMARK_ROW_COUNT=1000\n",
        }

    monkeypatch.setattr("scripts.common.benchmark.run_with_memory_monitor", fake_monitor)
    cfg = json.loads((REPO_ROOT / "config.json").read_text(encoding="utf-8"))
    out = run_benchmark_suite(config=cfg, results_dir=tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert len(calls) == 5
    assert len(data["data_size_scaling"]) == 3
    assert all("speedup" not in row for row in data["data_size_scaling"])
    assert len(data["worker_scaling"]) == 3
    assert all("speedup" in row for row in data["worker_scaling"])


def test_get_spark_session_accepts_custom_master_and_memory(monkeypatch) -> None:
    """Verify get_spark_session applies tuned Spark settings for custom master and driver memory."""
    from scripts.common import spark as spark_mod

    recorded: dict[str, object] = {"configs": {}}

    class FakeBuilder:
        def master(self, value: str):
            recorded["master"] = value
            return self

        def appName(self, value: str):
            recorded["app_name"] = value
            return self

        def config(self, key: str, value: str):
            recorded["configs"][key] = value
            return self

        def getOrCreate(self):
            return recorded

    class FakeSparkSession:
        builder = FakeBuilder()

    monkeypatch.setattr(spark_mod, "SparkSession", FakeSparkSession)
    result = spark_mod.get_spark_session(
        app_name="flight_delay_benchmark",
        master="local[2]",
        driver_memory="8g",
    )

    assert result["master"] == "local[2]"
    assert result["app_name"] == "flight_delay_benchmark"
    assert result["configs"]["spark.driver.memory"] == "8g"
    assert result["configs"]["spark.sql.shuffle.partitions"] == "32"
    assert result["configs"]["spark.local.dir"].endswith(".spark_tmp")
