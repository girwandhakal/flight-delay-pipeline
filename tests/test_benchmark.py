"""Tests for the systems performance and scalability benchmark suite (Issue #4)."""

import json
from pathlib import Path
import sys

from pipeline.main import parse_args
from scripts.common.benchmark import (
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
