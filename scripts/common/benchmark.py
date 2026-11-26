"""Systems performance and scalability benchmark runner for Issue #4."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional

import psutil
import pyspark

from .config import ROOT
from .results import replace_file
from .spark import configure_spark_env


def collect_hardware_metadata() -> dict[str, Any]:
    """Collect CPU, RAM, Python, Spark, and OS metadata for benchmark reproducibility."""
    mem_bytes = psutil.virtual_memory().total
    return {
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(mem_bytes / (1024 ** 3), 2),
        "spark_version": pyspark.__version__,
        "python_version": platform.python_version(),
        "os": platform.platform(),
    }


def _sample_process_tree_rss_bytes(proc: psutil.Process) -> int:
    """Return total RSS in bytes for proc and all live child processes."""
    total = 0
    try:
        total += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def run_with_memory_monitor(
    cmd: list[str],
    poll_interval: float = 0.25,
    cwd: Optional[Path] = None,
) -> dict[str, Any]:
    """Run a command in a subprocess while polling peak RSS memory (GB) and wall time (s)."""
    started = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ps_proc = psutil.Process(process.pid)
    peak_rss_bytes = 0

    while process.poll() is None:
        peak_rss_bytes = max(peak_rss_bytes, _sample_process_tree_rss_bytes(ps_proc))
        time.sleep(poll_interval)

    stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark subprocess failed (exit {process.returncode}): {' '.join(cmd)}\n{stderr}"
        )

    return {
        "wall_time_seconds": round(elapsed, 2),
        "peak_memory_gb": round(peak_rss_bytes / (1024 ** 3), 3),
        "stdout": stdout,
    }


def compute_speedup(baseline_seconds: float, target_seconds: float) -> Optional[float]:
    """Compute speedup as baseline_seconds / target_seconds, returning None if invalid."""
    if baseline_seconds <= 0 or target_seconds <= 0:
        return None
    return round(baseline_seconds / target_seconds, 2)


def _parse_row_count(stdout: str) -> int:
    """Extract BENCHMARK_ROW_COUNT=<int> from worker stdout."""
    match = re.search(r"BENCHMARK_ROW_COUNT=(\d+)", stdout)
    if not match:
        raise ValueError(f"Could not parse BENCHMARK_ROW_COUNT from worker output:\n{stdout}")
    return int(match.group(1))


def _write_benchmark_results(payload: dict[str, Any], results_dir: Path) -> Path:
    """Atomically write benchmark_results.json into results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "benchmark_results.json"
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(prefix=".benchmark-", dir=results_dir) as tmp:
        staged = Path(tmp) / "benchmark_results.json"
        staged.write_text(serialized, encoding="utf-8")
        replace_file(staged, target)
    return target


def _print_summary_table(data_rows: list[dict[str, Any]], worker_rows: list[dict[str, Any]]) -> None:
    """Print a concise summary table of scaling results to stdout."""
    print("\nBENCHMARK SUMMARY (Issue #4)", flush=True)
    print("-" * 78, flush=True)
    print(f"{'Sweep':<14} {'Fraction':>8} {'Rows':>10} {'Workers':>10} {'Time (s)':>10} {'Speedup':>8} {'Peak GB':>9}", flush=True)
    print("-" * 78, flush=True)
    for row in data_rows:
        print(
            f"{'Data Size':<14} {row['data_fraction']:>8.0%} {row['rows']:>10,} "
            f"{row['workers']:>10} {row['wall_time_seconds']:>10.2f} {'—':>8} {row['peak_memory_gb']:>9.3f}",
            flush=True,
        )
    for row in worker_rows:
        sp = f"{row['speedup']:.2f}x" if row["speedup"] is not None else "—"
        print(
            f"{'Workers':<14} {row['data_fraction']:>8.0%} {row['rows']:>10,} "
            f"{row['workers']:>10} {row['wall_time_seconds']:>10.2f} {sp:>8} {row['peak_memory_gb']:>9.3f}",
            flush=True,
        )
    print("-" * 78, flush=True)


def run_benchmark_suite(
    config: Optional[dict[str, Any]] = None,
    results_dir: Optional[Path] = None,
) -> Path:
    """Run data-size and worker scaling benchmarks and publish results/benchmark_results.json."""
    cfg = config if config is not None else json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    bench_cfg = cfg["benchmark"]
    workers: list[str] = bench_cfg["workers"]
    fractions: list[float] = bench_cfg["data_fractions"]
    fixed_worker: str = bench_cfg["fixed_worker_for_size_scaling"]
    seed: int = int(bench_cfg["seed"])
    poll_interval: float = float(bench_cfg.get("poll_interval_seconds", 0.25))

    def _run_case(master: str, fraction: float) -> dict[str, Any]:
        cmd = [
            sys.executable,
            "-m",
            "scripts.common.benchmark",
            "--master",
            master,
            "--fraction",
            str(fraction),
            "--seed",
            str(seed),
        ]
        res = run_with_memory_monitor(cmd, poll_interval=poll_interval)
        return {
            "framework": "Spark",
            "data_fraction": fraction,
            "rows": _parse_row_count(res["stdout"]),
            "workers": master,
            "wall_time_seconds": res["wall_time_seconds"],
            "peak_memory_gb": res["peak_memory_gb"],
        }

    data_size_scaling = [_run_case(fixed_worker, frac) for frac in fractions]

    worker_scaling = []
    baseline_time: Optional[float] = None
    for master in workers:
        row = _run_case(master, 1.0)
        if baseline_time is None:
            baseline_time = row["wall_time_seconds"]
        row["speedup"] = compute_speedup(baseline_time, row["wall_time_seconds"])
        worker_scaling.append(row)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": collect_hardware_metadata(),
        "data_size_scaling": data_size_scaling,
        "worker_scaling": worker_scaling,
    }
    out_path = _write_benchmark_results(payload, results_dir or (ROOT / "results"))
    _print_summary_table(data_size_scaling, worker_scaling)
    print(f"Benchmark results written to: {out_path}", flush=True)
    return out_path


def _run_worker_task(master: str, fraction: float, seed: int) -> None:
    """Execute a deterministic Spark workload on Gold/Silver data for one benchmark point."""
    from pyspark.sql import SparkSession

    configure_spark_env()
    python_path = os.environ.get("PYTHONPATH", "")
    if str(ROOT) not in python_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = str(ROOT) + (os.pathsep + python_path if python_path else "")

    spark = (
        SparkSession.builder.master(master)
        .appName("flight_delay_benchmark")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    try:
        candidates = [
            ROOT / "data" / "gold" / "flight_delay" / "train",
            ROOT / "data" / "silver" / "bts",
        ]
        data_path = next((p for p in candidates if p.exists()), None)
        if data_path is not None:
            df = spark.read.parquet(str(data_path))
        else:
            # Fallback synthetic dataset when local bulk data is absent (e.g., CI/smoke tests).
            df = spark.range(0, 10000).withColumnRenamed("id", "FlightID")

        if fraction < 1.0:
            df = df.sample(withReplacement=False, fraction=fraction, seed=seed)

        row_count = df.count()
        if "Reporting_Airline" in df.columns:
            df.groupBy("Reporting_Airline").count().collect()
        print(f"BENCHMARK_ROW_COUNT={row_count}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal benchmark worker entrypoint.")
    parser.add_argument("--master", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _run_worker_task(args.master, args.fraction, args.seed)
