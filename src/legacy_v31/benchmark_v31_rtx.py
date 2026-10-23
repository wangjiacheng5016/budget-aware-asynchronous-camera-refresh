#!/usr/bin/env python3
"""RTX benchmark for the exact temporal V31 shared-weight student.

Each measured scene executes K sequential current+previous-image passes through
one shared checkpoint.  NVIDIA board power is sampled only during the measured
window; a model-resident idle baseline is measured separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark_v31_jetson import (
    EXPECTED_PARAMETERS,
    build_model,
    make_inputs,
    percentile,
    sha256,
)


POWER_PATTERN = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")


def start_nvidia_smi(path: Path, interval_ms: int):
    stream = path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,power.draw,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ],
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, stream


def stop_sampler(handle) -> None:
    process, stream = handle
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    stream.close()


def parse_samples(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        values = []
        for part in parts[1:]:
            match = POWER_PATTERN.search(part)
            if match is None:
                break
            values.append(float(match.group(1)))
        if len(values) == 3:
            rows.append((values[0], values[1], values[2]))
    power = [row[0] for row in rows]
    utility = [row[1] for row in rows]
    temperature = [row[2] for row in rows]
    return {
        "source": "nvidia-smi gpu board power.draw",
        "samples": len(rows),
        "power_mean_w": statistics.fmean(power) if power else None,
        "power_p95_w": percentile(power, 0.95) if power else None,
        "utilization_mean_pct": statistics.fmean(utility) if utility else None,
        "temperature_peak_c": max(temperature) if temperature else None,
    }


def sample_idle(output_dir: Path, interval_ms: int, seconds: float) -> dict:
    path = output_dir / "nvidia_smi_idle.csv"
    handle = start_nvidia_smi(path, interval_ms)
    try:
        time.sleep(seconds)
    finally:
        stop_sampler(handle)
    return parse_samples(path)


def benchmark(
    model,
    inputs: dict,
    camera_count: int,
    warmup: int,
    iterations: int,
    power_interval_ms: int,
    idle_settle_seconds: float,
    idle_seconds: float,
    output_dir: Path,
) -> dict:
    def forward_once():
        outputs = None
        for _ in range(camera_count):
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = model.forward_outputs(**inputs)
        return outputs

    latencies = []
    with torch.inference_mode():
        for _ in range(warmup):
            forward_once()
        torch.cuda.synchronize()
        time.sleep(idle_settle_seconds)
        idle = sample_idle(output_dir, power_interval_ms, idle_seconds)
        torch.cuda.reset_peak_memory_stats()
        power_path = output_dir / "nvidia_smi_active.csv"
        handle = start_nvidia_smi(power_path, power_interval_ms)
        wall_started = time.perf_counter()
        try:
            for _ in range(iterations):
                started = torch.cuda.Event(enable_timing=True)
                ended = torch.cuda.Event(enable_timing=True)
                started.record()
                forward_once()
                ended.record()
                ended.synchronize()
                latencies.append(float(started.elapsed_time(ended)))
        finally:
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_started
            stop_sampler(handle)

    active = parse_samples(power_path)
    active_power = active["power_mean_w"]
    idle_power = idle["power_mean_w"]
    gross_energy = None if active_power is None else active_power * wall_seconds / iterations
    net_energy = (
        None
        if active_power is None or idle_power is None
        else max(0.0, active_power - idle_power) * wall_seconds / iterations
    )
    with (output_dir / "latencies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", "latency_ms"])
        writer.writerows(enumerate(latencies, 1))
    mean_ms = statistics.fmean(latencies)
    return {
        "runtime": "PyTorch CUDA FP16",
        "camera_count": camera_count,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "wall_seconds": wall_seconds,
        "throughput_scenes_per_second": 1000.0 / mean_ms,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "idle_power": idle,
        "active_power": active,
        "gross_energy_per_scene_j": gross_energy,
        "idle_subtracted_energy_per_scene_j": net_energy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-count", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--power-interval-ms", type=int, default=50)
    parser.add_argument("--idle-settle-seconds", type=float, default=3.0)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()
    if not 1 <= args.camera_count <= 6:
        raise ValueError("camera-count must be between 1 and 6")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    torch.backends.cudnn.benchmark = True
    _training, base, model, checkpoint = build_model(checkpoint_path)
    inputs = make_inputs(base)
    row = benchmark(
        model,
        inputs,
        args.camera_count,
        args.warmup,
        args.iterations,
        args.power_interval_ms,
        args.idle_settle_seconds,
        args.idle_seconds,
        output_dir,
    )
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "forward-only; K sequential complete temporal current+previous passes",
        "energy_scope": "NVIDIA GPU board power; gross and model-resident idle-subtracted",
        "run_label": args.run_label,
        "input_resolution": [640, 360],
        "depth_bins": 32,
        "bev_grid": [200, 200],
        "parameter_count": EXPECTED_PARAMETERS,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "hardware": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "row": row,
    }
    (output_dir / "v31_rtx_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
