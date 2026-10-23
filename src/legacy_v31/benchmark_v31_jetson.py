#!/usr/bin/env python3
"""Jetson benchmark for the exact V31 shared-weight student.

The primary scope is model-forward latency. Each K-camera sample is evaluated by
K sequential executions of the same single-view weights, matching the deployed
variable-camera design. Raw per-iteration latency and tegrastats logs are kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "train_official_nuscenes_nds_kd_v2.py"
EXPECTED_PARAMETERS = 7_191_090


def load_training_module():
    spec = importlib.util.spec_from_file_location("p1_jetson_training", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def command_output(command: list[str]) -> dict:
    if shutil.which(command[0]) is None:
        return {"returncode": 127, "stdout": "", "stderr": "not found"}
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def start_tegrastats(path: Path, interval_ms: int):
    executable = shutil.which("tegrastats")
    if executable is None:
        return None
    stream = path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [executable, "--interval", str(interval_ms)],
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, stream


def stop_tegrastats(handle) -> None:
    if handle is None:
        return
    process, stream = handle
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    stream.close()


def parse_power(path: Path) -> dict:
    if not path.exists():
        return {"source": None, "samples": 0, "mean_mw": None, "p95_mw": None}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    patterns = (
        ("VDD_IN", re.compile(r"\bVDD_IN\s+(\d+)mW(?:/(\d+)mW)?")),
        ("POM_5V_IN", re.compile(r"\bPOM_5V_IN\s+(\d+)(?:/(\d+))?")),
    )
    for source, pattern in patterns:
        values = [float(match.group(1)) for line in lines if (match := pattern.search(line))]
        if values:
            return {
                "source": source,
                "samples": len(values),
                "mean_mw": statistics.fmean(values),
                "p95_mw": percentile(values, 0.95),
            }
    return {"source": None, "samples": 0, "mean_mw": None, "p95_mw": None}


def build_model(checkpoint_path: Path):
    training = load_training_module()
    base = training.load_base_module()
    config = SimpleNamespace(
        teacher_img_w=640,
        teacher_img_h=360,
        student_img_w=640,
        student_img_h=360,
        depth_bins=32,
        bev_x_min=-50.0,
        bev_x_max=50.0,
        bev_y_min=-50.0,
        bev_y_max=50.0,
        bev_res=0.5,
        fast_runtime=True,
        compact_backbone=True,
        splat_depth_chunk=4,
    )
    training.configure_bev(base, config)
    training.configure_fast_runtime(base, config)
    model = training.OfficialStudent(
        base,
        pretrained_backbone=False,
        use_kd_adapter=False,
        backbone_variant="multiscale",
        camera_aware_depth=False,
        bev_norm="group",
        attribute_head=True,
        temporal_bev=True,
        temporal_pre_encoder=True,
        temporal_delta_limit=16.0,
        class_aware_motion=True,
        class_aware_size=True,
    )
    training.compact_student_backbone(model, enabled=True)
    device = torch.device("cuda")
    training.instantiate_student_lazy(base, model, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    training.load_model_state_compatible(model, checkpoint, strict=True)
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(f"parameter count {parameter_count} != {EXPECTED_PARAMETERS}")
    return training, base, model, checkpoint


def make_inputs(base):
    device = torch.device("cuda")
    image = torch.randn(1, 3, base.IMG_H, base.IMG_W, device=device)
    intrinsic = torch.tensor(
        [[800.0, 0.0, base.IMG_W / 2], [0.0, 800.0, base.IMG_H / 2], [0.0, 0.0, 1.0]],
        device=device,
    ).unsqueeze(0)
    rotation = torch.eye(3, device=device).unsqueeze(0)
    translation = torch.tensor([[1.5, 0.0, 1.5]], device=device)
    return {
        "img": image,
        "k": intrinsic,
        "r": rotation,
        "t": translation,
        "prev_img": torch.randn_like(image),
        "prev_k": intrinsic,
        "prev_r": rotation,
        "prev_t": translation,
        "prev_to_current_r": rotation,
        "prev_to_current_t": torch.zeros(1, 3, device=device),
        "prev_valid": torch.ones(1, device=device),
    }


def max_temperature_c() -> float:
    values = []
    for path in Path("/sys/devices/virtual/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(path.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(values) if values else float("nan")


def wait_for_cooldown(target_c: float, timeout_seconds: int) -> float:
    started = time.monotonic()
    while True:
        temperature = max_temperature_c()
        if temperature != temperature or temperature <= target_c:
            return temperature
        if time.monotonic() - started >= timeout_seconds:
            raise RuntimeError(
                f"cooldown timed out at {temperature:.1f} C (target {target_c:.1f} C)"
            )
        print(f"[COOLDOWN] {temperature:.1f} C > {target_c:.1f} C", flush=True)
        time.sleep(10)


def benchmark_arm(model, inputs, camera_count: int, warmup: int, iterations: int,
                  power_interval_ms: int, output_dir: Path, max_temp_c: float,
                  cooldown_target_c: float, cooldown_timeout_seconds: int,
                  segment_size: int):
    def forward_once():
        outputs = None
        for _ in range(camera_count):
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = model.forward_outputs(**inputs)
        return outputs

    if segment_size <= 0:
        segment_size = iterations
    segment_sizes = []
    remaining = iterations
    while remaining:
        current = min(segment_size, remaining)
        segment_sizes.append(current)
        remaining -= current

    latencies = []
    segment_records = []
    power_paths = []
    wall_seconds = 0.0
    peak_temperature_c = float("-inf")
    start_temperature_c = None
    warmup_temperature_c = None
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for segment_index, current_iterations in enumerate(segment_sizes, 1):
            segment_start_c = wait_for_cooldown(
                cooldown_target_c, cooldown_timeout_seconds
            )
            if start_temperature_c is None:
                start_temperature_c = segment_start_c
            segment_warmup = warmup if segment_index == 1 else min(warmup, 10)
            for _ in range(segment_warmup):
                forward_once()
            segment_warmup_c = max_temperature_c()
            if warmup_temperature_c is None:
                warmup_temperature_c = segment_warmup_c
            peak_temperature_c = max(
                peak_temperature_c, segment_start_c, segment_warmup_c
            )
            if segment_warmup_c >= max_temp_c:
                raise RuntimeError(
                    f"thermal guard after segment {segment_index} warmup: "
                    f"{segment_warmup_c:.1f} C >= {max_temp_c:.1f} C"
                )
            torch.cuda.synchronize()
            power_path = output_dir / (
                f"tegrastats_k{camera_count}_segment{segment_index:02d}.log"
            )
            power_paths.append(power_path)
            power_handle = start_tegrastats(power_path, power_interval_ms)
            segment_latencies = []
            wall_started = time.perf_counter()
            try:
                for iteration in range(current_iterations):
                    started = torch.cuda.Event(enable_timing=True)
                    ended = torch.cuda.Event(enable_timing=True)
                    started.record()
                    forward_once()
                    ended.record()
                    ended.synchronize()
                    segment_latencies.append(float(started.elapsed_time(ended)))
                    if iteration % 10 == 9 or iteration + 1 == current_iterations:
                        temperature = max_temperature_c()
                        peak_temperature_c = max(peak_temperature_c, temperature)
                        if temperature >= max_temp_c:
                            raise RuntimeError(
                                f"thermal guard in segment {segment_index} at iteration "
                                f"{iteration + 1}: {temperature:.1f} C >= {max_temp_c:.1f} C"
                            )
            finally:
                stop_tegrastats(power_handle)
            torch.cuda.synchronize()
            wall_seconds += time.perf_counter() - wall_started
            latencies.extend(segment_latencies)
            segment_peak_c = max_temperature_c()
            peak_temperature_c = max(peak_temperature_c, segment_peak_c)
            segment_records.append({
                "segment": segment_index,
                "warmup_iterations": segment_warmup,
                "measured_iterations": current_iterations,
                "temperature_start_c": segment_start_c,
                "temperature_after_warmup_c": segment_warmup_c,
                "temperature_end_c": segment_peak_c,
            })
            with (output_dir / f"latencies_k{camera_count}_segment{segment_index:02d}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["iteration", "latency_ms"])
                writer.writerows(enumerate(segment_latencies, 1))

    combined_power_path = output_dir / f"tegrastats_k{camera_count}.log"
    with combined_power_path.open("w", encoding="utf-8") as combined:
        for power_path in power_paths:
            if power_path.exists():
                combined.write(power_path.read_text(encoding="utf-8", errors="replace"))
    power = parse_power(combined_power_path)
    mean_ms = statistics.fmean(latencies)
    mean_power_w = None if power["mean_mw"] is None else power["mean_mw"] / 1000.0
    energy_j = None if mean_power_w is None else mean_power_w * mean_ms / 1000.0
    with (output_dir / f"latencies_k{camera_count}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", "latency_ms"])
        writer.writerows(enumerate(latencies, 1))
    return {
        "runtime": "PyTorch CUDA FP16",
        "camera_count": camera_count,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p90_ms": percentile(latencies, 0.90),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "throughput_scenes_per_second": 1000.0 / mean_ms,
        "throughput_camera_images_per_second": 1000.0 * camera_count / mean_ms,
        "wall_seconds": wall_seconds,
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "power": power,
        "energy_per_scene_j": energy_j,
        "temperature_start_c": start_temperature_c,
        "temperature_after_warmup_c": warmup_temperature_c,
        "temperature_peak_c": peak_temperature_c,
        "thermal_protocol": (
            "continuous" if len(segment_sizes) == 1
            else "segmented timing with full cooldown between segments"
        ),
        "segment_size": segment_size,
        "segments": segment_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--camera-counts", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--power-interval-ms", type=int, default=100)
    parser.add_argument("--max-temp-c", type=float, default=90.0)
    parser.add_argument("--cooldown-target-c", type=float, default=70.0)
    parser.add_argument("--cooldown-timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--segment-size", type=int, default=0,
        help="Measured iterations per thermal segment; 0 runs continuously.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(count < 1 or count > 6 for count in args.camera_counts):
        raise ValueError("camera counts must be between 1 and 6")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    torch.backends.cudnn.benchmark = True
    _training, base, model, checkpoint = build_model(checkpoint_path)
    inputs = make_inputs(base)
    rows = []
    for count in args.camera_counts:
        wait_for_cooldown(args.cooldown_target_c, args.cooldown_timeout_seconds)
        rows.append(benchmark_arm(
            model, inputs, count, args.warmup, args.iterations,
            args.power_interval_ms, output_dir, args.max_temp_c,
            args.cooldown_target_c, args.cooldown_timeout_seconds,
            args.segment_size,
        ))
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "model forward only; K sequential passes of one shared V31 checkpoint",
        "method": "V31 shared-balanced",
        "paper_accuracy": {"NDS": 0.2951, "mAP": 0.2123},
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
        "nvpmodel": command_output(["nvpmodel", "-q", "--verbose"]),
        "jetson_clocks": command_output(["jetson_clocks", "--show"]),
        "rows": rows,
    }
    (output_dir / "v31_jetson_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "v31_jetson_benchmark.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "runtime", "camera_count", "latency_mean_ms", "latency_p50_ms",
            "latency_p90_ms", "latency_p95_ms", "latency_p99_ms",
            "throughput_scenes_per_second", "throughput_camera_images_per_second",
            "peak_memory_allocated_mib", "peak_memory_reserved_mib", "energy_per_scene_j",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
