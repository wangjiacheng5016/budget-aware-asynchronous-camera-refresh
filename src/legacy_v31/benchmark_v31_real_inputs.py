#!/usr/bin/env python3
"""Online-style V31 benchmark with real nuScenes current and previous images."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark_v31_jetson import EXPECTED_PARAMETERS, build_model, percentile, sha256
from benchmark_v31_rtx import parse_samples, sample_idle, start_nvidia_smi, stop_sampler


def move_student_item(item, device: torch.device) -> tuple:
    current = tuple(item[index].unsqueeze(0).to(device) for index in range(5, 9))
    temporal = tuple(item[index].unsqueeze(0).to(device) for index in range(18, 25))
    return current + temporal


def selected_camera_indices(sample_index: int, camera_count: int) -> tuple[int, ...]:
    """Balance file access across cameras without reading any unselected view."""
    start = sample_index % 6
    return tuple((start + offset) % 6 for offset in range(camera_count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-count", type=int, required=True)
    parser.add_argument("--warmup-scenes", type=int, default=10)
    parser.add_argument("--measured-scenes", type=int, default=100)
    parser.add_argument("--power-interval-ms", type=int, default=50)
    parser.add_argument("--idle-settle-seconds", type=float, default=3.0)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
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
    training, base, model, checkpoint = build_model(checkpoint_path)
    transform_teacher, transform_student = training.make_transforms(base)
    dataset = training.NuScenesOfficialDetectionDataset(
        base,
        args.data_root,
        "v1.0-trainval",
        "val",
        transform_teacher,
        transform_student,
        max_samples=args.warmup_scenes + args.measured_scenes,
        build_depth=False,
        need_teacher_inputs=False,
        need_student_inputs=True,
        need_teacher_depth=False,
        need_student_depth=False,
        need_temporal_inputs=True,
        need_history_targets=False,
        build_detection_targets=False,
        target_scope="student_fov",
        student_view_mode="front",
        student_camera="CAM_FRONT",
        student_cameras=list(base.CAM_CHANNELS),
    )
    device = torch.device("cuda")

    def run_scene(sample_index: int) -> tuple[float, float, float]:
        scene_started = time.perf_counter()
        data_seconds = 0.0
        gpu_ms = 0.0
        for camera_index in selected_camera_indices(sample_index, args.camera_count):
            data_started = time.perf_counter()
            item = dataset[(sample_index, camera_index)]
            inputs = move_student_item(item, device)
            data_seconds += time.perf_counter() - data_started
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                model.forward_outputs(*inputs)
            ended.record()
            ended.synchronize()
            gpu_ms += float(started.elapsed_time(ended))
        return (
            (time.perf_counter() - scene_started) * 1000.0,
            data_seconds * 1000.0,
            gpu_ms,
        )

    with torch.inference_mode():
        for sample_index in range(args.warmup_scenes):
            run_scene(sample_index)
        torch.cuda.synchronize()
        time.sleep(args.idle_settle_seconds)
        idle = sample_idle(output_dir, args.power_interval_ms, args.idle_seconds)
        torch.cuda.reset_peak_memory_stats()
        power_path = output_dir / "nvidia_smi_active.csv"
        power_handle = start_nvidia_smi(power_path, args.power_interval_ms)
        rows = []
        measured_started = time.perf_counter()
        try:
            for offset in range(args.measured_scenes):
                sample_index = args.warmup_scenes + offset
                wall_ms, data_ms, gpu_ms = run_scene(sample_index)
                rows.append((offset + 1, wall_ms, data_ms, gpu_ms))
        finally:
            torch.cuda.synchronize()
            measured_wall_seconds = time.perf_counter() - measured_started
            stop_sampler(power_handle)

    active = parse_samples(power_path)
    active_power = active["power_mean_w"]
    idle_power = idle["power_mean_w"]
    gross_energy = (
        None
        if active_power is None
        else active_power * measured_wall_seconds / args.measured_scenes
    )
    net_energy = (
        None
        if active_power is None or idle_power is None
        else max(0.0, active_power - idle_power)
        * measured_wall_seconds
        / args.measured_scenes
    )
    wall_values = [row[1] for row in rows]
    data_values = [row[2] for row in rows]
    gpu_values = [row[3] for row in rows]
    with (output_dir / "real_input_latencies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene", "wall_ms", "decode_preprocess_h2d_ms", "gpu_forward_ms"])
        writer.writerows(rows)
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "real nuScenes image decode, resize/normalize, calibration, H2D, and complete "
            "temporal current+previous forward; excludes box decode and memory merge"
        ),
        "camera_selection_for_timing": "balanced cyclic K-view access; unselected views unread",
        "camera_count": args.camera_count,
        "warmup_scenes": args.warmup_scenes,
        "measured_scenes": args.measured_scenes,
        "latency_wall_mean_ms": statistics.fmean(wall_values),
        "latency_wall_p50_ms": percentile(wall_values, 0.50),
        "latency_wall_p95_ms": percentile(wall_values, 0.95),
        "decode_preprocess_h2d_mean_ms": statistics.fmean(data_values),
        "gpu_forward_mean_ms": statistics.fmean(gpu_values),
        "measured_wall_seconds": measured_wall_seconds,
        "throughput_scenes_per_second": 1000.0 / statistics.fmean(wall_values),
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "idle_power": idle,
        "active_power": active,
        "gross_energy_per_scene_j": gross_energy,
        "idle_subtracted_energy_per_scene_j": net_energy,
        "parameter_count": EXPECTED_PARAMETERS,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (output_dir / "v31_real_input_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
