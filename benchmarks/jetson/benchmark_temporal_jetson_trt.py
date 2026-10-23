#!/usr/bin/env python3
"""Jetson Orin NX FP16 TensorRT benchmark for the V31 temporal engine.

Sibling of ``benchmark_activecam_jetson_trt.py``.  This runner targets the V31
``current+previous`` temporal engine, which has eleven inputs (img/k/r/t plus
prev_img/prev_k/prev_r/prev_t/prev_to_current_r/prev_to_current_t/prev_valid)
and four outputs (hm/reg/class_motion/class_size).  It is timing-only: the
previous-frame branch is fed deterministic placeholder tensors
(prev_img = current preprocessed tensor, prev_k/r/t = current calibration,
prev_to_current_r = identity, prev_to_current_t = zeros, prev_valid = 1.0).

Binding names are discovered dynamically from the deserialized engine rather
than reusing the old four-input hardcoded names.  Timing definitions, thermal
gating, cooldown, rotated K order, and the tegrastats VDD_IN window slicing are
all inherited from the original runner so the frozen protocol is preserved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# Make the original runner importable from the new experiment directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorrt as trt
from cuda.bindings import driver

from benchmark_activecam_jetson_trt import (
    CAMERAS, SUBSETS, Engine, check, current_temp_c, parse_tegrastats,
    preprocess, rss_mib, sha256_file, stats_summary, wait_for_cooldown,
)

# Semantic roles of the eleven temporal-engine inputs.
INPUT_ROLES = (
    "img", "k", "r", "t",
    "prev_img", "prev_k", "prev_r", "prev_t",
    "prev_to_current_r", "prev_to_current_t", "prev_valid",
)

# Deterministic previous-frame placeholders (batch size 1).
IDENTITY_R = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
ZERO_T = np.zeros(3, dtype=np.float32).reshape(1, 3)
PREV_VALID = np.ones(1, dtype=np.float32)


def resolve_bindings(engine: Engine) -> tuple[dict[str, str], list[str]]:
    """Discover the engine's actual IO tensor names and map semantic roles.

    Returns (input_by_role, output_names).  Raises if any expected input role is
    absent so a mismatched engine fails fast instead of binding silently.
    """
    input_by_role: dict[str, str] = {}
    output_names: list[str] = []
    for name in engine.names:
        mode = engine.engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    actual_inputs = [
        name for name in engine.names
        if engine.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    available = set(actual_inputs)
    missing = [role for role in INPUT_ROLES if role not in available]
    if missing:
        raise RuntimeError(
            f"temporal engine is missing input bindings {missing}; "
            f"actual inputs are {actual_inputs}"
        )
    for role in INPUT_ROLES:
        input_by_role[role] = role
    return input_by_role, output_names


def upload_temporal_frame(
    engine: Engine, bind: dict[str, str], value: np.ndarray,
    k: np.ndarray, r: np.ndarray, t: np.ndarray,
) -> float:
    """Upload one camera frame: 4 current inputs + 7 previous placeholders."""
    h2d_ms = 0.0
    h2d_ms += engine.upload(bind["img"], value)
    h2d_ms += engine.upload(bind["k"], k)
    h2d_ms += engine.upload(bind["r"], r)
    h2d_ms += engine.upload(bind["t"], t)
    # Deterministic previous-frame placeholders (timing-only protocol).
    h2d_ms += engine.upload(bind["prev_img"], value)
    h2d_ms += engine.upload(bind["prev_k"], k)
    h2d_ms += engine.upload(bind["prev_r"], r)
    h2d_ms += engine.upload(bind["prev_t"], t)
    h2d_ms += engine.upload(bind["prev_to_current_r"], IDENTITY_R)
    h2d_ms += engine.upload(bind["prev_to_current_t"], ZERO_T)
    h2d_ms += engine.upload(bind["prev_valid"], PREV_VALID)
    return h2d_ms


def run_one_temporal(
    engine: Engine, bind: dict[str, str], calibration: dict[str, Any],
    images: Path, cameras: tuple[str, ...],
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    preprocess_ms = 0.0
    h2d_ms = 0.0
    forward_ms = 0.0
    d2h_ms = 0.0
    route_start = time.perf_counter()
    camera_data = [(camera, calibration["channels"][camera]) for camera in cameras]
    routing_ms = (time.perf_counter() - route_start) * 1000.0
    for camera, info in camera_data:
        value, elapsed = preprocess(images / info["image"])
        preprocess_ms += elapsed
        k = np.asarray(info["K_640x360"], dtype=np.float32)[None, ...]
        r = np.asarray(info["R_cam_to_ego"], dtype=np.float32)[None, ...]
        t = np.asarray(info["t_cam_to_ego"], dtype=np.float32)[None, ...]
        h2d_ms += upload_temporal_frame(engine, bind, value[None, ...], k, r, t)
        forward_ms += engine.infer()
        d2h_ms += engine.download()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {
        "camera_count": len(cameras), "cameras": list(cameras),
        "wall_ms": wall_ms, "preprocess_ms": preprocess_ms,
        "h2d_ms": h2d_ms, "model_forward_gpu_ms": forward_ms,
        "output_copy_ms": d2h_ms, "routing_ms": routing_ms,
        "reuse_merge_ms": 0.0, "reuse_merge_scope": "not_applicable_temporal_timing_only",
        "fps": 1000.0 / wall_ms if wall_ms else 0.0,
        "deadline_violation": wall_ms > 500.0,
        "rss_mib": rss_mib(),
        "temp_now_c": current_temp_c(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--onnx", default=None,
                        help="path to the source ONNX, used to record onnx_sha256")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-temp-max", type=float, default=70.0)
    parser.add_argument("--abort-temp", type=float, default=90.0)
    parser.add_argument("--cooldown-target-c", type=float, default=70.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--power-mode", default="25W locked clocks (nvpmodel mode 1)")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=100)
    parser.add_argument("--camera-counts", type=int, nargs="+", default=[3, 4, 6])
    args = parser.parse_args()
    bundle = Path(args.bundle)
    engine_path = Path(args.engine)
    output = Path(args.output)
    if current_temp_c() > args.start_temp_max:
        raise RuntimeError(
            f"thermal gate: current max temperature {current_temp_c():.2f}C "
            f"> {args.start_temp_max:.2f}C"
        )
    calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))
    engine = Engine(engine_path)
    bind, output_names = resolve_bindings(engine)
    free_before, _ = check(driver.cuMemGetInfo(), "cuMemGetInfo(before)")
    # A short initial pass confirms every real JPEG, calibration entry, and all
    # eleven temporal bindings on each of the six cameras.
    for camera in CAMERAS:
        info = calibration["channels"][camera]
        value, _ = preprocess(bundle / "images" / info["image"])
        k = np.asarray(info["K_640x360"], dtype=np.float32)[None, ...]
        r = np.asarray(info["R_cam_to_ego"], dtype=np.float32)[None, ...]
        t = np.asarray(info["t_cam_to_ego"], dtype=np.float32)[None, ...]
        upload_temporal_frame(engine, bind, value[None, ...], k, r, t)
        engine.infer(); engine.download()
    check(driver.cuStreamSynchronize(engine.stream), "cuStreamSynchronize(smoke)")
    free_after, _ = check(driver.cuMemGetInfo(), "cuMemGetInfo(after)")
    gpu_alloc_mib = max(0.0, (int(free_before) - int(free_after)) / (1024.0 * 1024.0))

    stats_rows: list[dict[str, Any]] = []
    stats_proc = subprocess.Popen(
        ["tegrastats", "--interval", str(args.tegrastats_interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def collect() -> None:
        assert stats_proc.stdout is not None
        for line in stats_proc.stdout:
            if line.strip():
                stats_rows.append(parse_tegrastats(line))

    thread = threading.Thread(target=collect, daemon=True); thread.start()
    requested = tuple(dict.fromkeys(args.camera_counts))
    if not requested or any(k not in SUBSETS for k in requested):
        raise ValueError(f"camera counts must be a non-empty subset of {sorted(SUBSETS)}")
    if requested == (3, 4, 6):
        order = ((3, 4, 6), (4, 6, 3), (6, 3, 4))
    elif requested == (3, 6):
        order = ((3, 6), (6, 3), (3, 6))
    else:
        order = (requested, requested[::-1], requested)
    rows: list[dict[str, Any]] = []
    try:
        for repeat in range(min(args.repeats, len(order))):
            for k in order[repeat]:
                cameras = SUBSETS[k]
                cooldown_s = wait_for_cooldown(args.cooldown_target_c, args.cooldown_timeout_s)
                if current_temp_c() > args.start_temp_max:
                    raise RuntimeError(
                        f"thermal gate before K{k}: {current_temp_c():.2f}C "
                        f"> {args.start_temp_max:.2f}C"
                    )
                for _ in range(args.warmup):
                    run_one_temporal(engine, bind, calibration, bundle / "images", cameras)
                start = time.time()
                measured = [
                    run_one_temporal(engine, bind, calibration, bundle / "images", cameras)
                    for _ in range(args.measured)
                ]
                end = time.time()
                if current_temp_c() >= args.abort_temp:
                    raise RuntimeError(
                        f"thermal abort after K{k}: {current_temp_c():.2f}C "
                        f">= {args.abort_temp:.2f}C"
                    )
                summary = stats_summary(stats_rows, start, end)
                wall = [float(x["wall_ms"]) for x in measured]
                forward = [float(x["model_forward_gpu_ms"]) for x in measured]
                p95 = float(np.percentile(wall, 95))
                row = {
                    "repeat": repeat + 1, "camera_count": k, "cameras": list(cameras),
                    "cooldown_before_s": float(cooldown_s),
                    "measured_frames": len(measured), "mean_wall_ms": float(np.mean(wall)),
                    "p50_wall_ms": float(np.percentile(wall, 50)), "p95_wall_ms": p95,
                    "fps": 1000.0 / float(np.mean(wall)),
                    "mean_model_forward_gpu_ms": float(np.mean(forward)),
                    "mean_preprocess_ms": float(np.mean([x["preprocess_ms"] for x in measured])),
                    "mean_h2d_ms": float(np.mean([x["h2d_ms"] for x in measured])),
                    "mean_output_copy_ms": float(np.mean([x["output_copy_ms"] for x in measured])),
                    "mean_routing_ms": float(np.mean([x["routing_ms"] for x in measured])),
                    "reuse_merge_ms": 0.0,
                    "reuse_merge_scope": "not_applicable_temporal_timing_only",
                    "deadline_violations": int(sum(bool(x["deadline_violation"]) for x in measured)),
                    "peak_rss_mib": float(max(x["rss_mib"] for x in measured)),
                    "peak_gpu_allocated_mib_est": gpu_alloc_mib,
                    "energy_per_frame_gross_j": (
                        summary["power_mean_w"] * float(np.mean(wall)) / 1000.0
                        if summary["power_mean_w"] is not None else None),
                    "power_scope": "VDD_IN_module_rail",
                    **summary,
                }
                rows.append(row)
    finally:
        stats_proc.terminate()
        try: stats_proc.wait(timeout=3)
        except subprocess.TimeoutExpired: stats_proc.kill()
        engine.close()

    onnx_sha256 = None
    onnx_bytes = None
    onnx_path = Path(args.onnx) if args.onnx else (engine_path.parent / "v31_temporal_detection.onnx")
    if onnx_path.exists():
        onnx_sha256 = sha256_file(onnx_path)
        onnx_bytes = onnx_path.stat().st_size

    payload = {
        "protocol": {
            "device_scope": f"Jetson Orin NX 16GB, {args.power_mode}, PWM255",
            "precision": "TensorRT FP16 engine; input/output bindings are FP32",
            "batch_size": 1, "camera_counts": list(requested),
            "warmup_per_run": args.warmup, "measured_per_run": args.measured,
            "repeats": min(args.repeats, 3), "orders": order[:min(args.repeats, 3)],
            "scope": (
                "V31 current+previous temporal engine; timing only; deterministic "
                "placeholder previous-frame inputs (prev_img=current, identity "
                "prev_to_current, prev_valid=1)"
            ),
            "input_bindings": [bind[role] for role in INPUT_ROLES],
            "output_bindings": output_names,
            "onnx_sha256": onnx_sha256,
            "engine_sha256": sha256_file(engine_path),
            "quality_metrics": "not re-evaluated on Jetson; inherit frozen validation quality only outside this benchmark",
            "parity": "TensorRT repeatability smoke only; PyTorch/reference output unavailable on this fresh device",
            "deadline_definition": "wall_ms > 500ms per processed frame",
            "energy_definition": "VDD_IN module-rail mean power × frame wall time; not sensor/vehicle energy",
        },
        "engine_sha256": sha256_file(engine_path), "engine_bytes": engine_path.stat().st_size,
        "onnx_sha256": onnx_sha256, "onnx_bytes": onnx_bytes,
        "calibration_sha256": sha256_file(bundle / "calibration.json"),
        "bundle": str(bundle), "rows": rows,
        "all_tegrastats_rows": stats_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "rows": len(rows),
        "engine_sha256": payload["engine_sha256"], "onnx_sha256": onnx_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
