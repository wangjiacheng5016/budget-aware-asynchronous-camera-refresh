#!/usr/bin/env python3
"""Jetson Orin NX FP16 TensorRT benchmark for the current-camera engine.

This runner intentionally uses only the small TensorRT engine and the six-image
real bundle.  The engine has four inputs (image plus K/R/t) and therefore does
not contain the V31 temporal ``current+previous`` branch.  Results are labeled
as a current-only TensorRT deployment benchmark; they must not be merged with
the temporal RTX table as if they were the same graph.

The script avoids PyTorch and uses the CUDA 12.6 Python driver bindings.  It
records JPEG decode/preprocess, H2D, TensorRT forward, output copy, RSS, CUDA
allocation, tegrastats power/temperature/utilisation, and a repeatability
parity check.  It refuses to start above the configured thermal limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import tensorrt as trt
from cuda.bindings import driver


CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
SUBSETS = {
    3: ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"),
    4: ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"),
    6: CAMERAS,
}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(result: Any, name: str) -> Any:
    """Unwrap a CUDA Python result and raise on a non-success code."""
    if isinstance(result, tuple):
        err, *values = result
    else:
        err, values = result, []
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name} failed: {err}")
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return tuple(values)


def current_temp_c() -> float:
    values = []
    for p in Path("/sys/devices/virtual/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(p.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(values) if values else 0.0


def rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def parse_tegrastats(raw: str) -> dict[str, Any]:
    row: dict[str, Any] = {"time": time.time(), "raw": raw.strip()}
    m = re.search(r"VDD_IN\s+(\d+)mW", raw)
    if m:
        row["power_w"] = float(m.group(1)) / 1000.0
    temps = []
    for key in ("tj", "gpu", "cpu", "soc1"):
        m = re.search(rf"{key}@([0-9.]+)C", raw)
        if m:
            row[f"{key}_c"] = float(m.group(1))
            temps.append(float(m.group(1)))
    if temps:
        row["max_temp_c"] = max(temps)
    m = re.search(r"GR3D_FREQ\s+(\d+)%", raw)
    if m:
        row["gpu_util_pct"] = float(m.group(1))
    m = re.search(r"RAM\s+(\d+)/(\d+)MB", raw)
    if m:
        row["ram_used_mib"] = float(m.group(1))
        row["ram_total_mib"] = float(m.group(2))
    return row


def stats_summary(rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    selected = [r for r in rows if start <= float(r["time"]) <= end]
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in selected if key in r]
    power = vals("power_w")
    temp = vals("max_temp_c")
    gpu = vals("gpu_util_pct")
    ram = vals("ram_used_mib")
    return {
        "tegrastats_samples": len(selected),
        "power_mean_w": float(np.mean(power)) if power else None,
        "power_peak_w": float(np.max(power)) if power else None,
        "temp_start_c": float(temp[0]) if temp else None,
        "temp_end_c": float(temp[-1]) if temp else None,
        "temp_peak_c": float(np.max(temp)) if temp else None,
        "gpu_util_mean_pct": float(np.mean(gpu)) if gpu else None,
        "gpu_util_peak_pct": float(np.max(gpu)) if gpu else None,
        "ram_peak_mib": float(np.max(ram)) if ram else None,
    }


def preprocess(path: Path) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    with Image.open(path) as image:
        resampling = getattr(Image, "Resampling", Image)
        image = image.convert("RGB").resize((640, 360), resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    value = ((array - MEAN) / STD).transpose(2, 0, 1)
    return np.ascontiguousarray(value, dtype=np.float32), (time.perf_counter() - started) * 1000.0


def dtype_for(value: trt.DataType) -> np.dtype:
    if value == trt.DataType.FLOAT:
        return np.dtype(np.float32)
    if value == trt.DataType.HALF:
        return np.dtype(np.float16)
    if value == trt.DataType.INT32:
        return np.dtype(np.int32)
    raise TypeError(f"unsupported TensorRT dtype {value}")


class Engine:
    def __init__(self, path: Path):
        check(driver.cuInit(0), "cuInit")
        device = check(driver.cuDeviceGet(0), "cuDeviceGet")
        self.context_handle = check(driver.cuCtxCreate(0, device), "cuCtxCreate")
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self.execution = self.engine.create_execution_context()
        self.stream = check(driver.cuStreamCreate(0), "cuStreamCreate")
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs: dict[str, np.ndarray] = {}
        self.outputs: dict[str, np.ndarray] = {}
        self.device_ptrs: dict[str, Any] = {}
        for name in self.names:
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            host = np.empty(shape, dtype=dtype_for(self.engine.get_tensor_dtype(name)))
            ptr = check(driver.cuMemAlloc(host.nbytes), f"cuMemAlloc({name})")
            self.device_ptrs[name] = ptr
            self.execution.set_tensor_address(name, int(ptr))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = host
            else:
                self.outputs[name] = host

    def upload(self, name: str, value: np.ndarray) -> float:
        target = self.inputs[name]
        np.copyto(target, value.astype(target.dtype, copy=False), casting="unsafe")
        started = time.perf_counter()
        check(driver.cuMemcpyHtoD(self.device_ptrs[name], target.ctypes.data, target.nbytes), f"HtoD({name})")
        return (time.perf_counter() - started) * 1000.0

    def infer(self) -> float:
        start = check(driver.cuEventCreate(0), "cuEventCreate(start)")
        end = check(driver.cuEventCreate(0), "cuEventCreate(end)")
        check(driver.cuEventRecord(start, self.stream), "cuEventRecord(start)")
        ok = self.execution.execute_async_v3(int(self.stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        check(driver.cuEventRecord(end, self.stream), "cuEventRecord(end)")
        check(driver.cuEventSynchronize(end), "cuEventSynchronize")
        elapsed = check(driver.cuEventElapsedTime(start, end), "cuEventElapsedTime")
        driver.cuEventDestroy(start)
        driver.cuEventDestroy(end)
        return float(elapsed)

    def download(self) -> float:
        started = time.perf_counter()
        for name, host in self.outputs.items():
            check(driver.cuMemcpyDtoH(host.ctypes.data, self.device_ptrs[name], host.nbytes), f"DtoH({name})")
        return (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        # Release TensorRT objects before destroying the CUDA context.
        for ptr in self.device_ptrs.values():
            driver.cuMemFree(ptr)
        driver.cuStreamDestroy(self.stream)
        self.execution = None
        self.engine = None
        self.runtime = None
        driver.cuCtxDestroy(self.context_handle)


def wait_for_cooldown(target_c: float, timeout_s: float) -> float:
    started = time.monotonic()
    while True:
        temperature = current_temp_c()
        if temperature <= target_c:
            return time.monotonic() - started
        if time.monotonic() - started >= timeout_s:
            raise RuntimeError(
                f"cooldown timed out at {temperature:.2f}C (target {target_c:.2f}C)"
            )
        time.sleep(5.0)


def run_one(engine: Engine, calibration: dict[str, Any], images: Path, cameras: tuple[str, ...]) -> dict[str, Any]:
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
        h2d_ms += engine.upload("img", value[None, ...])
        h2d_ms += engine.upload("k", np.asarray(info["K_640x360"], dtype=np.float32)[None, ...])
        h2d_ms += engine.upload("r", np.asarray(info["R_cam_to_ego"], dtype=np.float32)[None, ...])
        h2d_ms += engine.upload("t", np.asarray(info["t_cam_to_ego"], dtype=np.float32)[None, ...])
        forward_ms += engine.infer()
        d2h_ms += engine.download()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {
        "camera_count": len(cameras), "cameras": list(cameras),
        "wall_ms": wall_ms, "preprocess_ms": preprocess_ms,
        "h2d_ms": h2d_ms, "model_forward_gpu_ms": forward_ms,
        "output_copy_ms": d2h_ms, "routing_ms": routing_ms,
        "reuse_merge_ms": 0.0, "reuse_merge_scope": "not_applicable_current_only_engine",
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-temp-max", type=float, default=70.0)
    parser.add_argument("--abort-temp", type=float, default=90.0)
    parser.add_argument("--cooldown-target-c", type=float, default=70.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--power-mode", default="25W (nvpmodel mode 1)")
    parser.add_argument("--tegrastats-interval-ms", type=int, default=100)
    parser.add_argument("--camera-counts", type=int, nargs="+", default=[3, 4, 6])
    args = parser.parse_args()
    bundle = Path(args.bundle)
    engine_path = Path(args.engine)
    output = Path(args.output)
    if current_temp_c() > args.start_temp_max:
        raise RuntimeError(f"thermal gate: current max temperature {current_temp_c():.2f}C > {args.start_temp_max:.2f}C")
    calibration = json.loads((bundle / "calibration.json").read_text(encoding="utf-8"))
    engine = Engine(engine_path)
    free_before, total_mem = check(driver.cuMemGetInfo(), "cuMemGetInfo(before)")
    # A short initial pass also confirms every real JPEG and calibration entry.
    for camera in CAMERAS:
        info = calibration["channels"][camera]
        value, _ = preprocess(bundle / "images" / info["image"])
        engine.upload("img", value[None, ...])
        engine.upload("k", np.asarray(info["K_640x360"], dtype=np.float32)[None, ...])
        engine.upload("r", np.asarray(info["R_cam_to_ego"], dtype=np.float32)[None, ...])
        engine.upload("t", np.asarray(info["t_cam_to_ego"], dtype=np.float32)[None, ...])
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
                    raise RuntimeError(f"thermal gate before K{k}: {current_temp_c():.2f}C > {args.start_temp_max:.2f}C")
                for _ in range(args.warmup):
                    run_one(engine, calibration, bundle / "images", cameras)
                start = time.time()
                measured = [run_one(engine, calibration, bundle / "images", cameras) for _ in range(args.measured)]
                end = time.time()
                if current_temp_c() >= args.abort_temp:
                    raise RuntimeError(f"thermal abort after K{k}: {current_temp_c():.2f}C >= {args.abort_temp:.2f}C")
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
                    "reuse_merge_ms": 0.0, "reuse_merge_scope": "not_applicable_current_only_engine",
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

    payload = {
        "protocol": {
            "device_scope": f"Jetson Orin NX 16GB, {args.power_mode}, PWM255",
            "precision": "TensorRT FP16 engine; input/output bindings are FP32",
            "batch_size": 1, "camera_counts": list(requested),
            "warmup_per_run": args.warmup, "measured_per_run": args.measured,
            "repeats": min(args.repeats, 3), "orders": order[:min(args.repeats, 3)],
            "scope": "current-only single-camera TensorRT engine; no V31 previous-frame branch",
            "quality_metrics": "not re-evaluated on Jetson; inherit frozen validation quality only outside this benchmark",
            "parity": "TensorRT repeatability smoke only; PyTorch/reference output unavailable on this fresh device",
            "deadline_definition": "wall_ms > 500ms per processed frame",
            "energy_definition": "VDD_IN module-rail mean power × frame wall time; not sensor/vehicle energy",
        },
        "engine_sha256": sha256_file(engine_path), "engine_bytes": engine_path.stat().st_size,
        "calibration_sha256": sha256_file(bundle / "calibration.json"),
        "bundle": str(bundle), "rows": rows,
        "all_tegrastats_rows": stats_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "engine_sha256": payload["engine_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
