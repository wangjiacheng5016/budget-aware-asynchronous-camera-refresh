#!/usr/bin/env python3
"""CPU-side scheduler and sparse-memory overhead benchmark on real V31 caches.

This is deliberately separate from the current-only TensorRT engine timing.  It
replays the frozen K3 validation route and real per-camera detection records,
using the production motion propagation and exact spatial-index rotated NMS
semantics.  Timings therefore quantify scheduler plus reuse/merge overhead; they
must not be described as detector-forward latency.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import resource
except ImportError:  # Windows-side regression smoke; Jetson uses resource.
    resource = None


CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
DETECTION_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def yaw_from_quaternion(values: list[float]) -> float:
    w, x, y, z = (float(value) for value in values)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def propagate(record: dict, age_s: float) -> dict:
    output = dict(record)
    translation = [float(value) for value in record["translation"]]
    velocity = [float(value) for value in record.get("velocity", (0.0, 0.0))]
    translation[0] += velocity[0] * age_s
    translation[1] += velocity[1] * age_s
    output["translation"] = translation
    # Final protocol is motion-only with effectively no additional score decay.
    output["detection_score"] = float(record["detection_score"])
    return output


def to_tuple(record: dict, camera_index: int) -> tuple[float, ...]:
    x, y, z = (float(value) for value in record["translation"])
    w, length, height = (float(value) for value in record["size"])
    vx, vy = (float(value) for value in record.get("velocity", (0.0, 0.0))[:2])
    cls_id = DETECTION_NAMES.index(str(record["detection_name"]))
    return (
        x, y, z, w, length, height, yaw_from_quaternion(record["rotation"]),
        vx, vy, float(record["detection_score"]), cls_id, -1, camera_index,
    )


def cv_rotated_box(box: tuple[float, ...]):
    return (
        (float(box[0]), float(box[1])),
        (float(max(box[3], 1e-3)), float(max(box[4], 1e-3))),
        float(box[6] * 180.0 / math.pi),
    )


def rotated_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    area1 = max(float(first[3] * first[4]), 1e-6)
    area2 = max(float(second[3] * second[4]), 1e-6)
    try:
        _, points = cv2.rotatedRectangleIntersection(cv_rotated_box(first), cv_rotated_box(second))
    except Exception:
        return 0.0
    if points is None:
        return 0.0
    inter = abs(cv2.contourArea(cv2.convexHull(points, returnPoints=True)))
    return float(inter / max(area1 + area2 - inter, 1e-6))


def aabb(box: tuple[float, ...]) -> tuple[float, float, float, float]:
    points = cv2.boxPoints(cv_rotated_box(box))
    return (
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()),
    )


def cells(bounds: tuple[float, float, float, float], size: float = 5.0):
    min_x, min_y, max_x, max_y = bounds
    for gx in range(math.floor(min_x / size), math.floor(max_x / size) + 1):
        for gy in range(math.floor(min_y / size), math.floor(max_y / size) + 1):
            yield gx, gy


def nms_classwise(boxes: list[tuple[float, ...]], threshold: float = 0.05):
    kept: list[tuple[float, ...]] = []
    for cls_id in range(len(DETECTION_NAMES)):
        candidates = sorted(
            (box for box in boxes if int(box[10]) == cls_id),
            key=lambda box: box[9], reverse=True,
        )
        class_kept: list[tuple[float, ...]] = []
        grid: dict[tuple[int, int], list[int]] = {}
        for box in candidates:
            occupied = tuple(cells(aabb(box)))
            possible: set[int] = set()
            for cell in occupied:
                possible.update(grid.get(cell, ()))
            if any(rotated_iou(class_kept[index], box) >= threshold for index in sorted(possible)):
                continue
            index = len(class_kept)
            class_kept.append(box)
            for cell in occupied:
                grid.setdefault(cell, []).append(index)
        kept.extend(class_kept)
    return sorted(kept, key=lambda box: box[9], reverse=True)[:500]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": float(max(values)),
    }


def peak_rss_mib() -> float | None:
    if resource is None:
        return None
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def replay(payload: dict, repeats: int) -> dict:
    all_rows = []
    mismatch_count = 0
    for repeat in range(repeats):
        memory: dict[str, list[dict]] = {}
        ages: dict[str, float] = {}
        scene = None
        for row in payload["trace"]:
            if row["scene_token"] != scene:
                scene = row["scene_token"]
                memory.clear()
                ages.clear()
            token = row["sample_token"]
            expected = tuple(row["selected"])

            start = time.perf_counter_ns()
            unseen = [camera for camera in CAMERAS if camera not in memory]
            chosen = unseen[:3] if len(unseen) >= 3 else unseen + [
                camera for camera in sorted(CAMERAS, key=lambda name: (-ages.get(name, 0.0), CAMERAS.index(name)))
                if camera not in unseen
            ][:3 - len(unseen)]
            scheduler_ms = (time.perf_counter_ns() - start) / 1e6
            if set(chosen) != set(expected):
                mismatch_count += 1
                # The benchmark uses the frozen formal action after auditing the
                # independently recomputed oldest-first choice.
            chosen = list(expected)

            start = time.perf_counter_ns()
            sources: list[tuple[dict, int]] = []
            stale_records = 0
            current_records = 0
            for camera_index, camera in enumerate(CAMERAS):
                if camera in chosen:
                    records = payload["caches"][camera][token]
                    sources.extend((record, camera_index) for record in records)
                    current_records += len(records)
                elif camera in memory:
                    records = [propagate(record, ages[camera]) for record in memory[camera]]
                    sources.extend((record, camera_index) for record in records)
                    stale_records += len(records)
            propagation_ms = (time.perf_counter_ns() - start) / 1e6

            start = time.perf_counter_ns()
            tuples = [to_tuple(record, camera_index) for record, camera_index in sources]
            conversion_ms = (time.perf_counter_ns() - start) / 1e6

            start = time.perf_counter_ns()
            merged = nms_classwise(tuples)
            nms_ms = (time.perf_counter_ns() - start) / 1e6

            start = time.perf_counter_ns()
            for camera in chosen:
                memory[camera] = payload["caches"][camera][token]
                ages[camera] = 0.0
            interval = float(row.get("actual_next_interval_s", 0.5))
            for camera in list(ages):
                ages[camera] += interval
            update_ms = (time.perf_counter_ns() - start) / 1e6

            all_rows.append({
                "repeat": repeat + 1, "sample_token": token,
                "scheduler_ms": scheduler_ms, "propagation_ms": propagation_ms,
                "tuple_conversion_ms": conversion_ms, "nms_ms": nms_ms,
                "memory_update_ms": update_ms,
                "reuse_merge_ms": propagation_ms + conversion_ms + nms_ms + update_ms,
                "control_reuse_total_ms": scheduler_ms + propagation_ms + conversion_ms + nms_ms + update_ms,
                "current_input_records": current_records, "stale_input_records": stale_records,
                "merged_records": len(merged),
            })
    metric_names = (
        "scheduler_ms", "propagation_ms", "tuple_conversion_ms", "nms_ms",
        "memory_update_ms", "reuse_merge_ms", "control_reuse_total_ms",
    )
    return {
        "rows": all_rows,
        "summary": {name: summarize([float(row[name]) for row in all_rows]) for name in metric_names},
        "frozen_selection_mismatch_frames_across_repeats": mismatch_count,
        "peak_process_rss_mib": peak_rss_mib(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    with gzip.open(args.bundle, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    result = replay(payload, args.repeats)
    result["protocol"] = {
        "scope": "CPU-side final K3 scheduler plus motion-memory reuse and production exact rotated NMS",
        "frame_count": len(payload["trace"]), "repeats": args.repeats,
        "memory": "motion/no-extra-decay", "merge_iou": 0.05, "max_boxes": 500,
        "detector_forward_included": False,
        "bundle_sha256": sha256(args.bundle), "script_sha256": sha256(Path(__file__)),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "summary": result["summary"]}))


if __name__ == "__main__":
    main()
