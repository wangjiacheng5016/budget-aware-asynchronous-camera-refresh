#!/usr/bin/env python3
"""End-to-end RTX benchmark for deadline-aware refresh with real nuScenes images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from activecam_core import CAMERAS, propagate_detection
from benchmark_v31_jetson import EXPECTED_PARAMETERS, build_model, percentile, sha256
from benchmark_v31_real_inputs import move_student_item
from benchmark_v31_rtx import parse_samples, sample_idle, start_nvidia_smi, stop_sampler
from mini_preflight import load_dependencies, merge_sources
from router_core import build_pre_action_state, enumerate_actions, feasible_actions


PARITY_ASSOCIATION_GATE_M = 0.05
DETECTION_SCORE_THRESHOLD = 0.05


def age_priority_index(actions, ages_s, valid_indices) -> int:
    def key(index):
        chosen = set(actions[index])
        next_ages = [
            0.0 if camera in chosen else age
            for camera, age in zip(CAMERAS, ages_s)
        ]
        return max(next_ages), sum(next_ages), index

    return min(valid_indices, key=key)


def compare_records(
    reference, actual, tolerance: float | None = None
) -> dict[str, float]:
    """Compare decoded per-camera records against the frozen FP32 export cache."""
    count_difference = abs(len(reference) - len(actual))
    translation_max_absolute_error = 0.0
    size_max_absolute_error = 0.0
    rotation_max_equivalent_error = 0.0
    rotation_max_angle_deg = 0.0
    velocity_max_absolute_error = 0.0
    score_max_absolute_error = 0.0
    class_count_l1 = 0
    attribute_mismatch_count = 0
    unmatched_reference = []
    unmatched_actual = []
    pairs = []
    classes = sorted(
        {str(record["detection_name"]) for record in reference}.union(
            str(record["detection_name"]) for record in actual
        )
    )
    for detection_name in classes:
        expected_group = [
            record for record in reference
            if str(record["detection_name"]) == detection_name
        ]
        observed_group = [
            record for record in actual
            if str(record["detection_name"]) == detection_name
        ]
        class_count_l1 += abs(len(expected_group) - len(observed_group))
        if not expected_group:
            unmatched_actual.extend(observed_group)
            continue
        if not observed_group:
            unmatched_reference.extend(expected_group)
            continue
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = np.zeros((len(expected_group), len(observed_group)), dtype=np.float64)
        for expected_index, expected in enumerate(expected_group):
            for observed_index, observed in enumerate(observed_group):
                center_distance = math.sqrt(sum(
                    (float(first) - float(second)) ** 2
                    for first, second in zip(
                        expected["translation"], observed["translation"]
                    )
                ))
                score_distance = abs(
                    float(expected["detection_score"])
                    - float(observed["detection_score"])
                )
                cost[expected_index, observed_index] = (
                    center_distance + 0.01 * score_distance
                )
        expected_indices, observed_indices = linear_sum_assignment(cost)
        matched_expected_indices = set()
        matched_observed_indices = set()
        for expected_index, observed_index in zip(
            expected_indices.tolist(), observed_indices.tolist()
        ):
            expected = expected_group[expected_index]
            observed = observed_group[observed_index]
            center_distance = math.sqrt(sum(
                (float(first) - float(second)) ** 2
                for first, second in zip(
                    expected["translation"], observed["translation"]
                )
            ))
            if center_distance <= PARITY_ASSOCIATION_GATE_M:
                pairs.append((expected, observed))
                matched_expected_indices.add(expected_index)
                matched_observed_indices.add(observed_index)
        unmatched_reference.extend(
            record for index, record in enumerate(expected_group)
            if index not in matched_expected_indices
        )
        unmatched_actual.extend(
            record for index, record in enumerate(observed_group)
            if index not in matched_observed_indices
        )
        if len(expected_indices) != min(len(expected_group), len(observed_group)):
            raise RuntimeError(f"decode parity matching failed for {detection_name}")

    for expected, observed in pairs:
        for name in ("sample_token", "detection_name"):
            if expected[name] != observed[name]:
                raise RuntimeError(f"decode parity discrete mismatch for {name}")
        attribute_mismatch_count += int(
            expected["attribute_name"] != observed["attribute_name"]
        )
        score_error = abs(
            float(expected["detection_score"]) - float(observed["detection_score"])
        )
        score_max_absolute_error = max(score_max_absolute_error, score_error)
        for name in ("translation", "size", "velocity"):
            if len(expected[name]) != len(observed[name]):
                raise RuntimeError(f"decode parity shape mismatch for {name}")
            error = max(
                abs(float(first) - float(second))
                for first, second in zip(expected[name], observed[name])
            )
            if name == "translation":
                translation_max_absolute_error = max(
                    translation_max_absolute_error, error
                )
            elif name == "size":
                size_max_absolute_error = max(size_max_absolute_error, error)
            else:
                velocity_max_absolute_error = max(
                    velocity_max_absolute_error, error
                )
        expected_rotation = [float(value) for value in expected["rotation"]]
        observed_rotation = [float(value) for value in observed["rotation"]]
        rotation_error = min(
            max(abs(first - second) for first, second in zip(expected_rotation, observed_rotation)),
            max(abs(first + second) for first, second in zip(expected_rotation, observed_rotation)),
        )
        rotation_max_equivalent_error = max(
            rotation_max_equivalent_error, rotation_error
        )
        expected_norm = math.sqrt(sum(value * value for value in expected_rotation))
        observed_norm = math.sqrt(sum(value * value for value in observed_rotation))
        if expected_norm <= 0.0 or observed_norm <= 0.0:
            raise RuntimeError("decode parity encountered a zero-norm quaternion")
        quaternion_dot = abs(sum(
            first * second
            for first, second in zip(expected_rotation, observed_rotation)
        ) / (expected_norm * observed_norm))
        quaternion_dot = min(1.0, max(-1.0, quaternion_dot))
        rotation_angle_deg = math.degrees(2.0 * math.acos(quaternion_dot))
        rotation_max_angle_deg = max(rotation_max_angle_deg, rotation_angle_deg)
    unmatched_score_threshold_distance = max(
        (
            abs(float(record["detection_score"]) - DETECTION_SCORE_THRESHOLD)
            for record in unmatched_reference + unmatched_actual
        ),
        default=0.0,
    )
    numeric_errors = {
        "translation": translation_max_absolute_error,
        "size": size_max_absolute_error,
        "rotation_equivalent": rotation_max_equivalent_error,
        "rotation_angle_deg": rotation_max_angle_deg,
        "velocity": velocity_max_absolute_error,
        "score": score_max_absolute_error,
        "count_difference": float(count_difference),
        "class_count_l1": float(class_count_l1),
        "attribute_mismatch_count": float(attribute_mismatch_count),
        "unmatched_reference_count": float(len(unmatched_reference)),
        "unmatched_actual_count": float(len(unmatched_actual)),
        "unmatched_score_threshold_distance": unmatched_score_threshold_distance,
    }
    continuous_errors = {
        name: numeric_errors[name]
        for name in (
            "translation", "size", "rotation_equivalent", "velocity", "score"
        )
    }
    if tolerance is not None and max(continuous_errors.values()) > tolerance:
        raise RuntimeError(
            f"decode parity numeric mismatch: errors={numeric_errors} "
            f"tolerance={tolerance:.3g}"
        )
    return {
        "box_count": float(len(actual)),
        "reference_box_count": float(len(reference)),
        "geometry_max_absolute_error": max(
            translation_max_absolute_error,
            size_max_absolute_error,
            rotation_max_equivalent_error,
            velocity_max_absolute_error,
        ),
        "score_max_absolute_error": score_max_absolute_error,
        "parity_association_gate_m": PARITY_ASSOCIATION_GATE_M,
        "detection_score_threshold": DETECTION_SCORE_THRESHOLD,
        **{f"{name}_max_absolute_error": value for name, value in numeric_errors.items()},
    }


def decode_current_camera(training, base, model, nusc, token, camera_index, item, outputs):
    hm, reg, depth, _bev, attr_logits, class_motion = outputs[:6]
    # Match the frozen formal export: suppress BEV cells outside the selected
    # camera's calibrated field of view before box decoding.
    bev_fov_mask = item[15].unsqueeze(0).to(device=hm.device, non_blocking=True)
    hm = hm.masked_fill(bev_fov_mask <= 0, -30.0)
    if model.azimuth_bin_head is not None or model.class_size_head is not None:
        azimuth_bins, class_size = outputs[6:8]
    else:
        azimuth_bins = class_size = None
    interval = item[25]
    interval_s = float(interval.item() if hasattr(interval, "item") else interval)
    boxes_batch = training.decode_official_boxes(
        hm.cpu(),
        reg.cpu(),
        base,
        0.05,
        500,
        0.2,
        attr_logits.cpu(),
        velocity_intervals=[interval_s],
        class_motion=class_motion.cpu() if class_motion is not None else None,
        azimuth_bins=azimuth_bins.cpu() if azimuth_bins is not None else None,
        class_size=class_size.cpu() if class_size is not None else None,
        k=item[6].unsqueeze(0),
        r_cam_to_ego=item[7].unsqueeze(0),
        t_cam_to_ego=item[8].unsqueeze(0),
        attribute_velocity_gate=False,
    )
    camera = base.CAM_CHANNELS[camera_index]
    return [
        training.transform_box_between_ego_frames(
            nusc,
            token,
            tuple(box) + (camera_index,),
            camera,
            "CAM_FRONT",
        )
        for box in boxes_batch[0]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-count", type=int, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=200)
    parser.add_argument("--planned-interval-s", type=float, default=0.5)
    parser.add_argument("--max-age-s", type=float, default=1.5)
    parser.add_argument("--power-interval-ms", type=int, default=50)
    parser.add_argument("--idle-settle-seconds", type=float, default=3.0)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--reference-cache-dir")
    parser.add_argument("--parity-frames", type=int, default=5)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--parity-envelope",
        help="Production-repeatability envelope JSON generated by build_decode_parity_envelope.py.",
    )
    parser.add_argument("--run-label", default="")
    args = parser.parse_args()
    if not 1 <= args.camera_count <= len(CAMERAS):
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
    # The formal V31 exporter repairs invalid BatchNorm running buffers after
    # loading. The generic hardware builder historically skipped this step.
    repaired_batchnorm_buffers = training.prepare_model_for_eval(model, checkpoint)
    model.to(torch.device("cuda")).eval()
    legacy, _NuScenes, _config_factory, _NuScenesEval, _splits = load_dependencies(
        Path(args.legacy_src)
    )
    reference_caches = None
    reference_manifest_sha256 = None
    parity_envelope = None
    parity_envelope_sha256 = None
    if args.parity_envelope:
        parity_envelope_path = Path(args.parity_envelope).resolve()
        parity_envelope = json.loads(parity_envelope_path.read_text(encoding="utf-8"))
        parity_envelope_sha256 = sha256(parity_envelope_path)
        if not math.isclose(
            float(parity_envelope["parity_association_gate_m"]),
            PARITY_ASSOCIATION_GATE_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("parity envelope association gate mismatch")
        if not math.isclose(
            float(parity_envelope["detection_score_threshold"]),
            DETECTION_SCORE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("parity envelope detection threshold mismatch")
    if args.reference_cache_dir:
        reference_root = Path(args.reference_cache_dir).resolve()
        reference_manifest = reference_root / "manifest.json"
        if not reference_manifest.is_file():
            raise FileNotFoundError(reference_manifest)
        reference_manifest_sha256 = sha256(reference_manifest)
        reference_caches = {
            camera: legacy.read_cache(reference_root / camera / "results_nusc.json")
            for camera in CAMERAS
        }
    if args.precision != "fp32" and reference_caches is not None:
        raise ValueError("frozen cache parity is defined for the formal FP32 path only")
    transform_teacher, transform_student = training.make_transforms(base)
    dataset = training.NuScenesOfficialDetectionDataset(
        base,
        args.data_root,
        "v1.0-trainval",
        "val",
        transform_teacher,
        transform_student,
        max_samples=args.warmup_frames + args.measured_frames,
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
    nusc = dataset.nusc
    device = torch.device("cuda")
    actions = enumerate_actions(args.camera_count)
    state = {"scene": None, "memory": {}, "previous_action": ()}
    parity_rows = []
    actual_deadline_violation_frames = 0
    actual_deadline_violating_camera_events = 0

    def run_frame(sample_index: int, *, audit_parity: bool = False) -> dict[str, float]:
        nonlocal actual_deadline_violation_frames
        nonlocal actual_deadline_violating_camera_events
        token = dataset.sample_tokens[sample_index]
        sample = nusc.get("sample", token)
        scene_token = sample["scene_token"]
        timestamp = int(sample["timestamp"])
        if scene_token != state["scene"]:
            state["scene"] = scene_token
            state["memory"] = {}
            state["previous_action"] = ()
        wall_started = time.perf_counter()
        route_started = time.perf_counter()
        payload = build_pre_action_state(
            memory=state["memory"],
            decision_timestamp=timestamp,
            previous_action=state["previous_action"],
            max_age_s=args.max_age_s,
            uncertainty_tau_s=1.0,
            strict_past=True,
        )
        initialized = [value >= 0 for value in payload["source_timestamps"]]
        feasible = feasible_actions(
            actions=actions,
            ages_s=payload["ages_s"],
            planned_interval_s=args.planned_interval_s,
            max_age_s=args.max_age_s,
            initialized=initialized,
        )
        valid = [index for index, allowed in enumerate(feasible) if allowed]
        emergency = not valid
        if emergency:
            valid = list(range(len(actions)))
        selected = actions[age_priority_index(actions, payload["ages_s"], valid)]
        route_ms = (time.perf_counter() - route_started) * 1000.0

        data_ms = 0.0
        gpu_ms = 0.0
        decode_ms = 0.0
        current_records = {}
        for camera in selected:
            camera_index = CAMERAS.index(camera)
            data_started = time.perf_counter()
            item = dataset[(sample_index, camera_index)]
            inputs = move_student_item(item, device)
            data_ms += (time.perf_counter() - data_started) * 1000.0
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            with torch.autocast(
                "cuda", enabled=args.precision == "fp16", dtype=torch.float16
            ):
                outputs = model.forward_outputs(*inputs)
            ended.record()
            ended.synchronize()
            gpu_ms += float(started.elapsed_time(ended))
            decode_started = time.perf_counter()
            canonical_boxes = decode_current_camera(
                training, base, model, nusc, token, camera_index, item, outputs
            )
            # Match production export exactly: run the per-camera merge NMS in
            # canonical CAM_FRONT ego tuples and convert to global JSON once.
            canonical_boxes = training.nms_classwise(canonical_boxes, 0.05)
            canonical_boxes = sorted(
                canonical_boxes, key=lambda box: box[9], reverse=True
            )[:500]
            current_records[camera] = [
                training.ego_box_to_nusc_result(
                    nusc, token, box, "CAM_FRONT"
                )
                for box in canonical_boxes
            ]
            if audit_parity:
                if reference_caches is None:
                    raise RuntimeError("parity audit requested without reference caches")
                row = compare_records(
                    reference_caches[camera][token],
                    current_records[camera],
                    None if parity_envelope is not None else args.parity_tolerance,
                )
                if parity_envelope is not None:
                    thresholds = parity_envelope["thresholds"]
                    errors = {
                        name: row[f"{name}_max_absolute_error"]
                        for name in parity_envelope["thresholds"]
                    }
                    exceeded = {
                        name: value
                        for name, value in errors.items()
                        if value > float(thresholds[name])
                    }
                    if exceeded:
                        raise RuntimeError(
                            "decode parity exceeds production repeatability envelope: "
                            f"camera={camera} sample_token={token} "
                            f"exceeded={exceeded} row={row}"
                        )
                row.update({"sample_token": token, "camera": camera})
                parity_rows.append(row)
            decode_ms += (time.perf_counter() - decode_started) * 1000.0

        memory_started = time.perf_counter()
        sources = []
        for camera in CAMERAS:
            if camera in current_records:
                records = current_records[camera]
            elif camera in state["memory"]:
                source_timestamp, stored = state["memory"][camera]
                age_s = max(0.0, (timestamp - source_timestamp) / 1_000_000.0)
                records = [
                    propagate_detection(
                        record,
                        age_s=age_s,
                        tau_s=1.0e9,
                        use_velocity=True,
                    )
                    for record in stored
                ]
            else:
                records = []
            sources.append((camera, records))
        merged = merge_sources(legacy, token, sources, 0.05, 500)
        memory_ms = (time.perf_counter() - memory_started) * 1000.0
        for camera, records in current_records.items():
            state["memory"][camera] = (timestamp, records)
        state["previous_action"] = selected
        next_interval_s = 0.0
        if sample["next"]:
            next_timestamp = int(nusc.get("sample", sample["next"])["timestamp"])
            next_interval_s = max(0.0, (next_timestamp - timestamp) / 1_000_000.0)
        violation_events = sum(
            1
            for camera, age_s, source_timestamp in zip(
                CAMERAS, payload["ages_s"], payload["source_timestamps"]
            )
            if source_timestamp >= 0
            and camera not in selected
            and age_s + next_interval_s > args.max_age_s + 0.01
        )
        actual_deadline_violation_frames += int(violation_events > 0)
        actual_deadline_violating_camera_events += violation_events
        if len(merged) > 500:
            raise RuntimeError("merge exceeded the official 500-box limit")
        return {
            "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
            "routing_ms": route_ms,
            "decode_preprocess_h2d_ms": data_ms,
            "gpu_forward_ms": gpu_ms,
            "box_decode_transform_ms": decode_ms,
            "memory_propagate_merge_ms": memory_ms,
            "prediction_count": float(len(merged)),
            "emergency": float(emergency),
        }

    with torch.inference_mode():
        for sample_index in range(args.warmup_frames):
            run_frame(
                sample_index,
                audit_parity=(
                    reference_caches is not None and sample_index < args.parity_frames
                ),
            )
        torch.cuda.synchronize()
        time.sleep(args.idle_settle_seconds)
        idle = sample_idle(output_dir, args.power_interval_ms, args.idle_seconds)
        torch.cuda.reset_peak_memory_stats()
        power_path = output_dir / "nvidia_smi_active.csv"
        power_handle = start_nvidia_smi(power_path, args.power_interval_ms)
        rows = []
        measured_started = time.perf_counter()
        try:
            for offset in range(args.measured_frames):
                rows.append(run_frame(args.warmup_frames + offset))
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
        else active_power * measured_wall_seconds / args.measured_frames
    )
    net_energy = (
        None
        if active_power is None or idle_power is None
        else max(0.0, active_power - idle_power)
        * measured_wall_seconds
        / args.measured_frames
    )
    fields = list(rows[0])
    with (output_dir / "end_to_end_latencies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in rows]

    wall = values("wall_ms")
    measured_tokens = dataset.sample_tokens[
        args.warmup_frames : args.warmup_frames + args.measured_frames
    ]
    token_manifest_sha256 = hashlib.sha256(
        "\n".join(measured_tokens).encode("utf-8")
    ).hexdigest()
    script_sha256 = sha256(Path(__file__).resolve())
    config_payload = {
        "camera_count": args.camera_count,
        "precision": args.precision,
        "warmup_frames": args.warmup_frames,
        "measured_frames": args.measured_frames,
        "planned_interval_s": args.planned_interval_s,
        "max_age_s": args.max_age_s,
        "score_threshold": 0.05,
        "per_view_nms_iou": 0.2,
        "merge_iou": 0.05,
        "max_boxes": 500,
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    builder_source = Path(inspect.getsourcefile(build_model) or "").resolve()
    training_source = Path(training.__file__).resolve()
    legacy_source = Path(legacy.__file__).resolve()
    legacy_bev_source = Path(legacy.bev.__file__).resolve()
    try:
        nvidia_smi_snapshot = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,power.limit,clocks.current.graphics,clocks.current.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        nvidia_smi_snapshot = None
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "real nuScenes current+previous image read, preprocessing, H2D, temporal "
            "forward, official box decode, global transform, deadline-aware routing, "
            "constant-velocity sparse memory propagation, and cross-view NMS merge"
        ),
        "camera_count": args.camera_count,
        "run_label": args.run_label,
        "precision": args.precision,
        "warmup_frames": args.warmup_frames,
        "measured_frames": args.measured_frames,
        "latency_wall_mean_ms": statistics.fmean(wall),
        "latency_wall_p50_ms": percentile(wall, 0.50),
        "latency_wall_p95_ms": percentile(wall, 0.95),
        "routing_mean_ms": statistics.fmean(values("routing_ms")),
        "routing_p95_ms": percentile(values("routing_ms"), 0.95),
        "decode_preprocess_h2d_mean_ms": statistics.fmean(
            values("decode_preprocess_h2d_ms")
        ),
        "decode_preprocess_h2d_p95_ms": percentile(
            values("decode_preprocess_h2d_ms"), 0.95
        ),
        "gpu_forward_mean_ms": statistics.fmean(values("gpu_forward_ms")),
        "gpu_forward_p95_ms": percentile(values("gpu_forward_ms"), 0.95),
        "box_decode_transform_mean_ms": statistics.fmean(
            values("box_decode_transform_ms")
        ),
        "box_decode_transform_p95_ms": percentile(
            values("box_decode_transform_ms"), 0.95
        ),
        "memory_propagate_merge_mean_ms": statistics.fmean(
            values("memory_propagate_merge_ms")
        ),
        "memory_propagate_merge_p95_ms": percentile(
            values("memory_propagate_merge_ms"), 0.95
        ),
        "prediction_count_mean": statistics.fmean(values("prediction_count")),
        "emergency_count": int(sum(values("emergency"))),
        "actual_deadline_violation_frames_including_warmup": actual_deadline_violation_frames,
        "actual_deadline_violating_camera_events_including_warmup": (
            actual_deadline_violating_camera_events
        ),
        "measured_wall_seconds": measured_wall_seconds,
        "throughput_frames_per_second": 1000.0 / statistics.fmean(wall),
        "peak_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "idle_power": idle,
        "active_power": active,
        "gross_energy_per_frame_j": gross_energy,
        "idle_subtracted_energy_per_frame_j": net_energy,
        "parameter_count": EXPECTED_PARAMETERS,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "repaired_batchnorm_buffers": repaired_batchnorm_buffers,
        "benchmark_script_sha256": script_sha256,
        "source_provenance": {
            "model_builder_path": str(builder_source),
            "model_builder_sha256": sha256(builder_source),
            "training_module_path": str(training_source),
            "training_module_sha256": sha256(training_source),
            "legacy_evaluator_path": str(legacy_source),
            "legacy_evaluator_sha256": sha256(legacy_source),
            "legacy_bev_path": str(legacy_bev_source),
            "legacy_bev_sha256": sha256(legacy_bev_source),
        },
        "config_sha256": config_sha256,
        "config": config_payload,
        "measured_token_manifest_sha256": token_manifest_sha256,
        "measured_first_token": measured_tokens[0],
        "measured_last_token": measured_tokens[-1],
        "measured_scene_count": len(
            {nusc.get("sample", token)["scene_token"] for token in measured_tokens}
        ),
        "decode_parity": {
            "enabled": reference_caches is not None,
            "audited_camera_passes": len(parity_rows),
            "audited_boxes": int(sum(row["box_count"] for row in parity_rows)),
            "geometry_max_absolute_error": max(
                (row["geometry_max_absolute_error"] for row in parity_rows), default=None
            ),
            "score_max_absolute_error": max(
                (row["score_max_absolute_error"] for row in parity_rows), default=None
            ),
            "observed_maxima": {
                name: max(
                    (row[f"{name}_max_absolute_error"] for row in parity_rows),
                    default=None,
                )
                for name in (
                    "translation", "size", "rotation_angle_deg", "velocity", "score",
                    "count_difference", "class_count_l1", "attribute_mismatch_count",
                    "unmatched_reference_count", "unmatched_actual_count",
                    "unmatched_score_threshold_distance",
                )
            },
            "tolerance": args.parity_tolerance,
            "production_repeatability_envelope_sha256": parity_envelope_sha256,
            "production_repeatability_thresholds": (
                parity_envelope["thresholds"] if parity_envelope else None
            ),
            "reference_cache_manifest_sha256": reference_manifest_sha256,
        },
        "hardware": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi_snapshot": nvidia_smi_snapshot,
        "protocol": {
            "strict_empty_memory_at_scene_start": True,
            "selection": "deadline-feasible oldest-first",
            "memory": "global-frame constant-velocity propagation without score decay",
            "max_age_s": args.max_age_s,
            "planned_interval_s": args.planned_interval_s,
            "deadline_audit": "ex-post actual next timestamp; not used by the scheduler",
            "unselected_current_images_read": False,
        },
    }
    (output_dir / "activecam_end_to_end_benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
