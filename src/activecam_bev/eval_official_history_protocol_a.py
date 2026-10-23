#!/usr/bin/env python3
"""Official val replay for strict current-only ActiveCam protocol A."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from .activecam_core import CAMERAS, propagate_detection
    from .mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        merge_sources,
        ordered_split_frames,
    )
    from .official_history_protocol import (
        MemoryProvenance,
        build_protocol_a_predictions,
        camera_capture_timestamps_us,
    )
    from .router_core import (
        build_pre_action_state,
        enumerate_actions,
        feasible_actions,
    )
except ImportError:
    from activecam_core import CAMERAS, propagate_detection
    from mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        merge_sources,
        ordered_split_frames,
    )
    from official_history_protocol import (
        MemoryProvenance,
        build_protocol_a_predictions,
        camera_capture_timestamps_us,
    )
    from router_core import (
        build_pre_action_state,
        enumerate_actions,
        feasible_actions,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-count", required=True, type=int)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--skip-official-eval", action="store_true")
    return parser.parse_args(argv)


def _oldest_index(actions, ages, valid):
    def key(index):
        chosen = set(actions[index])
        next_ages = [
            0.0 if camera in chosen else age
            for camera, age in zip(CAMERAS, ages)
        ]
        return max(next_ages), sum(next_ages), index

    return min(valid, key=key)


def _validate_manifest(path: Path, expected_samples: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_name") != "v31_current_only_official_history_A":
        raise RuntimeError("cache manifest is not current-only protocol A")
    if payload.get("need_temporal_inputs") is not False:
        raise RuntimeError("cache manifest requested temporal inputs")
    if payload.get("previous_image_constructed") is not False:
        raise RuntimeError("cache manifest constructed previous images")
    if payload.get("previous_image_forwarded") is not False:
        raise RuntimeError("cache manifest forwarded previous images")
    if payload.get("prev_valid_events") != 0:
        raise RuntimeError("cache manifest has prev_valid events")
    if payload.get("expected_samples") != expected_samples:
        raise RuntimeError("cache manifest sample count mismatch")
    if set(payload.get("cameras", {})) != set(CAMERAS):
        raise RuntimeError("cache manifest must contain all six cameras")
    return payload


def main() -> None:
    args = parse_args()
    if args.camera_count not in (3, 4, 6):
        raise ValueError("formal minimal matrix is K3/K4/K6")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    legacy, NuScenes, config_factory, NuScenesEval, splits = load_dependencies(
        Path(args.legacy_src)
    )
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    scene_names = sorted(splits()[args.eval_set])
    if args.max_scenes is not None:
        scene_names = scene_names[: args.max_scenes]
    ordered_frames = ordered_split_frames(nusc, scene_names)
    tokens = [token for token, _scene in ordered_frames]
    manifest_path = Path(args.cache_manifest).resolve()
    manifest = _validate_manifest(manifest_path, len(tokens))
    caches = {
        camera: legacy.read_cache(
            Path(args.cache_dir) / camera / "results_nusc.json"
        )
        for camera in CAMERAS
    }
    for camera, cache in caches.items():
        if set(cache) != set(tokens):
            raise RuntimeError(f"{camera}: cache token mismatch")

    actions = enumerate_actions(args.camera_count)
    active_scene = None
    memory = {}
    routing_memory = {}
    predictions = {}
    selections = {}
    trace = []
    violation_frames = 0
    count_violation_frames = 0
    stale_drops = 0
    past_counts = []
    max_age = 0.0
    started = time.time()
    for token, scene_token in ordered_frames:
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {}
            routing_memory = {}
        sample = nusc.get("sample", token)
        sample_timestamp = int(sample["timestamp"])
        capture_times = camera_capture_timestamps_us(nusc, token, CAMERAS)
        state = build_pre_action_state(
            memory=routing_memory,
            decision_timestamp=sample_timestamp,
            previous_action=(),
            max_age_s=0.5,
            uncertainty_tau_s=1.0,
            strict_past=True,
        )
        initialized = [value >= 0 for value in state["source_timestamps"]]
        feasible = feasible_actions(
            actions=actions,
            ages_s=state["ages_s"],
            planned_interval_s=0.5,
            max_age_s=0.5,
            initialized=initialized,
        )
        valid = [index for index, allowed in enumerate(feasible) if allowed]
        # A nominal sample-time shield is insufficient for official ancestry;
        # selection remains oldest-first but raw eligibility is audited below.
        if not valid:
            valid = list(range(len(actions)))
        selected = actions[_oldest_index(actions, state["ages_s"], valid)]
        frame_predictions, audit = build_protocol_a_predictions(
            token=token,
            current_sample_timestamp_us=sample_timestamp,
            cameras=CAMERAS,
            selected=selected,
            current_capture_timestamps_us=capture_times,
            current_only_caches=caches,
            memory=memory,
            propagate=lambda records, age: [
                propagate_detection(
                    record, age_s=age, tau_s=1.0e9, use_velocity=True
                )
                for record in records
            ],
            merge=lambda sources: merge_sources(legacy, token, sources, 0.05, 500),
        )
        predictions[token] = frame_predictions
        selections[token] = list(selected)
        stale_drops += len(audit.stale_memory_cameras)
        past_counts.append(audit.past_image_count)
        max_age = max(max_age, audit.max_past_image_age_s)
        violation_frames += int(audit.history_window_violations > 0)
        count_violation_frames += int(audit.past_image_count_violations > 0)
        trace.append({"sample_token": token, **audit.to_dict()})
        for camera in selected:
            provenance = MemoryProvenance(
                raw_image_timestamp_us=capture_times[camera],
                source_sample_timestamp_us=sample_timestamp,
            )
            memory[camera] = (provenance, caches[camera][token])
            routing_memory[camera] = (sample_timestamp, caches[camera][token])

    method = f"official_history_A_current_only_k{args.camera_count}"
    metrics = None
    if not args.skip_official_eval:
        metrics = evaluate_predictions(
            method=method,
            predictions=predictions,
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output,
        )
    diagnostics = {
        "protocol_name": "official_history_A_current_only",
        "method": method,
        "camera_count_per_frame": args.camera_count,
        "frame_count": len(tokens),
        "raw_current_images_per_frame": args.camera_count,
        "past_image_count_mean": statistics.fmean(past_counts),
        "past_image_count_max": max(past_counts, default=0),
        "max_past_image_age_s": max_age,
        "stale_memory_camera_events_dropped": stale_drops,
        "history_window_violations": violation_frames,
        "past_image_count_violations": count_violation_frames,
        "unselected_current_reads": 0,
        "previous_branch_valid_events": 0,
        "external_memory_rule": (
            "current-only source detection; raw age by same-camera sample_data "
            "timestamp <=0.5s; strict drop when >0.5s"
        ),
        "motion_propagation_timebase": "source/current sample timestamp",
        "cache_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "cache_manifest": manifest,
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output / "selections.json").write_text(
        json.dumps(selections), encoding="utf-8"
    )
    (output / "ancestry_trace.json").write_text(
        json.dumps(trace), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
