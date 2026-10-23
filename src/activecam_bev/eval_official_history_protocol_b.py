#!/usr/bin/env python3
"""Cache-level official-history Protocol B validation.

Selected cameras use the frozen temporal V31 prediction only when the raw
previous image is no more than 0.5 s old; otherwise they fall back to the
current-only prediction.  Unselected cameras may reuse only current-only
predictions whose same-camera raw-image age is no more than 0.5 s.  Thus the
number of past images is bounded by K + (6-K) = 6 without retraining.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

try:
    from .activecam_core import CAMERAS, propagate_detection
    from .mini_preflight import evaluate_predictions, load_dependencies, merge_sources, ordered_split_frames
    from .official_history_protocol import MemoryProvenance, audit_protocol_a_frame, camera_capture_timestamps_us
except ImportError:
    from activecam_core import CAMERAS, propagate_detection
    from mini_preflight import evaluate_predictions, load_dependencies, merge_sources, ordered_split_frames
    from official_history_protocol import MemoryProvenance, audit_protocol_a_frame, camera_capture_timestamps_us


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--current-cache-dir", required=True)
    p.add_argument("--current-cache-manifest", required=True)
    p.add_argument("--temporal-cache-dir", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--legacy-src", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--camera-count", required=True, type=int)
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--eval-set", default="val")
    return p.parse_args(argv)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _previous_capture_times(nusc, sample, cameras):
    previous = sample.get("prev")
    if not previous:
        return None
    return camera_capture_timestamps_us(nusc, previous, cameras)


def main():
    args = parse_args()
    if args.camera_count not in (3, 4, 6):
        raise ValueError("formal Protocol B matrix is K3/K4/K6")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    legacy, NuScenes, config_factory, NuScenesEval, splits = load_dependencies(Path(args.legacy_src))
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    frames = ordered_split_frames(nusc, sorted(splits()[args.eval_set]))
    tokens = [token for token, _scene in frames]
    token_set = set(tokens)

    current_root = Path(args.current_cache_dir).resolve()
    temporal_root = Path(args.temporal_cache_dir).resolve()
    current = {camera: legacy.read_cache(current_root / camera / "results_nusc.json") for camera in CAMERAS}
    temporal = {camera: legacy.read_cache(temporal_root / camera / "results_nusc.json") for camera in CAMERAS}
    for label, caches in (("current", current), ("temporal", temporal)):
        for camera, cache in caches.items():
            if set(cache) != token_set:
                raise RuntimeError(f"{label}/{camera}: token set mismatch")
    selections_path = Path(args.selections).resolve()
    selections = json.loads(selections_path.read_text(encoding="utf-8"))
    if set(selections) != token_set:
        raise RuntimeError("selection token set mismatch")

    active_scene = None
    memory = {}
    predictions = {}
    ancestry = []
    past_counts = []
    max_past_age = 0.0
    selected_temporal_events = 0
    selected_current_fallback_events = 0
    stale_memory_events = 0
    started = time.time()
    for token, scene_token in frames:
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {}
        sample = nusc.get("sample", token)
        sample_ts = int(sample["timestamp"])
        capture = camera_capture_timestamps_us(nusc, token, CAMERAS)
        previous_capture = _previous_capture_times(nusc, sample, CAMERAS)
        selected = tuple(selections[token])
        if len(selected) != args.camera_count or len(set(selected)) != args.camera_count:
            raise RuntimeError(f"{token}: selection is not exact K")
        external = audit_protocol_a_frame(
            cameras=CAMERAS,
            selected=selected,
            current_capture_timestamps_us=capture,
            memory={camera: value[0] for camera, value in memory.items()},
        )
        valid_memory = set(external.valid_memory_cameras)
        sources = []
        temporal_ages = []
        selected_set = set(selected)
        for camera in CAMERAS:
            if camera in selected_set:
                valid_previous = False
                previous_age = 0.0
                if previous_capture is not None:
                    previous_age = (capture[camera] - previous_capture[camera]) / 1_000_000.0
                    valid_previous = 0.0 <= previous_age <= 0.5
                if valid_previous:
                    records = temporal[camera][token]
                    temporal_ages.append(previous_age)
                    selected_temporal_events += 1
                else:
                    records = current[camera][token]
                    selected_current_fallback_events += 1
            elif camera in valid_memory:
                provenance, stored = memory[camera]
                motion_age = (sample_ts - provenance.source_sample_timestamp_us) / 1_000_000.0
                if motion_age < 0:
                    raise RuntimeError("future memory source")
                records = [
                    propagate_detection(record, age_s=motion_age, tau_s=1.0e9, use_velocity=True)
                    for record in stored
                ]
            else:
                records = []
            sources.append((camera, records))
        predictions[token] = merge_sources(legacy, token, sources, 0.05, 500)
        combined_ages = temporal_ages + [
            (capture[camera] - memory[camera][0].raw_image_timestamp_us) / 1_000_000.0
            for camera in external.valid_memory_cameras
        ]
        past_count = len(combined_ages)
        if past_count > 6 or any(age < 0.0 or age > 0.5 for age in combined_ages):
            raise RuntimeError(f"{token}: official history bound violated")
        past_counts.append(past_count)
        max_past_age = max(max_past_age, max(combined_ages, default=0.0))
        stale_memory_events += len(external.stale_memory_cameras)
        ancestry.append({
            "sample_token": token,
            "selected": selected,
            "selected_temporal_past_images": len(temporal_ages),
            "external_memory_past_images": len(external.valid_memory_cameras),
            "past_image_count": past_count,
            "max_past_image_age_s": max(combined_ages, default=0.0),
        })
        for camera in selected:
            memory[camera] = (
                MemoryProvenance(capture[camera], sample_ts),
                current[camera][token],
            )

    method = f"official_history_B_shared_current_memory_k{args.camera_count}"
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
        "protocol_name": "official_history_B_shared_current_memory",
        "cache_level_upper_bound": True,
        "requires_shared_current_only_head_seam_before_test": True,
        "method": method,
        "camera_count_per_frame": args.camera_count,
        "frame_count": len(tokens),
        "past_image_count_mean": statistics.fmean(past_counts),
        "past_image_count_max": max(past_counts, default=0),
        "max_past_image_age_s": max_past_age,
        "selected_temporal_events": selected_temporal_events,
        "selected_current_fallback_events": selected_current_fallback_events,
        "stale_memory_events_dropped": stale_memory_events,
        "history_window_violations": 0,
        "past_image_count_violations": 0,
        "unselected_current_reads": 0,
        "memory_source_previous_branch_valid_events": 0,
        "metrics": metrics,
        "current_cache_manifest_sha256": _sha(Path(args.current_cache_manifest)),
        "selections_sha256": _sha(selections_path),
        "elapsed_seconds": time.time() - started,
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (output / "ancestry_trace.json").write_text(json.dumps(ancestry), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
