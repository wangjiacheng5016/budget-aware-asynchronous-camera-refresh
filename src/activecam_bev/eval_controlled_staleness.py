#!/usr/bin/env python3
"""Evaluate a controlled historical-age diagnostic on frozen camera caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .activecam_core import CAMERAS, propagate_detection
    from .mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        merge_sources,
        ordered_split_frames,
    )
except ImportError:  # Direct execution on the experiment server.
    from activecam_core import CAMERAS, propagate_detection
    from mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        merge_sources,
        ordered_split_frames,
    )


METRIC_NAMES = ("NDS", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE")


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(
        len(ordered) - 1,
        max(0, round((len(ordered) - 1) * quantile)),
    )
    return ordered[index]


def select_controlled_histories(
    *,
    ordered_frames: Sequence[tuple[str, str]],
    timestamps: Mapping[str, int],
    target_age_s: float,
    tolerance_s: float,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any]]:
    """Assign the closest strictly-past token within each scene."""
    if target_age_s <= 0.0 or tolerance_s < 0.0:
        raise ValueError("target age must be positive and tolerance non-negative")
    scenes: dict[str, list[str]] = defaultdict(list)
    for token, scene_token in ordered_frames:
        if token not in timestamps:
            raise KeyError(f"missing timestamp for {token}")
        scenes[str(scene_token)].append(str(token))
    assignments: dict[str, dict[str, Any] | None] = {}
    matched_ages: list[float] = []
    absolute_errors: list[float] = []
    for scene_tokens in scenes.values():
        previous: list[str] = []
        for token in scene_tokens:
            current_timestamp = int(timestamps[token])
            candidates = []
            for history_token in previous:
                history_timestamp = int(timestamps[history_token])
                age_s = (current_timestamp - history_timestamp) / 1_000_000.0
                if age_s <= 0.0:
                    continue
                error_s = abs(age_s - target_age_s)
                candidates.append((error_s, -history_timestamp, history_token, age_s))
            if candidates:
                error_s, _negative_timestamp, history_token, age_s = min(candidates)
            else:
                error_s = math.inf
                history_token = ""
                age_s = math.nan
            if error_s <= tolerance_s:
                assignments[token] = {
                    "history_token": history_token,
                    "actual_age_s": age_s,
                    "absolute_error_s": error_s,
                }
                matched_ages.append(age_s)
                absolute_errors.append(error_s)
            else:
                assignments[token] = None
            previous.append(token)
    frame_count = len(ordered_frames)
    matched_frames = len(matched_ages)
    return assignments, {
        "target_age_s": target_age_s,
        "tolerance_s": tolerance_s,
        "frame_count": frame_count,
        "matched_frames": matched_frames,
        "skipped_frames": frame_count - matched_frames,
        "coverage_fraction": matched_frames / frame_count if frame_count else 0.0,
        "actual_age_mean_s": (
            statistics.fmean(matched_ages) if matched_ages else None
        ),
        "actual_age_p95_s": _percentile(matched_ages, 0.95),
        "actual_age_min_s": min(matched_ages, default=None),
        "actual_age_max_s": max(matched_ages, default=None),
        "absolute_error_mean_s": (
            statistics.fmean(absolute_errors) if absolute_errors else None
        ),
        "absolute_error_p95_s": _percentile(absolute_errors, 0.95),
    }


def build_controlled_predictions(
    *,
    legacy: Any,
    caches: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    assignments: Mapping[str, Mapping[str, Any] | None],
    ordered_frames: Sequence[tuple[str, str]],
    mode: str,
    merge_iou: float,
    max_boxes: int,
) -> dict[str, list[dict[str, Any]]]:
    """Build full-token predictions; unmatched frames intentionally stay empty."""
    if mode not in {"static", "motion"}:
        raise ValueError("mode must be static or motion")
    predictions: dict[str, list[dict[str, Any]]] = {}
    for token, _scene_token in ordered_frames:
        assignment = assignments[token]
        if assignment is None:
            predictions[token] = []
            continue
        history_token = str(assignment["history_token"])
        age_s = float(assignment["actual_age_s"])
        sources = []
        for camera in CAMERAS:
            if history_token not in caches[camera]:
                raise RuntimeError(
                    f"{camera} cache is missing historical token {history_token}"
                )
            records = [
                propagate_detection(
                    record,
                    age_s=age_s,
                    tau_s=1.0e9,
                    use_velocity=mode == "motion",
                )
                for record in caches[camera][history_token]
            ]
            sources.append((camera, records))
        predictions[token] = merge_sources(
            legacy, token, sources, merge_iou, max_boxes
        )
    return predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled-staleness diagnostic using the nearest strictly-past "
            "same-scene cache token. This is not a scheduler or deadline run."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-age-s", type=float, required=True)
    parser.add_argument("--tolerance-s", type=float, default=0.15)
    parser.add_argument("--mode", choices=("static", "motion"), required=True)
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--skip-official-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    manifest_path = Path(args.cache_manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != "v1.0-trainval"
        or manifest.get("split") != "val"
        or manifest.get("expected_samples") != 6019
        or len(manifest.get("cameras", {})) != len(CAMERAS)
    ):
        raise RuntimeError("cache manifest is not a complete formal validation cache")

    legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes = (
        load_dependencies(Path(args.legacy_src))
    )
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=args.data_root, verbose=False
    )
    split_names = sorted(create_splits_scenes()[args.eval_set])
    ordered_frames = ordered_split_frames(nusc, split_names)
    if len(ordered_frames) != 6019:
        raise RuntimeError(f"expected 6019 validation frames, found {len(ordered_frames)}")
    expected = {token for token, _scene in ordered_frames}
    timestamps = {
        token: int(nusc.get("sample", token)["timestamp"])
        for token, _scene in ordered_frames
    }
    assignments, age_diagnostics = select_controlled_histories(
        ordered_frames=ordered_frames,
        timestamps=timestamps,
        target_age_s=args.target_age_s,
        tolerance_s=args.tolerance_s,
    )
    caches = {
        camera: legacy.read_cache(cache_dir / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    for camera, cache in caches.items():
        missing = expected.difference(cache)
        if missing:
            raise RuntimeError(f"{camera} cache is missing {len(missing)} val tokens")

    started = time.time()
    predictions = build_controlled_predictions(
        legacy=legacy,
        caches=caches,
        assignments=assignments,
        ordered_frames=ordered_frames,
        mode=args.mode,
        merge_iou=args.merge_iou,
        max_boxes=args.max_boxes,
    )
    if set(predictions) != expected:
        raise RuntimeError("controlled predictions do not cover the full val token set")
    target_label = str(args.target_age_s).replace(".", "p")
    method = f"controlled_staleness_age{target_label}_{args.mode}"
    metrics = None
    if not args.skip_official_eval:
        metrics = evaluate_predictions(
            method=method,
            predictions=predictions,
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output_dir,
        )
    diagnostics = {
        "method": method,
        "protocol_name": "controlled_staleness",
        "diagnostic_only": True,
        "not_a_scheduler_or_deadline_operating_point": True,
        "eval_set": args.eval_set,
        "frame_count": len(ordered_frames),
        "scene_count": len(split_names),
        "mode": args.mode,
        "age": age_diagnostics,
        "metrics": metrics,
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": _sha256(manifest_path),
        "elapsed_seconds_total": time.time() - started,
        "protocol": {
            "history_selection": (
                "closest strictly-past token within the same scene; empty "
                "prediction when no token is within tolerance"
            ),
            "uses_current_token_cache": False,
            "uses_cross_scene_history": False,
            "camera_count_from_history": len(CAMERAS),
            "memory_mode": args.mode,
            "motion_model": (
                "global-frame constant velocity without practical score decay"
                if args.mode == "motion"
                else "global-frame position held static without practical score decay"
            ),
            "merge": "production global tuple conversion, classwise BEV NMS, top-500",
            "merge_iou": args.merge_iou,
            "max_boxes": args.max_boxes,
        },
    }
    (output_dir / "controlled_staleness_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output_dir / "history_assignments.json").write_text(
        json.dumps(assignments, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
