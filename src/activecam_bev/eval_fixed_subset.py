#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .activecam_core import CAMERAS
    from .mini_preflight import evaluate_predictions, load_dependencies, ordered_split_frames
except ImportError:
    from activecam_core import CAMERAS
    from mini_preflight import evaluate_predictions, load_dependencies, ordered_split_frames


def parse_cameras(value: str) -> tuple[str, ...]:
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise argparse.ArgumentTypeError("at least one camera is required")
    if len(cameras) != len(set(cameras)):
        raise argparse.ArgumentTypeError("camera list contains duplicates")
    invalid = sorted(set(cameras).difference(CAMERAS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown cameras: {invalid}")
    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen fixed-camera subset.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cameras", type=parse_cameras, required=True)
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--selection-source", default="predeclared")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes = load_dependencies(
        Path(args.legacy_src)
    )
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=False)
    split_names = sorted(create_splits_scenes()[args.eval_set])
    ordered_frames = ordered_split_frames(nusc, split_names)
    expected = {token for token, _scene in ordered_frames}
    cache_dir = Path(args.cache_dir)
    selected_caches = {
        camera: legacy.read_cache(cache_dir / camera / "results_nusc.json")
        for camera in args.cameras
    }
    for camera, cache in selected_caches.items():
        missing = expected.difference(cache)
        if missing:
            raise RuntimeError(f"{camera} cache is missing {len(missing)} tokens")

    started = time.time()
    predictions = legacy.merge_subset(
        selected_caches, args.cameras, args.merge_iou, args.max_boxes
    )
    method = "fixed_" + "__".join(args.cameras)
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
        "policy": "fixed",
        "eval_set": args.eval_set,
        "frame_count": len(ordered_frames),
        "scene_count": len(split_names),
        "cameras": list(args.cameras),
        "camera_count_per_frame": len(args.cameras),
        "exact_camera_passes": len(ordered_frames) * len(args.cameras),
        "free_full6_warm_start": False,
        "selection_source": args.selection_source,
        "metrics": metrics,
        "elapsed_seconds_total": time.time() - started,
        "protocol": {
            "memory_mode": "none",
            "uses_current_unselected_detection_or_gt": False,
            "merge_iou": args.merge_iou,
            "max_boxes": args.max_boxes,
        },
    }
    (output_dir / "fixed_subset_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
