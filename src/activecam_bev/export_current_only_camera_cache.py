#!/usr/bin/env python3
"""Export V31 detections without constructing or reading any previous image.

This is a deliberately small formal seam around the frozen V31 checkpoint.
It preserves the trained temporal architecture and weights, but calls its
existing current-only path.  The separate routing/evaluation step owns sparse
memory and the official 0.5-second raw-ancestry audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

try:
    from .mini_preflight import load_dependencies
    from .result_io import write_result_json
except ImportError:
    from mini_preflight import load_dependencies
    from result_io import write_result_json


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _velocity_interval_s(nusc, token: str, camera: str) -> float:
    """Metadata-only displacement interval; this never reads the old image."""

    sample = nusc.get("sample", token)
    if not sample.get("prev"):
        return 0.5
    previous = nusc.get("sample", sample["prev"])
    if camera not in previous["data"]:
        return 0.5
    current_sd = nusc.get("sample_data", sample["data"][camera])
    previous_sd = nusc.get("sample_data", previous["data"][camera])
    return max(
        1.0e-3,
        (int(current_sd["timestamp"]) - int(previous_sd["timestamp"]))
        / 1_000_000.0,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    legacy_src = Path(args.legacy_src).resolve()
    # The active benchmark module imports the legacy V31 builder at module
    # import time. Register both directories before importing it; doing this
    # at file import time would be too early because --legacy-src is a CLI
    # argument and caused the first remote smoke to fail before any inference.
    active_src = Path(__file__).resolve().parent
    sys.path.insert(0, str(active_src))
    sys.path.insert(0, str(legacy_src))
    from benchmark_activecam_end_to_end import decode_current_camera
    import benchmark_v31_jetson as builder

    training, base, model, checkpoint = builder.build_model(
        Path(args.checkpoint).resolve()
    )
    # Match the formal exporter's non-finite BatchNorm-buffer repair.
    training.prepare_model_for_eval(model, checkpoint)
    model.to(torch.device("cuda")).eval()
    _legacy, _Nusc, _cfg, _Eval, create_splits_scenes = load_dependencies(
        legacy_src
    )
    if args.camera not in base.CAM_CHANNELS:
        raise ValueError(f"unknown camera: {args.camera}")
    transform_teacher, transform_student = training.make_transforms(base)
    dataset = training.NuScenesOfficialDetectionDataset(
        base,
        args.data_root,
        args.version,
        args.eval_set,
        transform_teacher,
        transform_student,
        build_depth=False,
        need_teacher_inputs=False,
        need_student_inputs=True,
        need_teacher_depth=False,
        need_student_depth=False,
        need_temporal_inputs=False,
        need_history_targets=False,
        build_detection_targets=False,
        target_scope="student_fov",
        student_view_mode="front",
        student_camera="CAM_FRONT",
        student_cameras=list(base.CAM_CHANNELS),
    )
    split_scenes = set(create_splits_scenes()[args.eval_set])
    scene_tokens = {
        scene["token"]
        for scene in dataset.nusc.scene
        if scene["name"] in split_scenes
    }
    if args.max_scenes is not None:
        names = sorted(
            scene["name"]
            for scene in dataset.nusc.scene
            if scene["token"] in scene_tokens
        )[: args.max_scenes]
        scene_tokens = {
            scene["token"]
            for scene in dataset.nusc.scene
            if scene["name"] in set(names)
        }
    sample_indices = [
        index
        for index, token in enumerate(dataset.sample_tokens)
        if dataset.nusc.get("sample", token)["scene_token"] in scene_tokens
    ]
    camera_index = list(base.CAM_CHANNELS).index(args.camera)
    predictions = {}
    device = torch.device("cuda")
    with torch.inference_mode():
        for progress, sample_index in enumerate(sample_indices, 1):
            item = dataset[(sample_index, camera_index)]
            token = item[0]
            current_inputs = tuple(
                item[index].unsqueeze(0).to(device) for index in range(5, 9)
            )
            current_img, current_k, current_r, current_t = current_inputs
            prev_valid = torch.zeros(1, device=device)
            masked_temporal_inputs = (
                # Current raw image is reused as an in-memory masked placeholder;
                # no previous image path is constructed or read. This keeps the
                # trained temporal pre-encoder/fusion graph active while the
                # valid mask makes its temporal delta exactly zero.
                current_img,
                current_k,
                current_r,
                current_t,
                torch.eye(3, device=device).unsqueeze(0),
                torch.zeros(1, 3, device=device),
                prev_valid,
            )
            outputs = model.forward_outputs(*current_inputs, *masked_temporal_inputs)
            # The checkpoint predicts velocity as inter-keyframe displacement.
            # Decoding requires only metadata timestamps, not the old raw image.
            decode_item = list(item)
            decode_item[25] = torch.tensor(
                _velocity_interval_s(dataset.nusc, token, args.camera),
                dtype=torch.float32,
            )
            canonical = decode_current_camera(
                training,
                base,
                model,
                dataset.nusc,
                token,
                camera_index,
                decode_item,
                outputs,
            )
            canonical = training.nms_classwise(canonical, 0.05)
            canonical = sorted(canonical, key=lambda box: box[9], reverse=True)[:500]
            predictions[token] = [
                training.ego_box_to_nusc_result(
                    dataset.nusc, token, box, "CAM_FRONT"
                )
                for box in canonical
            ]
            if progress % 500 == 0:
                print(f"[PROGRESS] {progress}/{len(sample_indices)}", flush=True)
    result_path = output / "results_nusc.json"
    result_sha = write_result_json(result_path, predictions)
    protocol = {
        "protocol_name": "v31_current_only_official_history_A",
        "dataset_version": args.version,
        "eval_set": args.eval_set,
        "camera": args.camera,
        "sample_count": len(predictions),
        "checkpoint_sha256": _sha256(Path(args.checkpoint).resolve()),
        "exporter_source_sha256": _sha256(Path(__file__).resolve()),
        "training_source_sha256": _sha256(
            legacy_src / "train_official_nuscenes_nds_kd_v2.py"
        ),
        "model_builder_source_sha256": _sha256(
            legacy_src / "benchmark_v31_jetson.py"
        ),
        "result_sha256": result_sha,
        "raw_image_inputs_per_prediction": 1,
        "need_temporal_inputs": False,
        "previous_image_constructed": False,
        "previous_image_forwarded": False,
        "prev_valid_events": 0,
        "masked_temporal_placeholder": "same current tensor, identity pose, prev_valid=0",
        "unique_raw_image_ancestry_per_prediction": 1,
        "velocity_interval_source": "sample_data timestamps only; no previous image read",
        "frozen_weight_reuse_without_retraining": True,
        "warning": (
            "This current-only inference distribution was not the V31 training "
            "operating point; quality must be established on validation."
        ),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
