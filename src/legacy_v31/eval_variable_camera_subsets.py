#!/usr/bin/env python3
"""Evaluate every non-empty subset of the six nuScenes cameras from caches.

The six input JSON files are produced by the same shared single-view checkpoint,
one camera at a time.  This script recombines those predictions with the exact
class-wise rotated NMS used by the main evaluator, so the network is evaluated
only six times rather than once for every one of the 63 camera subsets.

Two protocols are emitted:
  * official_360: unmodified nuScenes val detection protocol.
  * selected_fov: diagnostic protocol where GT and predictions are restricted
    to the union of the selected camera fields of view.  This is deliberately
    labelled non-official and must never be mixed with official NDS/mAP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from pyquaternion import Quaternion

import train_official_nuscenes_nds_kd_v2 as bev


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cache(path: Path) -> Dict[str, List[Dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"meta", "results"}:
        raise ValueError(f"unexpected nuScenes result payload: {path}")
    return payload["results"]


def result_to_global_tuple(record: Dict, camera_index: int) -> Tuple[float, ...]:
    quat = Quaternion(record["rotation"])
    yaw = float(quat.yaw_pitch_roll[0])
    cls_id = bev.DETECTION_NAMES.index(record["detection_name"])
    try:
        attr_id = bev.ATTRIBUTE_NAMES.index(record.get("attribute_name", ""))
    except ValueError:
        attr_id = -1
    x, y, z = (float(value) for value in record["translation"])
    w, length, height = (float(value) for value in record["size"])
    vx, vy = (float(value) for value in record["velocity"][:2])
    return (
        x, y, z, w, length, height, yaw, vx, vy,
        float(record["detection_score"]), int(cls_id), int(attr_id), int(camera_index),
    )


def global_tuple_to_result(token: str, box: Sequence[float]) -> Dict:
    x, y, z, w, length, height, yaw, vx, vy, score, cls_id = box[:11]
    attr_id = int(box[11]) if len(box) > 11 else -1
    det_name = bev.DETECTION_NAMES[int(cls_id)]
    attr_name = (
        bev.ATTRIBUTE_NAMES[attr_id]
        if 0 <= attr_id < len(bev.ATTRIBUTE_NAMES)
        else bev.attribute_for(det_name, float(vx), float(vy))
    )
    quat = Quaternion(axis=(0.0, 0.0, 1.0), angle=float(yaw))
    return {
        "sample_token": token,
        "translation": [float(x), float(y), float(z)],
        "size": [float(w), float(length), float(height)],
        "rotation": [float(value) for value in quat.elements],
        "velocity": [float(vx), float(vy)],
        "detection_name": det_name,
        "detection_score": float(score),
        "attribute_name": attr_name,
    }


def merge_subset(
    caches: Dict[str, Dict[str, List[Dict]]],
    cameras: Sequence[str],
    merge_iou: float,
    max_boxes: int,
    apply_nms: bool = True,
) -> Dict[str, List[Dict]]:
    tokens = tuple(next(iter(caches.values())).keys())
    expected = set(tokens)
    for camera in cameras:
        if set(caches[camera]) != expected:
            raise ValueError(f"sample-token mismatch in {camera} cache")
    output: Dict[str, List[Dict]] = {}
    for token in tokens:
        boxes: List[Tuple[float, ...]] = []
        for camera in cameras:
            camera_index = CAMERAS.index(camera)
            boxes.extend(
                result_to_global_tuple(record, camera_index)
                for record in caches[camera][token]
            )
        merged = bev.nms_classwise(boxes, merge_iou) if apply_nms else boxes
        merged = sorted(merged, key=lambda box: box[9], reverse=True)[:max_boxes]
        output[token] = [global_tuple_to_result(token, box) for box in merged]
    return output


def filter_to_camera_union(nusc: NuScenes, eval_boxes, cameras: Sequence[str]):
    for token in eval_boxes.sample_tokens:
        eval_boxes.boxes[token] = [
            box
            for box in eval_boxes.boxes[token]
            if any(
                bev.global_eval_box_visible_in_camera(nusc, token, box, camera)
                for camera in cameras
            )
        ]
    return eval_boxes


def run_evaluation(
    nusc: NuScenes,
    result_path: Path,
    output_dir: Path,
    cameras: Sequence[str],
    selected_fov: bool,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = NuScenesEval(
        nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=str(result_path),
        eval_set="val",
        output_dir=str(output_dir),
        verbose=False,
    )
    if selected_fov:
        evaluator.gt_boxes = filter_to_camera_union(nusc, evaluator.gt_boxes, cameras)
        evaluator.pred_boxes = filter_to_camera_union(nusc, evaluator.pred_boxes, cameras)
    evaluator.main(plot_examples=0, render_curves=False)
    return json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))


def metric_row(subset_id: str, cameras: Sequence[str], protocol: str, metrics: Dict) -> Dict:
    return {
        "subset_id": subset_id,
        "camera_count": len(cameras),
        "cameras": "+".join(cameras),
        "protocol": protocol,
        "official": protocol == "official_360",
        "NDS": float(metrics["nd_score"]),
        "mAP": float(metrics["mean_ap"]),
        "mATE": float(metrics["tp_errors"]["trans_err"]),
        "mASE": float(metrics["tp_errors"]["scale_err"]),
        "mAOE": float(metrics["tp_errors"]["orient_err"]),
        "mAVE": float(metrics["tp_errors"]["vel_err"]),
        "mAAE": float(metrics["tp_errors"]["attr_err"]),
    }


def summarise(rows: Sequence[Dict], six_camera_nds: float) -> List[Dict]:
    summary = []
    for protocol in ("official_360", "selected_fov"):
        protocol_rows = [row for row in rows if row["protocol"] == protocol]
        for count in range(1, 7):
            group = [row for row in protocol_rows if row["camera_count"] == count]
            nds = np.asarray([row["NDS"] for row in group], dtype=np.float64)
            maps = np.asarray([row["mAP"] for row in group], dtype=np.float64)
            best = max(group, key=lambda row: (row["NDS"], row["mAP"]))
            worst = min(group, key=lambda row: (row["NDS"], row["mAP"]))
            summary.append({
                "protocol": protocol,
                "official": protocol == "official_360",
                "camera_count": count,
                "subset_count": len(group),
                "NDS_mean": float(nds.mean()),
                "NDS_std": float(nds.std(ddof=0)),
                "NDS_min": float(nds.min()),
                "NDS_max": float(nds.max()),
                "mAP_mean": float(maps.mean()),
                "mAP_std": float(maps.std(ddof=0)),
                "mAP_min": float(maps.min()),
                "mAP_max": float(maps.max()),
                "official_NDS_retention_vs_6": (
                    float(nds.mean() / six_camera_nds)
                    if protocol == "official_360" and six_camera_nds > 0 else None
                ),
                "best_subset": best["cameras"],
                "worst_subset": worst["cameras"],
            })
    return summary


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_results_sha256(results: Dict[str, List[Dict]]) -> str:
    """Hash a prediction set after removing irrelevant JSON/dictionary ordering."""
    canonical = {
        token: sorted(
            records,
            key=lambda record: (
                record["detection_name"],
                -float(record["detection_score"]),
                tuple(round(float(value), 9) for value in record["translation"]),
                tuple(round(float(value), 9) for value in record["size"]),
                tuple(round(float(value), 9) for value in record["rotation"]),
            ),
        )
        for token, records in sorted(results.items())
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument(
        "--checkpoint-manifest",
        help="JSON mapping each camera name to its camera-specialist checkpoint",
    )
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_manifest:
        checkpoint_manifest_path = Path(args.checkpoint_manifest)
        checkpoint_payload = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        if set(checkpoint_payload) != set(CAMERAS):
            raise ValueError("checkpoint manifest must contain exactly the six cameras")
        checkpoint_paths = {camera: Path(checkpoint_payload[camera]) for camera in CAMERAS}
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing specialist checkpoints: {missing}")
        checkpoint = None
    else:
        checkpoint_manifest_path = None
        checkpoint = Path(args.checkpoint)
        checkpoint_paths = {camera: checkpoint for camera in CAMERAS}
    caches = {
        camera: read_cache(cache_dir / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    cache_hashes = {
        camera: sha256(cache_dir / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=False)
    temp_path = output_dir / "_current_results_nusc.json"
    rows_by_key: Dict[Tuple[str, str], Dict] = {}

    # A shared-view detector should not depend on the order in which its camera
    # predictions are supplied to the deterministic merge stage.  This audit is
    # inference-free and therefore adds no GPU cost.
    merged_forward = merge_subset(caches, CAMERAS, args.merge_iou, args.max_boxes)
    merged_reverse = merge_subset(
        caches, tuple(reversed(CAMERAS)), args.merge_iou, args.max_boxes,
    )
    forward_sha = canonical_results_sha256(merged_forward)
    reverse_sha = canonical_results_sha256(merged_reverse)
    order_invariant = forward_sha == reverse_sha
    if not order_invariant:
        print(
            "[WARN] camera-order audit found differences; the fixed canonical "
            "camera order remains the declared evaluation protocol",
            flush=True,
        )

    all_subsets = [
        combination
        for count in range(1, 7)
        for combination in itertools.combinations(CAMERAS, count)
    ]
    nested_diagnostic = tuple(tuple(CAMERAS[:count]) for count in range(1, 7))
    fov_subsets = tuple(dict.fromkeys(
        tuple(combination for combination in all_subsets if len(combination) == 1)
        + nested_diagnostic
    ))

    # Finish all 63 official results first so the publication-facing
    # camera-count curve is available before the selected-FOV diagnostic.  The
    # latter uses 11 predeclared subsets (all single cameras plus one nested
    # 1-to-6 sequence).  Leave-one-out is already exhausted by official_360,
    # so repeating it under the diagnostic protocol adds no main-paper evidence.
    for protocol, selected_fov in (("official_360", False), ("selected_fov", True)):
        protocol_subsets = all_subsets if protocol == "official_360" else fov_subsets
        for index, cameras in enumerate(protocol_subsets, start=1):
            subset_id = f"c{len(cameras)}_" + "__".join(cameras)
            subset_dir = output_dir / "subsets" / subset_id
            metrics_path = subset_dir / protocol / "metrics_summary.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                merged = merge_subset(caches, cameras, args.merge_iou, args.max_boxes)
                payload = {
                    "meta": {
                        "use_camera": True, "use_lidar": False, "use_radar": False,
                        "use_map": False, "use_external": False,
                    },
                    "results": merged,
                }
                temp_path.write_text(json.dumps(payload), encoding="utf-8")
                metrics = run_evaluation(
                    nusc, temp_path, subset_dir / protocol, cameras, selected_fov,
                )
            row = metric_row(subset_id, cameras, protocol, metrics)
            rows_by_key[(subset_id, protocol)] = row
            progress_rows = list(rows_by_key.values())
            write_csv(output_dir / "all_63_subsets.progress.csv", progress_rows)
            print(
                f"[{protocol} {index:02d}/{len(protocol_subsets)}] {subset_id}: "
                f"NDS={metrics['nd_score']:.4f} mAP={metrics['mean_ap']:.4f}",
                flush=True,
            )

        if protocol == "official_360":
            ablation_dir = output_dir / "fusion_ablation" / "no_global_nms"
            ablation_path = ablation_dir / "metrics_summary.json"
            if ablation_path.exists():
                no_nms_metrics = json.loads(ablation_path.read_text(encoding="utf-8"))
            else:
                no_nms_results = merge_subset(
                    caches, CAMERAS, args.merge_iou, args.max_boxes, apply_nms=False,
                )
                temp_path.write_text(json.dumps({
                    "meta": {
                        "use_camera": True, "use_lidar": False, "use_radar": False,
                        "use_map": False, "use_external": False,
                    },
                    "results": no_nms_results,
                }), encoding="utf-8")
                no_nms_metrics = run_evaluation(
                    nusc, temp_path, ablation_dir, CAMERAS, False,
                )
            six_id = "c6_" + "__".join(CAMERAS)
            with_nms = rows_by_key[(six_id, "official_360")]
            fusion_rows = [
                {
                    "fusion": "class-wise rotated global NMS (ours)",
                    "NDS": with_nms["NDS"], "mAP": with_nms["mAP"],
                    "mATE": with_nms["mATE"], "mASE": with_nms["mASE"],
                    "mAOE": with_nms["mAOE"], "mAVE": with_nms["mAVE"],
                    "mAAE": with_nms["mAAE"],
                },
                {
                    "fusion": "concatenate + global top-500 (no cross-view NMS)",
                    "NDS": float(no_nms_metrics["nd_score"]),
                    "mAP": float(no_nms_metrics["mean_ap"]),
                    "mATE": float(no_nms_metrics["tp_errors"]["trans_err"]),
                    "mASE": float(no_nms_metrics["tp_errors"]["scale_err"]),
                    "mAOE": float(no_nms_metrics["tp_errors"]["orient_err"]),
                    "mAVE": float(no_nms_metrics["tp_errors"]["vel_err"]),
                    "mAAE": float(no_nms_metrics["tp_errors"]["attr_err"]),
                },
            ]
            write_csv(output_dir / "fusion_ablation.csv", fusion_rows)

    rows = []
    for protocol, protocol_subsets in (
        ("official_360", all_subsets), ("selected_fov", fov_subsets),
    ):
        for cameras in protocol_subsets:
            subset_id = f"c{len(cameras)}_" + "__".join(cameras)
            rows.append(rows_by_key[(subset_id, protocol)])

    if temp_path.exists():
        temp_path.unlink()
    six = next(
        row for row in rows
        if row["protocol"] == "official_360" and row["camera_count"] == 6
    )
    summary = summarise(rows, six["NDS"])
    leave_one_out = [
        row for row in rows
        if row["protocol"] == "official_360" and row["camera_count"] == 5
    ]

    write_csv(output_dir / "all_63_subsets.csv", rows)
    write_csv(output_dir / "camera_count_summary.csv", summary)
    write_csv(output_dir / "leave_one_camera_out.csv", leave_one_out)
    progress_path = output_dir / "all_63_subsets.progress.csv"
    if progress_path.exists():
        progress_path.unlink()
    integrity = {
        "protocol_version": "variable_camera_subsets_v1",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": sha256(checkpoint) if checkpoint is not None else None,
        "checkpoint_manifest": (
            str(checkpoint_manifest_path) if checkpoint_manifest_path is not None else None
        ),
        "camera_checkpoint_sha256": {
            camera: sha256(path) for camera, path in checkpoint_paths.items()
        },
        "camera_cache_sha256": cache_hashes,
        "camera_order": list(CAMERAS),
        "official_subset_count": 63,
        "selected_fov_subset_count": len(fov_subsets),
        "result_row_count": len(rows),
        "official_protocol": "nuScenes val detection_cvpr_2019, full 6019 samples, 360-degree GT",
        "selected_fov_protocol": (
            "diagnostic non-official metric: GT and predictions visible in the union "
            "of selected camera image planes"
        ),
        "merge": {"type": "exact class-wise rotated NMS", "iou": args.merge_iou},
        "parameter_count": 7_191_090,
        "deployed_parameter_count": (
            7_191_090 if checkpoint is not None else 6 * 7_191_090
        ),
        "single_shared_checkpoint": checkpoint is not None,
        "camera_order_invariance": {
            "tested_orders": [list(CAMERAS), list(reversed(CAMERAS))],
            "forward_result_sha256": forward_sha,
            "reverse_result_sha256": reverse_sha,
            "exact_match": order_invariant,
        },
    }
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "leave_one_out": leave_one_out}, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote publication tables to {output_dir}")


if __name__ == "__main__":
    main()
