#!/usr/bin/env python3
"""Replay frozen ActiveCam artifacts and emit non-official stratified diagnostics."""

from __future__ import annotations

import math
import statistics
import gzip
import json
import argparse
import csv
import hashlib
import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .activecam_core import propagate_detection
except ImportError:
    from activecam_core import propagate_detection

AGE_BIN_EDGES_S = (0.0, 0.75, 1.25, 1.75, 2.25)
AGE_BIN_LABELS = ("0.5s", "1.0s", "1.5s", "2.0s")
DISTANCE_BIN_EDGES_M = (0.0, 20.0, 40.0, float("inf"))
DISTANCE_BIN_LABELS = ("0-20m", "20-40m", "40m+")
CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_result_sha256(results: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        results, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_result(path: Path) -> Path:
    candidates = (
        path,
        path / "results_nusc.json",
        path / "results_nusc.json.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    nested = sorted(path.glob("evaluations/*/results_nusc.json*")) if path.is_dir() else []
    if len(nested) == 1:
        return nested[0]
    raise FileNotFoundError(f"cannot resolve one result artifact from {path}")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_output_manifest(output_dir: Path, *, run_name: str) -> dict[str, Any]:
    outputs = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative == "manifest.json":
            continue
        outputs[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"schema_version": 1, "run": run_name, "outputs": outputs}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--run-dir", required=True, help="Frozen run containing selections/diagnostics/results.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-mode", choices=("motion", "static"), required=True)
    parser.add_argument("--memory-tau-s", type=float, default=1e9)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--match-distance-m", type=float, default=2.0)
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--expected-camera-count", type=int, required=True)
    parser.add_argument("--reference-result", help="Required for a production-parity replay; directory/json/json.gz.")
    return parser.parse_args(argv)


def _load_legacy(path: Path) -> Any:
    sys.path.insert(0, str(path.resolve()))
    return importlib.import_module("eval_variable_camera_subsets")


def _ordered_frames(nusc: Any, scene_names: Sequence[str]) -> list[tuple[str, str]]:
    allowed = set(scene_names)
    frames = []
    for scene in sorted(nusc.scene, key=lambda item: item["name"]):
        if scene["name"] not in allowed:
            continue
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            frames.append((token, scene["token"]))
            token = sample["next"]
    return frames


def _ground_truth(nusc: Any, legacy: Any, token: str) -> list[dict[str, Any]]:
    sample = nusc.get("sample", token)
    records = []
    for annotation_token in sample["anns"]:
        annotation = nusc.get("sample_annotation", annotation_token)
        if int(annotation["num_lidar_pts"]) + int(annotation["num_radar_pts"]) <= 0:
            continue
        name = legacy.bev.CATEGORY_TO_DETECTION.get(annotation["category_name"])
        if name is None:
            continue
        velocity = [
            0.0 if not math.isfinite(float(value)) else float(value)
            for value in nusc.box_velocity(annotation_token)[:2]
        ]
        records.append({
            "translation": [float(value) for value in annotation["translation"]],
            "size": [float(value) for value in annotation["size"]],
            "rotation": [float(value) for value in annotation["rotation"]],
            "velocity": velocity,
            "detection_name": name,
            "attribute_name": (
                nusc.get("attribute", annotation["attribute_tokens"][0])["name"]
                if annotation.get("attribute_tokens") else ""
            ),
        })
    return records


def _ego_translation(nusc: Any, token: str) -> list[float]:
    sample = nusc.get("sample", token)
    data_token = sample["data"]["LIDAR_TOP"]
    pose_token = nusc.get("sample_data", data_token)["ego_pose_token"]
    return [float(value) for value in nusc.get("ego_pose", pose_token)["translation"]]


def _aggregate_group_rows(frame_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["dimension"], row["group"], row["class"]) for row in frame_rows})
    output = []
    for dimension, group, class_name in keys:
        selected = [row for row in frame_rows if (
            row["dimension"], row["group"], row["class"]
        ) == (dimension, group, class_name)]
        tp = sum(int(row["tp"]) for row in selected)
        gt = sum(int(row["gt"]) for row in selected)
        entry = {
            "dimension": dimension, "group": group, "class": class_name,
            "gt": gt, "tp": tp, "fn": gt - tp,
            "recall": tp / gt if gt else None,
        }
        for metric in ("ate_m", "ase", "aoe_rad", "ave_mps", "aae"):
            numerator = sum(
                float(row[metric]) * int(row["tp"])
                for row in selected if row[metric] is not None
            )
            entry[metric] = numerator / tp if tp else None
        output.append(entry)
    return output


def _results_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return first == second


def _write_result(path: Path, predictions: Mapping[str, Any]) -> None:
    payload = {
        "meta": {
            "use_camera": True, "use_lidar": False, "use_radar": False,
            "use_map": False, "use_external": False,
        },
        "results": predictions,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))


def _official_evaluate(
    *, nusc: Any, predictions: Mapping[str, Any], eval_set: str,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    official_dir = output_dir / "official_evaluation"
    official_dir.mkdir(parents=True, exist_ok=True)
    result_path = official_dir / "results_nusc.json"
    _write_result(result_path, predictions)
    evaluator = NuScenesEval(
        nusc, config=config_factory("detection_cvpr_2019"),
        result_path=str(result_path), eval_set=eval_set,
        output_dir=str(official_dir), verbose=False,
    )
    evaluator.main(render_curves=False)
    metrics_path = official_dir / "metrics_summary.json"
    metrics = read_json(metrics_path)
    zipped = result_path.with_suffix(result_path.suffix + ".gz")
    with result_path.open("rb") as source, gzip.open(zipped, "wb", compresslevel=1) as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    result_path.unlink()
    return metrics, zipped


def assign_bin(value: float, edges: Sequence[float]) -> int | None:
    """Return a half-open bin index, with the final finite endpoint inclusive."""
    number = float(value)
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        if lower <= number < upper or (
            index == len(edges) - 2 and number == upper
        ):
            return index
    return None


def summarize_age_survivors(
    survivors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    assignments = []
    for item in survivors:
        index = assign_bin(float(item["source_age_s"]), AGE_BIN_EDGES_S)
        assignments.append(AGE_BIN_LABELS[index] if index is not None else "overflow")
    classes = sorted({str(item["record"]["detection_name"]) for item in survivors})
    rows = []
    for label in (*AGE_BIN_LABELS, "overflow"):
        for class_name in ("all", *classes):
            selected = [
                item for item, assigned in zip(survivors, assignments)
                if assigned == label and (
                    class_name == "all" or item["record"]["detection_name"] == class_name
                )
            ]
            ages = [float(item["source_age_s"]) for item in selected]
            scores = [float(item["record"]["detection_score"]) for item in selected]
            rows.append({
                "age_bin": label,
                "class": class_name,
                "survivor_count": len(selected),
                "actual_age_min_s": min(ages) if ages else None,
                "actual_age_mean_s": statistics.fmean(ages) if ages else None,
                "actual_age_max_s": max(ages) if ages else None,
                "mean_score": statistics.fmean(scores) if scores else None,
            })
    return rows


class AgeSurvivorAccumulator:
    """Streaming age statistics so provenance rows need not remain in memory."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], dict[str, float]] = {}
        self._classes: set[str] = set()

    def update(self, survivors: Sequence[Mapping[str, Any]]) -> None:
        for item in survivors:
            class_name = str(item["record"]["detection_name"])
            self._classes.add(class_name)
            age = float(item["source_age_s"])
            score = float(item["record"]["detection_score"])
            index = assign_bin(age, AGE_BIN_EDGES_S)
            label = AGE_BIN_LABELS[index] if index is not None else "overflow"
            for selected_class in ("all", class_name):
                state = self._values.setdefault((label, selected_class), {
                    "count": 0.0, "age_sum": 0.0, "age_min": age,
                    "age_max": age, "score_sum": 0.0,
                })
                state["count"] += 1
                state["age_sum"] += age
                state["age_min"] = min(state["age_min"], age)
                state["age_max"] = max(state["age_max"], age)
                state["score_sum"] += score

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for label in (*AGE_BIN_LABELS, "overflow"):
            for class_name in ("all", *sorted(self._classes)):
                state = self._values.get((label, class_name))
                count = int(state["count"]) if state else 0
                rows.append({
                    "age_bin": label, "class": class_name,
                    "survivor_count": count,
                    "actual_age_min_s": state["age_min"] if state else None,
                    "actual_age_mean_s": state["age_sum"] / count if state else None,
                    "actual_age_max_s": state["age_max"] if state else None,
                    "mean_score": state["score_sum"] / count if state else None,
                })
        return rows


def extract_official_per_class(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten official nuScenes per-class AP and TP-error fields without recomputation."""
    rows = []
    for class_name, threshold_aps in metrics.get("label_aps", {}).items():
        row = {
            "class": class_name,
            "mean_ap": metrics.get("mean_dist_aps", {}).get(class_name),
        }
        for threshold, value in threshold_aps.items():
            row[f"ap_{threshold}m"] = value
        row.update(metrics.get("label_tp_errors", {}).get(class_name, {}))
        rows.append(row)
    return rows


def _yaw(rotation: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in rotation)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_error(first: Sequence[float], second: Sequence[float]) -> float:
    difference = _yaw(first) - _yaw(second)
    return abs((difference + math.pi) % (2.0 * math.pi) - math.pi)


def _scale_error(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = math.prod(min(float(a), float(b)) for a, b in zip(first, second))
    first_volume = math.prod(float(value) for value in first)
    second_volume = math.prod(float(value) for value in second)
    return 1.0 - intersection / max(first_volume + second_volume - intersection, 1e-12)


def _match_predictions(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    match_distance_m: float,
) -> list[dict[str, Any]]:
    """Greedy class-wise center-distance matching in prediction-score order."""
    unmatched = set(range(len(ground_truth)))
    matches = []
    for prediction_index in sorted(
        range(len(predictions)),
        key=lambda index: float(predictions[index].get("detection_score", 0.0)),
        reverse=True,
    ):
        prediction = predictions[prediction_index]
        candidates = []
        for target_index in unmatched:
            target = ground_truth[target_index]
            if target["detection_name"] != prediction["detection_name"]:
                continue
            distance = math.dist(
                tuple(float(value) for value in prediction["translation"][:2]),
                tuple(float(value) for value in target["translation"][:2]),
            )
            if distance <= match_distance_m:
                candidates.append((distance, target_index))
        if not candidates:
            continue
        distance, target_index = min(candidates)
        unmatched.remove(target_index)
        target = ground_truth[target_index]
        velocity_error = math.dist(
            tuple(float(value) for value in prediction.get("velocity", (0.0, 0.0))[:2]),
            tuple(float(value) for value in target.get("velocity", (0.0, 0.0))[:2]),
        )
        matches.append({
            "prediction_index": prediction_index,
            "target_index": target_index,
            "class": str(target["detection_name"]),
            "ate_m": distance,
            "ase": _scale_error(prediction["size"], target["size"]),
            "aoe_rad": _angle_error(prediction["rotation"], target["rotation"]),
            "ave_mps": velocity_error,
            "aae": float(
                str(prediction.get("attribute_name", ""))
                != str(target.get("attribute_name", ""))
            ),
        })
    return matches


def _summarize_gt_groups(
    ground_truth: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any]],
    group_assignments: Sequence[str | None],
    *,
    dimension: str,
    groups: Sequence[str],
) -> list[dict[str, Any]]:
    classes = sorted({str(record["detection_name"]) for record in ground_truth})
    rows = []
    for group in groups:
        for class_name in ("all", *classes):
            target_indices = {
                index for index, (target, assigned) in enumerate(zip(ground_truth, group_assignments))
                if assigned == group and (
                    class_name == "all" or target["detection_name"] == class_name
                )
            }
            selected_matches = [
                match for match in matches if int(match["target_index"]) in target_indices
            ]
            tp = len(selected_matches)
            fn = len(target_indices) - tp
            row = {
                "dimension": dimension,
                "group": group,
                "class": class_name,
                "gt": len(target_indices),
                "tp": tp,
                "fn": fn,
                "recall": tp / len(target_indices) if target_indices else None,
            }
            for key in ("ate_m", "ase", "aoe_rad", "ave_mps", "aae"):
                values = [float(match[key]) for match in selected_matches]
                row[key] = statistics.fmean(values) if values else None
            rows.append(row)
    return rows


def stratify_matches(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    ego_translation: Sequence[float],
    dynamic_thresholds: Sequence[float] = (1.0, 0.5),
    match_distance_m: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return custom GT-defined range/motion recall and matched-box error tables."""
    matches = _match_predictions(predictions, ground_truth, match_distance_m)
    distance_assignments = []
    for target in ground_truth:
        distance = math.dist(
            tuple(float(value) for value in target["translation"][:2]),
            tuple(float(value) for value in ego_translation[:2]),
        )
        index = assign_bin(distance, DISTANCE_BIN_EDGES_M)
        distance_assignments.append(DISTANCE_BIN_LABELS[index] if index is not None else None)
    rows = _summarize_gt_groups(
        ground_truth, matches, distance_assignments,
        dimension="distance", groups=DISTANCE_BIN_LABELS,
    )
    for threshold in dynamic_thresholds:
        assignments = [
            "dynamic" if math.hypot(*[float(value) for value in target.get("velocity", (0.0, 0.0))[:2]]) > threshold else "static"
            for target in ground_truth
        ]
        label = f"motion_gt_speed_{threshold:g}mps"
        rows.extend(_summarize_gt_groups(
            ground_truth, matches, assignments,
            dimension=label, groups=("dynamic", "static"),
        ))
    return rows, matches


def merge_candidates_with_provenance(
    candidates: Sequence[Mapping[str, Any]],
    *,
    nms_fn: Callable[[Sequence[Sequence[float]], float], Sequence[Sequence[float]]],
    iou_fn: Callable[[Sequence[float], Sequence[float]], float],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Run production NMS and attach every directly suppressed input to its winner."""
    boxes = [tuple(candidate["box"]) for candidate in candidates]
    production_kept = [tuple(box) for box in nms_fn(boxes, iou_threshold)]
    kept_indices = []
    assigned: dict[int, list[int]] = {}
    classes = sorted({int(box[10]) for box in boxes})
    for class_id in classes:
        ordered = sorted(
            (index for index, box in enumerate(boxes) if int(box[10]) == class_id),
            key=lambda index: float(boxes[index][9]), reverse=True,
        )
        class_kept = []
        for index in ordered:
            suppressor = next(
                (
                    kept_index for kept_index in class_kept
                    if iou_fn(boxes[kept_index], boxes[index]) >= iou_threshold
                ),
                None,
            )
            if suppressor is None:
                class_kept.append(index)
                kept_indices.append(index)
                assigned[index] = [index]
            else:
                assigned[suppressor].append(index)
    replay_kept = [boxes[index] for index in kept_indices]
    if replay_kept != production_kept:
        raise RuntimeError("provenance NMS replay disagrees with production NMS")
    survivors = []
    for winner_index in kept_indices:
        winner = boxes[winner_index]
        contributors = [{
            "source_camera": str(candidates[index]["source_camera"]),
            "source_timestamp": int(candidates[index]["source_timestamp"]),
            "source_age_s": float(candidates[index]["source_age_s"]),
            "score": float(boxes[index][9]),
        } for index in assigned[winner_index]]
        contributors.sort(key=lambda item: (-item["score"], item["source_camera"]))
        ages = [item["source_age_s"] for item in contributors]
        output = dict(candidates[winner_index])
        output["contributors"] = contributors
        output["contributor_age_range_s"] = [min(ages), max(ages)]
        survivors.append(output)
    return survivors


def replay_frames(
    *,
    ordered_frames: Sequence[tuple[str, str]],
    timestamps: Mapping[str, int],
    selections: Mapping[str, Sequence[str]],
    caches: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    cameras: Sequence[str],
    legacy: Any,
    memory_mode: str,
    memory_tau_s: float,
    merge_iou: float,
    max_boxes: int,
    provenance_sink: Callable[[list[dict[str, Any]]], None] | None = None,
    survivor_sink: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Replay frozen selections using production propagation and rotated NMS."""
    if memory_mode not in {"motion", "static", "none"}:
        raise ValueError(f"unsupported memory mode: {memory_mode}")
    memory: dict[str, tuple[int, Sequence[Mapping[str, Any]]]] = {}
    active_scene = None
    predictions = {}
    for token, scene_token in ordered_frames:
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {}
        timestamp = int(timestamps[token])
        selected = set(selections[token])
        candidates = []
        for camera_index, camera in enumerate(cameras):
            if camera in selected:
                source_timestamp = timestamp
                records = caches[camera][token]
            elif camera in memory and memory_mode != "none":
                source_timestamp, stored = memory[camera]
                age_s = max(0.0, (timestamp - source_timestamp) / 1e6)
                records = [
                    propagate_detection(
                        record, age_s=age_s, tau_s=memory_tau_s,
                        use_velocity=memory_mode == "motion",
                    )
                    for record in stored
                ]
            else:
                continue
            age_s = max(0.0, (timestamp - source_timestamp) / 1e6)
            for record in records:
                candidates.append({
                    "box": tuple(legacy.result_to_global_tuple(record, camera_index)),
                    "record": dict(record),
                    "source_camera": camera,
                    "source_timestamp": source_timestamp,
                    "source_age_s": age_s,
                    "is_current": camera in selected,
                })
        survivors = merge_candidates_with_provenance(
            candidates,
            nms_fn=legacy.bev.nms_classwise,
            iou_fn=legacy.bev.rotated_iou,
            iou_threshold=merge_iou,
        )
        survivors.sort(key=lambda item: float(item["box"][9]), reverse=True)
        survivors = survivors[:max_boxes]
        predictions[token] = [
            legacy.global_tuple_to_result(token, survivor["box"])
            for survivor in survivors
        ]
        if survivor_sink is not None:
            survivor_sink([{
                "source_age_s": survivor["source_age_s"],
                "source_camera": survivor["source_camera"],
                "record": prediction,
            } for survivor, prediction in zip(survivors, predictions[token])])
        if provenance_sink is not None:
            provenance_sink([{
                "sample_token": token,
                "source_camera": survivor["source_camera"],
                "source_timestamp": survivor["source_timestamp"],
                "source_age_s": survivor["source_age_s"],
                "is_current": survivor["is_current"],
                "record": prediction,
                "contributors": survivor["contributors"],
                "contributor_age_range_s": survivor["contributor_age_range_s"],
            } for survivor, prediction in zip(survivors, predictions[token])])
        for camera in selected:
            memory[camera] = (timestamp, caches[camera][token])
    return predictions


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    legacy_src = Path(args.legacy_src).resolve()
    legacy = _load_legacy(legacy_src)
    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=str(data_root), verbose=False)
    ordered_frames = _ordered_frames(nusc, create_splits_scenes()[args.eval_set])
    expected_tokens = {token for token, _scene in ordered_frames}
    timestamps = {
        token: int(nusc.get("sample", token)["timestamp"])
        for token in expected_tokens
    }
    selections_path = run_dir / "selections.json"
    diagnostics_path = run_dir / "router_diagnostics.json"
    selections = read_json(selections_path)
    diagnostics = read_json(diagnostics_path)
    if set(selections) != expected_tokens:
        raise RuntimeError("selection token set is not the complete requested split")
    observed_counts = {len(action) for action in selections.values()}
    if observed_counts != {args.expected_camera_count}:
        raise RuntimeError(f"selection budget mismatch: {observed_counts}")
    if (
        int(diagnostics.get("frame_count", -1)) != len(ordered_frames)
        or int(diagnostics.get("camera_count_per_frame", -1)) != args.expected_camera_count
        or diagnostics.get("free_full6_warm_start") is not False
        or diagnostics.get("protocol", {}).get("uses_current_unselected_detection_or_gt") is not False
    ):
        raise RuntimeError("frozen routing diagnostics are incompatible")

    cache_paths = {}
    caches = {}
    for camera in CAMERAS:
        path = resolve_result(cache_dir / camera)
        payload = read_json(path)
        cache = payload.get("results", payload)
        if set(cache) != expected_tokens:
            raise RuntimeError(f"{camera}: cache token set mismatch")
        cache_paths[camera] = path
        caches[camera] = cache

    provenance_path = output_dir / "survivor_provenance.jsonl.gz"
    age_accumulator = AgeSurvivorAccumulator()
    with gzip.open(provenance_path, "wt", encoding="utf-8", compresslevel=6) as stream:
        def write_provenance(rows: list[dict[str, Any]]) -> None:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

        predictions = replay_frames(
            ordered_frames=ordered_frames,
            timestamps=timestamps,
            selections=selections,
            caches=caches,
            cameras=CAMERAS,
            legacy=legacy,
            memory_mode=args.memory_mode,
            memory_tau_s=args.memory_tau_s,
            merge_iou=args.merge_iou,
            max_boxes=args.max_boxes,
            provenance_sink=write_provenance,
            survivor_sink=age_accumulator.update,
        )
    parity = {
        "checked": args.reference_result is not None,
        "exact_payload_equal": None,
        "reference_path": None,
        "reference_sha256": None,
    }
    if args.reference_result:
        reference_path = resolve_result(Path(args.reference_result).resolve())
        reference_payload = read_json(reference_path)
        reference_results = reference_payload.get("results", reference_payload)
        equal = _results_equal(reference_results, predictions)
        parity.update({
            "exact_payload_equal": equal,
            "reference_path": str(reference_path),
            "reference_sha256": sha256(reference_path),
            "reference_semantic_results_sha256": semantic_result_sha256(reference_results),
            "replay_semantic_results_sha256": semantic_result_sha256(predictions),
        })
        if not equal:
            raise RuntimeError("replay predictions disagree with frozen production result")

    age_rows = age_accumulator.rows()
    frame_group_rows = []
    for token, _scene in ordered_frames:
        rows, _matches = stratify_matches(
            predictions[token], _ground_truth(nusc, legacy, token),
            ego_translation=_ego_translation(nusc, token),
            dynamic_thresholds=(1.0, 0.5),
            match_distance_m=args.match_distance_m,
        )
        frame_group_rows.extend(rows)
    group_rows = _aggregate_group_rows(frame_group_rows)

    metrics, replay_result_path = _official_evaluate(
        nusc=nusc, predictions=predictions, eval_set=args.eval_set,
        output_dir=output_dir,
    )
    official_rows = extract_official_per_class(metrics)

    _write_csv(output_dir / "age_survivors.csv", age_rows)
    _write_csv(output_dir / "gt_range_motion_groups.csv", group_rows)
    _write_csv(output_dir / "official_per_class.csv", official_rows)

    source_path = Path(__file__).resolve()
    protocol = {
        "schema_version": 1,
        "status": "custom diagnostic; not official AP or NDS",
        "version": args.version,
        "eval_set": args.eval_set,
        "frame_count": len(ordered_frames),
        "camera_count": args.expected_camera_count,
        "memory_mode": args.memory_mode,
        "memory_tau_s": args.memory_tau_s,
        "merge_iou": args.merge_iou,
        "max_boxes": args.max_boxes,
        "match_distance_m": args.match_distance_m,
        "match_semantics": "greedy class-wise center-distance in descending prediction-score order",
        "age_bin_edges_s": list(AGE_BIN_EDGES_S),
        "age_bin_display_labels": list(AGE_BIN_LABELS),
        "age_semantics": "NMS survivor winning-source age; contributors separately retained",
        "distance_bin_edges_m": [0.0, 20.0, 40.0, "inf"],
        "distance_semantics": "GT global center distance to sample ego pose",
        "dynamic_threshold_mps": 1.0,
        "dynamic_sensitivity_threshold_mps": 0.5,
        "dynamic_semantics": "GT planar speed strictly greater than threshold",
        "nms_semantics": "production exact rotated NMS; independent provenance replay must match kept tuple order",
        "parity": parity,
        "replay_result": {
            "path": str(replay_result_path),
            "sha256": sha256(replay_result_path),
        },
        "inputs": {
            "selections": {"path": str(selections_path), "sha256": sha256(selections_path)},
            "diagnostics": {"path": str(diagnostics_path), "sha256": sha256(diagnostics_path)},
            "caches": {camera: {"path": str(path), "sha256": sha256(path)} for camera, path in cache_paths.items()},
        },
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    manifest = build_output_manifest(output_dir, run_name=run_dir.name)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
