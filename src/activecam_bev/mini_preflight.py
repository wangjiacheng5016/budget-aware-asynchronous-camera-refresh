#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from activecam_core import (
    CAMERAS,
    evaluation_scope,
    local_detection_utility,
    oldest_first_selections,
    propagate_detection,
    random_selections,
    refresh_deadline_exceeded,
)


RESULT_META = {
    "use_camera": True,
    "use_lidar": False,
    "use_radar": False,
    "use_map": False,
    "use_external": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_dependencies(legacy_src: Path):
    sys.path.insert(0, str(legacy_src.resolve()))
    legacy = importlib.import_module("eval_variable_camera_subsets")
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.utils.splits import create_splits_scenes
    return legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes


def ordered_split_frames(nusc, split_names: list[str]) -> list[tuple[str, str]]:
    allowed = set(split_names)
    frames: list[tuple[str, str]] = []
    for scene in sorted(nusc.scene, key=lambda item: item["name"]):
        if scene["name"] not in allowed:
            continue
        token = scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            frames.append((token, scene["token"]))
            token = sample["next"]
    return frames


def load_ground_truth(nusc, legacy, tokens: list[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for token in tokens:
        sample = nusc.get("sample", token)
        records = []
        for annotation_token in sample["anns"]:
            annotation = nusc.get("sample_annotation", annotation_token)
            if int(annotation["num_lidar_pts"]) + int(annotation["num_radar_pts"]) <= 0:
                continue
            name = legacy.bev.CATEGORY_TO_DETECTION.get(annotation["category_name"])
            if name is None:
                continue
            records.append({
                "detection_name": name,
                "translation": [float(value) for value in annotation["translation"]],
                "size": [float(value) for value in annotation["size"]],
                "rotation": [float(value) for value in annotation["rotation"]],
                "velocity": [
                    0.0 if not math.isfinite(float(value)) else float(value)
                    for value in nusc.box_velocity(annotation_token)[:2]
                ],
            })
        output[token] = records
    return output


def merge_sources(legacy, token: str, sources, merge_iou: float, max_boxes: int):
    boxes = []
    for camera, records in sources:
        camera_index = CAMERAS.index(camera)
        boxes.extend(legacy.result_to_global_tuple(record, camera_index) for record in records)
    merged = legacy.bev.nms_classwise(boxes, merge_iou)
    merged = sorted(merged, key=lambda box: box[9], reverse=True)[:max_boxes]
    return [legacy.global_tuple_to_result(token, box) for box in merged]


def evaluate_predictions(
    *,
    method: str,
    predictions: dict[str, list[dict[str, Any]]],
    nusc,
    config_factory,
    NuScenesEval,
    eval_set: str,
    output_dir: Path,
) -> dict[str, Any]:
    run_dir = output_dir / "evaluations" / method
    metrics_path = run_dir / "metrics_summary.json"
    result_path = run_dir / "results_nusc.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        write_json(result_path, {"meta": RESULT_META, "results": predictions})
        evaluator = NuScenesEval(
            nusc,
            config=config_factory("detection_cvpr_2019"),
            result_path=str(result_path),
            eval_set=eval_set,
            output_dir=str(run_dir),
            verbose=False,
        )
        evaluator.main(plot_examples=0, render_curves=False)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "method": method,
        "NDS": float(metrics["nd_score"]),
        "mAP": float(metrics["mean_ap"]),
        "mATE": float(metrics["tp_errors"]["trans_err"]),
        "mASE": float(metrics["tp_errors"]["scale_err"]),
        "mAOE": float(metrics["tp_errors"]["orient_err"]),
        "mAVE": float(metrics["tp_errors"]["vel_err"]),
        "mAAE": float(metrics["tp_errors"]["attr_err"]),
    }


def build_policy_predictions(
    *,
    legacy,
    caches,
    ordered_frames,
    selections,
    merge_iou: float,
    max_boxes: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        token: merge_sources(
            legacy,
            token,
            [(camera, caches[camera][token]) for camera in selections[token]],
            merge_iou,
            max_boxes,
        )
        for token, _scene in ordered_frames
    }


def build_memory_predictions(
    *,
    legacy,
    caches,
    nusc,
    ordered_frames,
    selections,
    mode: str,
    tau_s: float,
    merge_iou: float,
    max_boxes: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if mode not in {"static", "motion"}:
        raise ValueError(f"unsupported memory mode: {mode}")
    predictions = {}
    ages = []
    active_scene = None
    memory = {}
    warm_start_frames = 0
    for token, scene_token in ordered_frames:
        sample = nusc.get("sample", token)
        timestamp = int(sample["timestamp"])
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {
                camera: (timestamp, caches[camera][token])
                for camera in CAMERAS
            }
            warm_start_frames += 1
        selected = set(selections[token])
        sources = []
        for camera in CAMERAS:
            if camera in selected:
                records = caches[camera][token]
                memory[camera] = (timestamp, records)
            else:
                memory_timestamp, stored = memory[camera]
                age_s = max(0.0, (timestamp - memory_timestamp) / 1_000_000.0)
                ages.append(age_s)
                records = [
                    propagate_detection(
                        record,
                        age_s=age_s,
                        tau_s=tau_s,
                        use_velocity=mode == "motion",
                    )
                    for record in stored
                ]
            sources.append((camera, records))
        predictions[token] = merge_sources(legacy, token, sources, merge_iou, max_boxes)
    age_summary = {
        "warm_start_frames": warm_start_frames,
        "memory_observations": len(ages),
        "mean_age_s": sum(ages) / len(ages) if ages else 0.0,
        "max_age_s": max(ages, default=0.0),
        "age_histogram_s": dict(sorted(Counter(round(value, 2) for value in ages).items())),
    }
    return predictions, age_summary


def build_memory_conditioned_oracle(
    *,
    legacy,
    caches,
    ground_truth,
    nusc,
    ordered_frames,
    combinations,
    mode: str,
    tau_s: float,
    max_age_s: float | None,
    proxy_score_threshold: float,
    merge_iou: float,
    max_boxes: int,
):
    """Greedily select each refresh set after simulating the available memory."""
    if mode not in {"static", "motion"}:
        raise ValueError(f"unsupported memory mode: {mode}")
    selections = {}
    predictions = {}
    chosen_ages = []
    active_scene = None
    memory = {}
    warm_start_frames = 0
    for token, scene_token in ordered_frames:
        sample = nusc.get("sample", token)
        timestamp = int(sample["timestamp"])
        next_interval_s = 0.0
        if sample["next"]:
            next_timestamp = int(nusc.get("sample", sample["next"])["timestamp"])
            next_interval_s = max(0.0, (next_timestamp - timestamp) / 1_000_000.0)
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {
                camera: (timestamp, caches[camera][token])
                for camera in CAMERAS
            }
            warm_start_frames += 1
        candidates = []
        for cameras in combinations:
            selected = set(cameras)
            sources = []
            candidate_ages = []
            violates_max_age = False
            for camera in CAMERAS:
                if camera in selected:
                    records = caches[camera][token]
                else:
                    memory_timestamp, stored = memory[camera]
                    age_s = max(0.0, (timestamp - memory_timestamp) / 1_000_000.0)
                    candidate_ages.append(age_s)
                    if (
                        max_age_s is not None
                        and refresh_deadline_exceeded(
                            age_s=age_s,
                            next_interval_s=next_interval_s,
                            max_age_s=max_age_s,
                        )
                    ):
                        violates_max_age = True
                    records = [
                        propagate_detection(
                            record,
                            age_s=age_s,
                            tau_s=tau_s,
                            use_velocity=mode == "motion",
                        )
                        for record in stored
                    ]
                sources.append((camera, records))
            if violates_max_age:
                continue
            merged = merge_sources(legacy, token, sources, merge_iou, max_boxes)
            utility = local_detection_utility(
                merged,
                ground_truth[token],
                score_threshold=proxy_score_threshold,
            )
            candidates.append((utility, tuple(cameras), merged, candidate_ages))
        if not candidates:
            raise RuntimeError(
                f"no feasible refresh set for token={token} max_age_s={max_age_s}"
            )
        _utility, selected_cameras, selected_predictions, selected_ages = sorted(
            candidates,
            key=lambda item: (-item[0], item[1]),
        )[0]
        selections[token] = selected_cameras
        predictions[token] = selected_predictions
        chosen_ages.extend(selected_ages)
        for camera in selected_cameras:
            memory[camera] = (timestamp, caches[camera][token])
    diagnostics = {
        "warm_start_frames": warm_start_frames,
        "memory_observations": len(chosen_ages),
        "mean_age_s": sum(chosen_ages) / len(chosen_ages) if chosen_ages else 0.0,
        "max_age_s": max(chosen_ages, default=0.0),
        "age_histogram_s": dict(
            sorted(Counter(round(value, 2) for value in chosen_ages).items())
        ),
    }
    return selections, predictions, diagnostics


def build_controlled_staleness(
    *,
    legacy,
    caches,
    nusc,
    ordered_frames,
    lag: int,
    mode: str,
    tau_s: float,
    merge_iou: float,
    max_boxes: int,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    scene_frames: dict[str, list[str]] = {}
    for token, scene in ordered_frames:
        scene_frames.setdefault(scene, []).append(token)
    output = {}
    ages = []
    for frames in scene_frames.values():
        for index, token in enumerate(frames):
            source_token = frames[max(0, index - lag)]
            current_time = int(nusc.get("sample", token)["timestamp"])
            source_time = int(nusc.get("sample", source_token)["timestamp"])
            age_s = max(0.0, (current_time - source_time) / 1_000_000.0)
            ages.append(age_s)
            sources = []
            for camera in CAMERAS:
                records = [
                    propagate_detection(
                        record,
                        age_s=age_s,
                        tau_s=tau_s,
                        use_velocity=mode == "motion",
                    )
                    for record in caches[camera][source_token]
                ]
                sources.append((camera, records))
            output[token] = merge_sources(legacy, token, sources, merge_iou, max_boxes)
    return output, sum(ages) / len(ages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--eval-set", default="mini_val")
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--proxy-score-threshold", type=float, default=0.05)
    parser.add_argument("--tau-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--skip-staleness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.camera_count <= len(CAMERAS):
        raise ValueError("camera-count must be in [1, 6]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes = load_dependencies(
        Path(args.legacy_src)
    )
    cache_dir = Path(args.cache_dir)
    caches = {
        camera: legacy.read_cache(cache_dir / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    split_names = create_splits_scenes()[args.eval_set]
    ordered_frames = ordered_split_frames(nusc, split_names)
    tokens = [token for token, _scene in ordered_frames]
    expected = set(tokens)
    for camera, cache in caches.items():
        if set(cache) != expected:
            raise ValueError(
                f"{camera} cache token mismatch: got={len(cache)} expected={len(expected)}"
            )
    ground_truth = load_ground_truth(nusc, legacy, tokens)
    combinations = list(itertools.combinations(CAMERAS, args.camera_count))
    current_by_subset = {
        cameras: legacy.merge_subset(caches, cameras, args.merge_iou, args.max_boxes)
        for cameras in combinations
    }

    rows = []
    static_rows = []
    for cameras in combinations:
        label = "fixed_" + "__".join(cameras)
        row = evaluate_predictions(
            method=label,
            predictions=current_by_subset[cameras],
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output_dir,
        )
        row.update({"policy": "fixed", "memory": "zero", "camera_count": args.camera_count})
        rows.append(row)
        static_rows.append(row)
        print(f"[FIXED] {label} NDS={row['NDS']:.4f} mAP={row['mAP']:.4f}", flush=True)

    full_predictions = legacy.merge_subset(caches, CAMERAS, args.merge_iou, args.max_boxes)
    full_row = evaluate_predictions(
        method="full6_current",
        predictions=full_predictions,
        nusc=nusc,
        config_factory=config_factory,
        NuScenesEval=NuScenesEval,
        eval_set=args.eval_set,
        output_dir=output_dir,
    )
    full_row.update({"policy": "full", "memory": "zero", "camera_count": 6})
    rows.append(full_row)

    best_fixed = max(static_rows, key=lambda row: (row["NDS"], row["mAP"]))
    oracle_selections = {}
    oracle_utilities = {}
    for token in tokens:
        ranked = sorted(
            (
                (
                    local_detection_utility(
                        current_by_subset[cameras][token],
                        ground_truth[token],
                        score_threshold=args.proxy_score_threshold,
                    ),
                    cameras,
                )
                for cameras in combinations
            ),
            key=lambda item: (-item[0], item[1]),
        )
        oracle_utilities[token], oracle_selections[token] = ranked[0]

    policy_selections = {
        "frame_oracle_proxy": oracle_selections,
        "age_only": oldest_first_selections(
            ordered_frames, camera_count=args.camera_count
        ),
        "random": random_selections(
            ordered_frames,
            camera_count=args.camera_count,
            seed=args.seed,
        ),
    }
    memory_diagnostics = {}
    for policy, selections in policy_selections.items():
        zero_predictions = build_policy_predictions(
            legacy=legacy,
            caches=caches,
            ordered_frames=ordered_frames,
            selections=selections,
            merge_iou=args.merge_iou,
            max_boxes=args.max_boxes,
        )
        row = evaluate_predictions(
            method=f"{policy}_zero",
            predictions=zero_predictions,
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output_dir,
        )
        row.update({"policy": policy, "memory": "zero", "camera_count": args.camera_count})
        rows.append(row)
        if policy not in {"age_only", "frame_oracle_proxy"}:
            continue
        for memory_mode in ("static", "motion"):
            memory_predictions, diagnostics = build_memory_predictions(
                legacy=legacy,
                caches=caches,
                nusc=nusc,
                ordered_frames=ordered_frames,
                selections=selections,
                mode=memory_mode,
                tau_s=args.tau_s,
                merge_iou=args.merge_iou,
                max_boxes=args.max_boxes,
            )
            method = f"{policy}_{memory_mode}_tau{args.tau_s:g}"
            memory_row = evaluate_predictions(
                method=method,
                predictions=memory_predictions,
                nusc=nusc,
                config_factory=config_factory,
                NuScenesEval=NuScenesEval,
                eval_set=args.eval_set,
                output_dir=output_dir,
            )
            memory_row.update({
                "policy": policy,
                "memory": memory_mode,
                "camera_count": args.camera_count,
            })
            rows.append(memory_row)
            memory_diagnostics[method] = diagnostics

    for memory_mode in ("static", "motion"):
        conditional_selections, conditional_predictions, diagnostics = (
            build_memory_conditioned_oracle(
                legacy=legacy,
                caches=caches,
                ground_truth=ground_truth,
                nusc=nusc,
                ordered_frames=ordered_frames,
                combinations=combinations,
                mode=memory_mode,
                tau_s=args.tau_s,
                max_age_s=None,
                proxy_score_threshold=args.proxy_score_threshold,
                merge_iou=args.merge_iou,
                max_boxes=args.max_boxes,
            )
        )
        method = f"memory_conditioned_oracle_{memory_mode}_tau{args.tau_s:g}"
        conditional_row = evaluate_predictions(
            method=method,
            predictions=conditional_predictions,
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output_dir,
        )
        conditional_row.update({
            "policy": "memory_conditioned_oracle",
            "memory": memory_mode,
            "camera_count": args.camera_count,
        })
        rows.append(conditional_row)
        policy_selections[method] = conditional_selections
        memory_diagnostics[method] = diagnostics

    infeasible_safety_constraints = [
        {
            "max_age_s": 0.6,
            "reason": (
                "capacity lower bound: six cameras at K=3 require two refresh slots, "
                "while adjacent nuScenes keyframes are approximately 0.5 s apart"
            ),
        }
    ]
    for max_age_s in (1.0, 1.1, 1.5, 2.0):
        try:
            constrained_selections, constrained_predictions, diagnostics = (
                build_memory_conditioned_oracle(
                    legacy=legacy,
                    caches=caches,
                    ground_truth=ground_truth,
                    nusc=nusc,
                    ordered_frames=ordered_frames,
                    combinations=combinations,
                    mode="motion",
                    tau_s=args.tau_s,
                    max_age_s=max_age_s,
                    proxy_score_threshold=args.proxy_score_threshold,
                    merge_iou=args.merge_iou,
                    max_boxes=args.max_boxes,
                )
            )
        except RuntimeError as error:
            infeasible_safety_constraints.append({
                "max_age_s": max_age_s,
                "reason": str(error),
            })
            print(f"[INFEASIBLE] max_age_s={max_age_s:g}: {error}")
            continue
        method = f"safety_conditioned_oracle_motion_maxage{max_age_s:g}_tau{args.tau_s:g}"
        constrained_row = evaluate_predictions(
            method=method,
            predictions=constrained_predictions,
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output_dir,
        )
        constrained_row.update({
            "policy": "safety_conditioned_oracle",
            "memory": "motion",
            "camera_count": args.camera_count,
        })
        rows.append(constrained_row)
        policy_selections[method] = constrained_selections
        diagnostics["configured_max_age_s"] = max_age_s
        memory_diagnostics[method] = diagnostics

    staleness_rows = []
    if not args.skip_staleness:
        for lag in (1, 2, 3, 4):
            for mode in ("static", "motion"):
                predictions, mean_age_s = build_controlled_staleness(
                    legacy=legacy,
                    caches=caches,
                    nusc=nusc,
                    ordered_frames=ordered_frames,
                    lag=lag,
                    mode=mode,
                    tau_s=args.tau_s,
                    merge_iou=args.merge_iou,
                    max_boxes=args.max_boxes,
                )
                method = f"stale_lag{lag}_{mode}_tau{args.tau_s:g}"
                row = evaluate_predictions(
                    method=method,
                    predictions=predictions,
                    nusc=nusc,
                    config_factory=config_factory,
                    NuScenesEval=NuScenesEval,
                    eval_set=args.eval_set,
                    output_dir=output_dir,
                )
                row.update({"policy": "controlled_staleness", "memory": mode, "camera_count": 0})
                rows.append(row)
                staleness_rows.append({
                    "method": method,
                    "lag_keyframes": lag,
                    "mean_age_s": mean_age_s,
                    "mode": mode,
                    "NDS": row["NDS"],
                    "mAP": row["mAP"],
                })

    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "staleness.csv", staleness_rows)
    write_json(output_dir / "selections.json", {
        policy: {token: list(cameras) for token, cameras in selections.items()}
        for policy, selections in policy_selections.items()
    })
    oracle_row = next(row for row in rows if row["method"] == "frame_oracle_proxy_zero")
    age_zero = next(row for row in rows if row["method"] == "age_only_zero")
    age_memory = max(
        (row for row in rows if row["policy"] == "age_only" and row["memory"] != "zero"),
        key=lambda row: row["NDS"],
    )
    oracle_memory = max(
        (
            row for row in rows
            if row["policy"] == "frame_oracle_proxy" and row["memory"] != "zero"
        ),
        key=lambda row: row["NDS"],
    )
    conditional_memory = max(
        (row for row in rows if row["policy"] == "memory_conditioned_oracle"),
        key=lambda row: row["NDS"],
    )
    safety_conditioned_memory = max(
        (row for row in rows if row["policy"] == "safety_conditioned_oracle"),
        key=lambda row: row["NDS"],
    )
    diagnostic_only, evaluation_reason = evaluation_scope(args.version, args.eval_set)
    summary = {
        "diagnostic_only": diagnostic_only,
        "reason": evaluation_reason,
        "sample_count": len(tokens),
        "scene_count": len({scene for _token, scene in ordered_frames}),
        "camera_count": args.camera_count,
        "full6": full_row,
        "best_fixed": best_fixed,
        "frame_oracle_proxy": oracle_row,
        "oracle_minus_best_fixed_NDS": oracle_row["NDS"] - best_fixed["NDS"],
        "best_frame_oracle_memory": oracle_memory,
        "oracle_memory_gain_over_oracle_zero_NDS": oracle_memory["NDS"] - oracle_row["NDS"],
        "best_memory_conditioned_oracle": conditional_memory,
        "memory_conditioned_gain_over_age_memory_NDS": (
            conditional_memory["NDS"] - age_memory["NDS"]
        ),
        "best_safety_conditioned_oracle": safety_conditioned_memory,
        "safety_conditioned_gain_over_unconstrained_NDS": (
            safety_conditioned_memory["NDS"] - conditional_memory["NDS"]
        ),
        "safety_feasibility": {
            "requested_max_age_s": [0.6, 1.0, 1.1, 1.5, 2.0],
            "infeasible": infeasible_safety_constraints,
            "timestamp_tolerance_s": 0.01,
        },
        "age_only_zero": age_zero,
        "best_age_only_memory": age_memory,
        "memory_gain_over_age_zero_NDS": age_memory["NDS"] - age_zero["NDS"],
        "memory_diagnostics": memory_diagnostics,
        "oracle_proxy_utility_mean": sum(oracle_utilities.values()) / len(oracle_utilities),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "protocol.json", {
        "version": args.version,
        "eval_set": args.eval_set,
        "data_root": str(Path(args.data_root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "cache_sha256": {
            camera: sha256(cache_dir / camera / "results_nusc.json")
            for camera in CAMERAS
        },
        "camera_order": list(CAMERAS),
        "merge_iou": args.merge_iou,
        "max_boxes": args.max_boxes,
        "proxy": {
            "purpose": "rank frame-level candidate actions only; never called frame NDS/mAP",
            "distance_thresholds_m": [0.5, 1.0, 2.0, 4.0],
            "false_positive_weight": 0.10,
            "score_threshold": args.proxy_score_threshold,
        },
        "memory": {
            "representation": "sparse global-frame detection records",
            "static": "global position held constant; ego motion is already compensated by the global frame",
            "motion": "global x/y propagated with the detector's predicted velocity",
            "score_decay": f"exp(-age/{args.tau_s:g})",
        },
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
