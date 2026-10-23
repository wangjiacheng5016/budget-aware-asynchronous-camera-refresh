#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .activecam_core import CAMERAS, propagate_detection
    from .mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        merge_sources,
        ordered_split_frames,
    )
    from .router_core import (
        STATE_FEATURES,
        build_pre_action_state,
        enumerate_actions,
        feasible_actions,
        switch_fraction,
    )
    from .router_model import BudgetConditionedSetRouter, IndependentCameraRouter
except ImportError:  # Direct script execution on the experiment server.
    from activecam_core import CAMERAS, propagate_detection
    from mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        merge_sources,
        ordered_split_frames,
    )
    from router_core import (
        STATE_FEATURES,
        build_pre_action_state,
        enumerate_actions,
        feasible_actions,
        switch_fraction,
    )
    from router_model import BudgetConditionedSetRouter, IndependentCameraRouter

try:
    from .result_io import write_result_json
except ImportError:
    from result_io import write_result_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe online simulation and official evaluation of the router."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-manifest")
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--policy", choices=("learned", "age_only", "random_safe"), default="learned")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--planned-interval-s", type=float, default=0.50)
    parser.add_argument("--max-age-s", type=float, default=1.50)
    parser.add_argument("--uncertainty-tau-s", type=float, default=1.00)
    parser.add_argument("--memory-tau-s", type=float, default=1.00)
    parser.add_argument(
        "--memory-mode",
        choices=("motion", "static", "none"),
        default="motion",
        help=(
            "motion propagates stored boxes with velocity, static only decays "
            "their scores, and none excludes unselected-camera memory from fusion."
        ),
    )
    parser.add_argument("--velocity-adaptive-decay-scale-mps", type=float)
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fallback-age-only", action="store_true")
    parser.add_argument("--override-threshold", type=float, default=0.0)
    parser.add_argument(
        "--max-stale-displacement-m",
        type=float,
        help=(
            "Fall back to age-only when a learned action would leave a camera "
            "with past-memory max_speed * next_age above this value."
        ),
    )
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument(
        "--result-json",
        help="Write a complete nuScenes-format prediction/submission JSON to this path.",
    )
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _age_priority_index(actions, ages_s, valid_indices) -> int:
    def key(index):
        chosen = set(actions[index])
        next_ages = [0.0 if camera in chosen else age for camera, age in zip(CAMERAS, ages_s)]
        return (max(next_ages), sum(next_ages), index)
    return min(valid_indices, key=key)


def _stale_displacement_risk_m(action, state_payload, planned_interval_s: float) -> float:
    selected = set(action)
    max_speed_index = STATE_FEATURES.index("max_speed_ratio")
    risks = []
    for camera, age_s, state_row in zip(
        CAMERAS, state_payload["ages_s"], state_payload["state"]
    ):
        if camera in selected:
            continue
        max_speed_mps = float(state_row[max_speed_index]) * 30.0
        risks.append(max_speed_mps * (float(age_s) + planned_interval_s))
    return max(risks, default=0.0)


def _build_predictions(legacy, token, timestamp, action, memory, caches, args):
    chosen = set(action)
    sources = []
    used_ages = []
    for camera in CAMERAS:
        if camera in chosen:
            records = caches[camera][token]
        elif camera in memory:
            source_timestamp, stored = memory[camera]
            age_s = max(0.0, (timestamp - source_timestamp) / 1_000_000.0)
            used_ages.append(age_s)
            if args.memory_mode == "none":
                records = []
            else:
                records = [
                    propagate_detection(
                        record,
                        age_s=age_s,
                        tau_s=args.memory_tau_s,
                        use_velocity=args.memory_mode == "motion",
                        velocity_decay_scale_mps=args.velocity_adaptive_decay_scale_mps,
                    )
                    for record in stored
                ]
        else:
            records = []
        sources.append((camera, records))
    predictions = merge_sources(legacy, token, sources, args.merge_iou, args.max_boxes)
    return predictions, used_ages


def main() -> None:
    args = parse_args()
    if args.policy == "learned" and not args.checkpoint:
        raise ValueError("--checkpoint is required for learned policy")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes = load_dependencies(
        Path(args.legacy_src)
    )
    if args.eval_set == "test" and not args.skip_official_eval:
        raise ValueError("nuScenes test has no local ground truth; use --skip-official-eval")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    split_names = list(create_splits_scenes()[args.eval_set])
    if args.max_scenes is not None:
        split_names = sorted(split_names)[: args.max_scenes]
    ordered_frames = ordered_split_frames(nusc, split_names)
    expected = {token for token, _scene in ordered_frames}
    cache_dir = Path(args.cache_dir)
    caches = {
        camera: legacy.read_cache(cache_dir / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    for camera, cache in caches.items():
        missing = expected.difference(cache)
        if missing:
            raise RuntimeError(f"{camera} cache is missing {len(missing)} evaluation tokens")

    actions = enumerate_actions(args.camera_count)
    device = torch.device(args.device)
    model = None
    checkpoint = None
    learned_model_type = None
    learned_target_mode = None
    action_masks = None
    if args.policy == "learned":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["camera_order"] != list(CAMERAS):
            raise RuntimeError("checkpoint camera order is incompatible")
        if checkpoint["actions"] != [list(action) for action in actions]:
            raise RuntimeError("checkpoint action table is incompatible")
        learned_model_type = checkpoint.get("model_type", "set")
        learned_target_mode = checkpoint.get("target_mode", "main")
        model_class = (
            IndependentCameraRouter
            if learned_model_type == "independent"
            else BudgetConditionedSetRouter
        )
        model = model_class(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        feature_mean = checkpoint["feature_mean"].to(device).view(1, 1, -1)
        feature_std = checkpoint["feature_std"].to(device).view(1, 1, -1)
        action_masks = torch.tensor(checkpoint["action_masks"], dtype=torch.float32, device=device)

    predictions: dict[str, list[dict[str, Any]]] = {}
    selections = {}
    route_rows = []
    all_memory_ages = []
    routing_times_ms = []
    propagation_merge_times_ms = []
    active_scene = None
    memory = {}
    previous_action: tuple[str, ...] = ()
    rng = random.Random(args.seed)
    emergency_overrides = 0
    actual_deadline_violation_frames = 0
    actual_deadline_violating_camera_events = 0
    learned_override_count = 0
    motion_guard_fallback_count = 0
    predicted_advantage_margins = []
    started = time.time()
    for frame_number, (token, scene_token) in enumerate(ordered_frames, 1):
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {}
            previous_action = ()
        timestamp = int(nusc.get("sample", token)["timestamp"])
        route_started = time.perf_counter()
        state_payload = build_pre_action_state(
            memory=memory,
            decision_timestamp=timestamp,
            previous_action=previous_action,
            max_age_s=args.max_age_s,
            uncertainty_tau_s=args.uncertainty_tau_s,
            strict_past=True,
        )
        initialized = [value >= 0 for value in state_payload["source_timestamps"]]
        feasible = list(feasible_actions(
            actions=actions,
            ages_s=state_payload["ages_s"],
            planned_interval_s=args.planned_interval_s,
            max_age_s=args.max_age_s,
            initialized=initialized,
        ))
        valid_indices = [index for index, valid in enumerate(feasible) if valid]
        emergency = False
        predicted_margin = None
        if not valid_indices:
            # Never hide an infeasible shield state; use a disclosed oldest-first fallback.
            emergency = True
            emergency_overrides += 1
            valid_indices = list(range(len(actions)))
        if args.policy == "age_only" or emergency:
            selected_index = _age_priority_index(actions, state_payload["ages_s"], valid_indices)
            score_values = None
        elif args.policy == "random_safe":
            selected_index = rng.choice(valid_indices)
            score_values = None
        else:
            effective_feasible = torch.zeros(1, len(actions), dtype=torch.bool, device=device)
            effective_feasible[0, valid_indices] = True
            state_tensor = torch.tensor(state_payload["state"], dtype=torch.float32, device=device)
            state_tensor = (state_tensor.unsqueeze(0) - feature_mean) / feature_std
            with torch.no_grad():
                scores = model(
                    state_tensor,
                    action_masks,
                    budget=torch.tensor([args.camera_count], device=device),
                    feasible_mask=effective_feasible,
                )
            model_index = int(scores.argmax(dim=1).item())
            baseline_index = _age_priority_index(
                actions, state_payload["ages_s"], valid_indices
            )
            predicted_margin = float(
                scores[0, model_index].item() - scores[0, baseline_index].item()
            )
            predicted_advantage_margins.append(predicted_margin)
            motion_risk_m = _stale_displacement_risk_m(
                actions[model_index], state_payload, args.planned_interval_s
            )
            if (
                args.fallback_age_only
                and model_index != baseline_index
                and predicted_margin <= args.override_threshold
            ):
                selected_index = baseline_index
            elif (
                args.max_stale_displacement_m is not None
                and model_index != baseline_index
                and motion_risk_m > args.max_stale_displacement_m
            ):
                selected_index = baseline_index
                motion_guard_fallback_count += 1
            else:
                selected_index = model_index
            learned_override_count += int(selected_index != baseline_index)
            score_values = [float(value) for value in scores.squeeze(0).cpu()]
        selected = actions[selected_index]
        actual_next_interval_s = 0.0
        sample = nusc.get("sample", token)
        if sample["next"]:
            next_timestamp = int(nusc.get("sample", sample["next"])["timestamp"])
            actual_next_interval_s = max(
                0.0, (next_timestamp - timestamp) / 1_000_000.0
            )
        actual_deadline_violating_cameras = [
            camera
            for camera, age_s, source_timestamp in zip(
                CAMERAS,
                state_payload["ages_s"],
                state_payload["source_timestamps"],
            )
            if source_timestamp >= 0
            and camera not in selected
            and age_s + actual_next_interval_s > args.max_age_s + 0.01
        ]
        actual_deadline_violation_frames += int(
            bool(actual_deadline_violating_cameras)
        )
        actual_deadline_violating_camera_events += len(
            actual_deadline_violating_cameras
        )
        routing_times_ms.append((time.perf_counter() - route_started) * 1000.0)
        propagation_merge_started = time.perf_counter()
        frame_predictions, memory_ages = _build_predictions(
            legacy, token, timestamp, selected, memory, caches, args
        )
        propagation_merge_times_ms.append(
            (time.perf_counter() - propagation_merge_started) * 1000.0
        )
        predictions[token] = frame_predictions
        all_memory_ages.extend(memory_ages)
        selections[token] = list(selected)
        route_rows.append({
            "sample_token": token,
            "scene_token": scene_token,
            "selected_index": selected_index,
            "selected": list(selected),
            "feasible_count": len([value for value in feasible if value]),
            "emergency_override": emergency,
            "actual_next_interval_s": actual_next_interval_s,
            "actual_deadline_violating_cameras": actual_deadline_violating_cameras,
            "max_pre_action_age_s": max(state_payload["ages_s"]),
            "switch_fraction": switch_fraction(selected, previous_action),
            "scores": score_values,
            "predicted_advantage_margin": predicted_margin,
            "selected_stale_displacement_risk_m": (
                _stale_displacement_risk_m(selected, state_payload, args.planned_interval_s)
                if args.policy == "learned" and not emergency
                else None
            ),
        })
        for camera in selected:
            memory[camera] = (timestamp, caches[camera][token])
        previous_action = selected
        if frame_number % 500 == 0:
            print(f"[PROGRESS] frames={frame_number}/{len(ordered_frames)}", flush=True)

    method_label = (
        f"{learned_model_type}_{learned_target_mode}"
        if args.policy == "learned"
        else args.policy
    )
    fallback_label = (
        f"_fallback{args.override_threshold:g}" if args.fallback_age_only else ""
    )
    memory_label = "" if args.memory_mode == "motion" else f"_{args.memory_mode}_memory"
    method = (
        f"{method_label}{fallback_label}{memory_label}"
        f"_k{args.camera_count}_exact_bootstrap"
    )
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
    result_json_path = Path(args.result_json).resolve() if args.result_json else None
    result_json_sha256 = None
    if result_json_path is not None:
        if set(predictions) != expected:
            raise RuntimeError(
                f"result token mismatch: expected={len(expected)} actual={len(predictions)}"
            )
        result_json_sha256 = write_result_json(result_json_path, predictions)
    checkpoint_hash = None
    if args.checkpoint:
        checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    cache_manifest_path = (
        Path(args.cache_manifest).resolve() if args.cache_manifest else None
    )
    cache_manifest_sha256 = None
    if cache_manifest_path is not None:
        if not cache_manifest_path.is_file():
            raise FileNotFoundError(cache_manifest_path)
        cache_manifest_sha256 = hashlib.sha256(
            cache_manifest_path.read_bytes()
        ).hexdigest()
    if args.memory_mode == "none":
        memory_description = "no unselected-camera memory"
    elif args.memory_mode == "static":
        memory_description = "global-frame static sparse detections"
    elif args.memory_tau_s >= 1.0e8:
        memory_description = (
            "global-frame constant-velocity propagation without score decay"
        )
    else:
        memory_description = (
            "global-frame constant-velocity propagation plus exponential score decay"
        )
    diagnostics = {
        "method": method,
        "policy": args.policy,
        "router_model_type": learned_model_type,
        "router_target_mode": learned_target_mode,
        "eval_set": args.eval_set,
        "frame_count": len(ordered_frames),
        "scene_count": len(split_names),
        "camera_count_per_frame": args.camera_count,
        "exact_camera_passes": len(ordered_frames) * args.camera_count,
        "free_full6_warm_start": False,
        "emergency_overrides": emergency_overrides,
        "shield_violation_rate": emergency_overrides / max(1, len(ordered_frames)),
        "scheduler_infeasible_decisions": emergency_overrides,
        "actual_deadline_violation_frames": actual_deadline_violation_frames,
        "actual_deadline_violation_rate": (
            actual_deadline_violation_frames / max(1, len(ordered_frames))
        ),
        "actual_deadline_violating_camera_events": (
            actual_deadline_violating_camera_events
        ),
        "learned_override_count": learned_override_count,
        "learned_override_rate": learned_override_count / max(1, len(ordered_frames)),
        "fallback_age_only": args.fallback_age_only,
        "override_threshold": args.override_threshold,
        "max_stale_displacement_m": args.max_stale_displacement_m,
        "motion_guard_fallback_count": motion_guard_fallback_count,
        "motion_guard_fallback_rate": motion_guard_fallback_count / max(1, len(ordered_frames)),
        "memory_age_mean_s": statistics.fmean(all_memory_ages) if all_memory_ages else 0.0,
        "memory_age_p95_s": _percentile(all_memory_ages, 0.95),
        "memory_age_max_s": max(all_memory_ages, default=0.0),
        "mean_switch_fraction": statistics.fmean(row["switch_fraction"] for row in route_rows),
        "routing_latency_mean_ms": statistics.fmean(routing_times_ms),
        "routing_latency_p95_ms": _percentile(routing_times_ms, 0.95),
        "propagation_merge_latency_mean_ms": statistics.fmean(
            propagation_merge_times_ms
        ),
        "propagation_merge_latency_p95_ms": _percentile(
            propagation_merge_times_ms, 0.95
        ),
        "router_parameter_count": checkpoint.get("parameter_count") if checkpoint else 0,
        "elapsed_seconds_total": time.time() - started,
        "checkpoint_sha256": checkpoint_hash,
        "cache_manifest": str(cache_manifest_path) if cache_manifest_path else None,
        "cache_manifest_sha256": cache_manifest_sha256,
        "metrics": metrics,
        "result_json": str(result_json_path) if result_json_path else None,
        "result_json_sha256": result_json_sha256,
        "protocol": {
            "dataset_version": args.version,
            "router_inputs": "past sparse detection memory, ages, camera identity, previous mask, budget",
            "uses_current_unselected_detection_or_gt": False,
            "memory": memory_description,
            "planned_interval_s": args.planned_interval_s,
            "max_age_s": args.max_age_s,
            "memory_tau_s": args.memory_tau_s,
            "memory_mode": args.memory_mode,
            "velocity_adaptive_decay_scale_mps": args.velocity_adaptive_decay_scale_mps,
            "merge_iou": args.merge_iou,
        },
    }
    (output_dir / "router_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output_dir / "selections.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8"
    )
    (output_dir / "route_trace.json").write_text(
        json.dumps(route_rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
