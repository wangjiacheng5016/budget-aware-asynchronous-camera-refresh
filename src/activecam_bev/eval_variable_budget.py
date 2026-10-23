#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

try:
    from .activecam_core import CAMERAS
    from .eval_router import _age_priority_index, _build_predictions, _percentile
    from .mini_preflight import evaluate_predictions, load_dependencies, ordered_split_frames
    from .router_core import (
        build_pre_action_state,
        enumerate_actions,
        feasible_actions,
        switch_fraction,
    )
    from .robustness_protocol import (
        audit_actual_deadline_after_action,
        summarize_deadline_violations,
    )
    from .variable_budget_schedule import budget_at_step, parse_budget_sequence
except ImportError:
    from activecam_core import CAMERAS
    from eval_router import _age_priority_index, _build_predictions, _percentile
    from mini_preflight import evaluate_predictions, load_dependencies, ordered_split_frames
    from router_core import (
        build_pre_action_state,
        enumerate_actions,
        feasible_actions,
        switch_fraction,
    )
    from robustness_protocol import (
        audit_actual_deadline_after_action,
        summarize_deadline_violations,
    )
    from variable_budget_schedule import budget_at_step, parse_budget_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe variable-budget camera refresh evaluation."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--policy", choices=("age_only", "random_safe"), default="age_only")
    parser.add_argument("--budget-sequence", type=parse_budget_sequence, required=True)
    parser.add_argument(
        "--phase-offset",
        type=int,
        default=0,
        help="Cyclic phase applied independently at every scene reset.",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--planned-interval-s", type=float, default=0.5)
    parser.add_argument("--max-age-s", type=float, default=1.8)
    parser.add_argument("--deadline-tolerance-s", type=float, default=0.01)
    parser.add_argument("--uncertainty-tau-s", type=float, default=1.0)
    parser.add_argument("--memory-tau-s", type=float, default=3.0)
    parser.add_argument("--memory-mode", choices=("motion", "static", "none"), default="motion")
    parser.add_argument("--velocity-adaptive-decay-scale-mps", type=float)
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes = load_dependencies(
        Path(args.legacy_src)
    )
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=False)
    split_names = sorted(create_splits_scenes()[args.eval_set])
    if args.max_scenes is not None:
        split_names = split_names[: args.max_scenes]
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
            raise RuntimeError(f"{camera} cache is missing {len(missing)} tokens")

    predictions = {}
    selections = {}
    route_rows = []
    all_memory_ages = []
    routing_times_ms = []
    budget_counts: Counter[int] = Counter()
    active_scene = None
    scene_frame_index = 0
    memory = {}
    previous_action: tuple[str, ...] = ()
    rng = random.Random(args.seed)
    emergency_overrides = 0
    started = time.time()
    for frame_number, (token, scene_token) in enumerate(ordered_frames, 1):
        if scene_token != active_scene:
            active_scene = scene_token
            scene_frame_index = 0
            memory = {}
            previous_action = ()
        budget = budget_at_step(
            args.budget_sequence, scene_frame_index, args.phase_offset
        )
        scene_frame_index += 1
        budget_counts[budget] += 1
        actions = enumerate_actions(budget)
        sample = nusc.get("sample", token)
        timestamp = int(sample["timestamp"])
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
        if not valid_indices:
            emergency = True
            emergency_overrides += 1
            valid_indices = list(range(len(actions)))
        if args.policy == "age_only" or emergency:
            selected_index = _age_priority_index(
                actions, state_payload["ages_s"], valid_indices
            )
        else:
            selected_index = rng.choice(valid_indices)
        selected = actions[selected_index]
        routing_times_ms.append((time.perf_counter() - route_started) * 1000.0)
        next_timestamp = (
            int(nusc.get("sample", sample["next"])["timestamp"])
            if sample["next"]
            else None
        )
        actual_next_interval_s, actual_deadline_violating_cameras = (
            audit_actual_deadline_after_action(
                ages_s=state_payload["ages_s"],
                source_timestamps=state_payload["source_timestamps"],
                selected=selected,
                current_timestamp_us=timestamp,
                next_timestamp_us=next_timestamp,
                max_age_s=args.max_age_s,
                tolerance_s=args.deadline_tolerance_s,
            )
        )
        frame_predictions, memory_ages = _build_predictions(
            legacy, token, timestamp, selected, memory, caches, args
        )
        predictions[token] = frame_predictions
        selections[token] = list(selected)
        all_memory_ages.extend(memory_ages)
        route_rows.append({
            "sample_token": token,
            "scene_token": scene_token,
            "budget": budget,
            "selected_index": selected_index,
            "selected": list(selected),
            "feasible_count": sum(feasible),
            "emergency_override": emergency,
            "actual_next_interval_s": actual_next_interval_s,
            "actual_deadline_violating_cameras": list(
                actual_deadline_violating_cameras
            ),
            "max_pre_action_age_s": max(state_payload["ages_s"]),
            "switch_fraction": switch_fraction(selected, previous_action),
        })
        for camera in selected:
            memory[camera] = (timestamp, caches[camera][token])
        previous_action = selected
        if frame_number % 500 == 0:
            print(f"[PROGRESS] frames={frame_number}/{len(ordered_frames)}", flush=True)

    sequence_label = "-".join(str(value) for value in args.budget_sequence)
    normalized_phase = args.phase_offset % len(args.budget_sequence)
    method = (
        f"{args.policy}_variable_k{sequence_label}_phase{normalized_phase}"
        "_exact_bootstrap"
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
    exact_camera_passes = sum(row["budget"] for row in route_rows)
    deadline_summary = summarize_deadline_violations([
        tuple(row["actual_deadline_violating_cameras"])
        for row in route_rows
    ])
    diagnostics = {
        "method": method,
        "policy": args.policy,
        "eval_set": args.eval_set,
        "frame_count": len(ordered_frames),
        "scene_count": len(split_names),
        "budget_sequence": list(args.budget_sequence),
        "phase_offset": normalized_phase,
        "budget_counts": dict(sorted(budget_counts.items())),
        "average_camera_count": exact_camera_passes / max(1, len(ordered_frames)),
        "exact_camera_passes": exact_camera_passes,
        "free_full6_warm_start": False,
        "emergency_overrides": emergency_overrides,
        "shield_violation_rate": emergency_overrides / max(1, len(ordered_frames)),
        "scheduler_infeasible_decisions": emergency_overrides,
        **deadline_summary,
        "memory_age_mean_s": statistics.fmean(all_memory_ages) if all_memory_ages else 0.0,
        "memory_age_p95_s": _percentile(all_memory_ages, 0.95),
        "memory_age_max_s": max(all_memory_ages, default=0.0),
        "mean_switch_fraction": statistics.fmean(row["switch_fraction"] for row in route_rows),
        "routing_latency_mean_ms": statistics.fmean(routing_times_ms),
        "routing_latency_p95_ms": _percentile(routing_times_ms, 0.95),
        "elapsed_seconds_total": time.time() - started,
        "metrics": metrics,
        "protocol": {
            "router_inputs": "past sparse detection memory, ages, previous mask, current external budget",
            "uses_current_unselected_detection_or_gt": False,
            "memory_mode": args.memory_mode,
            "memory_tau_s": args.memory_tau_s,
            "planned_interval_s": args.planned_interval_s,
            "max_age_s": args.max_age_s,
            "deadline_tolerance_s": args.deadline_tolerance_s,
            "deadline_audit": (
                "ex-post actual next timestamp; not used by the causal scheduler"
            ),
            "uses_future_interval": False,
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
