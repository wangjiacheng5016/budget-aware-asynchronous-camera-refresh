#!/usr/bin/env python3
"""Evaluate routing robustness to frame drops, budget bursts, and camera outages.

All perturbations are deterministic and reset at scene boundaries so policies
can be compared on exactly paired frames.  Frame drops retain a prediction for
the official split by propagating only past memory; no current camera cache is
read on a dropped frame.
"""
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
    from .robustness_protocol import (
        availability_at_step,
        deadline_violations_after_action,
        parse_capture_sequence,
        parse_outage,
    )
    from .router_core import build_pre_action_state, enumerate_actions, feasible_actions, switch_fraction
    from .variable_budget_schedule import budget_at_step, parse_budget_sequence
except ImportError:  # Direct script execution on the experiment server.
    from activecam_core import CAMERAS
    from eval_router import _age_priority_index, _build_predictions, _percentile
    from mini_preflight import evaluate_predictions, load_dependencies, ordered_split_frames
    from robustness_protocol import (
        availability_at_step,
        deadline_violations_after_action,
        parse_capture_sequence,
        parse_outage,
    )
    from router_core import build_pre_action_state, enumerate_actions, feasible_actions, switch_fraction
    from variable_budget_schedule import budget_at_step, parse_budget_sequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--policy", choices=("age_only", "random_safe"), default="age_only")
    parser.add_argument("--budget-sequence", type=parse_budget_sequence, default=(3,))
    parser.add_argument(
        "--capture-sequence",
        type=parse_capture_sequence,
        default=(True,),
        help="Cyclic 1=capture, 0=drop sequence reset at each scene.",
    )
    parser.add_argument(
        "--outage",
        type=parse_outage,
        action="append",
        default=[],
        help="Repeatable CAMERA:START:DURATION[:PERIOD] outage specification.",
    )
    parser.add_argument(
        "--interval-policy",
        choices=("fixed", "observed_previous"),
        default="observed_previous",
        help="Causal shield look-ahead interval; observed_previous uses only the last measured dt.",
    )
    parser.add_argument("--planned-interval-s", type=float, default=0.50)
    parser.add_argument("--phase-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-age-s", type=float, default=1.50)
    parser.add_argument("--deadline-tolerance-s", type=float, default=0.01)
    parser.add_argument("--uncertainty-tau-s", type=float, default=1.0)
    parser.add_argument(
        "--memory-tau-s",
        type=float,
        default=1.0e9,
        help="Frozen motion-only protocol uses 1e9 (effectively no score decay).",
    )
    parser.add_argument("--memory-mode", choices=("motion", "static", "none"), default="motion")
    parser.add_argument("--velocity-adaptive-decay-scale-mps", type=float)
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args()


def _validate_scenario_name(value: str) -> str:
    if not value or any(not (char.isalnum() or char in "_-") for char in value):
        raise ValueError("scenario-name may contain only letters, digits, '_' and '-'")
    return value


def main() -> None:
    args = parse_args()
    scenario_name = _validate_scenario_name(args.scenario_name)
    if (
        args.planned_interval_s <= 0.0
        or args.max_age_s <= 0.0
        or args.deadline_tolerance_s < 0.0
    ):
        raise ValueError("planned interval and max age must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
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
    memory_ages = []
    observed_intervals = []
    planned_intervals = []
    budget_counts: Counter[int] = Counter()
    effective_budget_counts: Counter[int] = Counter()
    unavailable_counts: Counter[str] = Counter()
    rng = random.Random(args.seed)
    active_scene = None
    scene_frame_index = 0
    memory = {}
    previous_action: tuple[str, ...] = ()
    previous_timestamp = None
    emergency_overrides = 0
    scheduler_infeasible_decisions = 0
    deadline_violation_frames = 0
    deadline_violating_camera_frames = 0
    dropped_frames = 0
    budget_shortfall_passes = 0
    started = time.time()

    for frame_number, (token, scene_token) in enumerate(ordered_frames, 1):
        if scene_token != active_scene:
            active_scene = scene_token
            scene_frame_index = 0
            memory = {}
            previous_action = ()
            previous_timestamp = None
        timestamp = int(nusc.get("sample", token)["timestamp"])
        observed_interval_s = (
            args.planned_interval_s
            if previous_timestamp is None
            else max(0.0, (timestamp - previous_timestamp) / 1_000_000.0)
        )
        if previous_timestamp is not None:
            observed_intervals.append(observed_interval_s)
        shield_interval_s = (
            observed_interval_s
            if args.interval_policy == "observed_previous"
            else args.planned_interval_s
        )
        planned_intervals.append(shield_interval_s)
        requested_budget = budget_at_step(
            args.budget_sequence, scene_frame_index, args.phase_offset
        )
        capture_available = args.capture_sequence[
            scene_frame_index % len(args.capture_sequence)
        ]
        availability = availability_at_step(scene_frame_index, tuple(args.outage))
        for camera, is_available in availability.items():
            if not is_available:
                unavailable_counts[camera] += 1
        available_cameras = tuple(
            camera for camera in CAMERAS if availability[camera]
        )
        effective_budget = (
            min(requested_budget, len(available_cameras)) if capture_available else 0
        )
        dropped_frames += int(not capture_available)
        budget_shortfall_passes += requested_budget - effective_budget
        budget_counts[requested_budget] += 1
        effective_budget_counts[effective_budget] += 1

        state_payload = build_pre_action_state(
            memory=memory,
            decision_timestamp=timestamp,
            previous_action=previous_action,
            max_age_s=args.max_age_s,
            uncertainty_tau_s=args.uncertainty_tau_s,
            availability=availability,
            strict_past=True,
        )
        emergency = False
        feasible_count = 0
        if effective_budget == 0:
            selected: tuple[str, ...] = ()
            selected_index = None
        else:
            actions = enumerate_actions(effective_budget, cameras=available_cameras)
            initialized = [value >= 0 for value in state_payload["source_timestamps"]]
            feasible = feasible_actions(
                actions=actions,
                ages_s=state_payload["ages_s"],
                planned_interval_s=shield_interval_s,
                max_age_s=args.max_age_s,
                availability=[availability[camera] for camera in CAMERAS],
                initialized=initialized,
            )
            valid_indices = [index for index, value in enumerate(feasible) if value]
            feasible_count = len(valid_indices)
            if not valid_indices:
                emergency = True
                emergency_overrides += 1
                scheduler_infeasible_decisions += 1
                valid_indices = list(range(len(actions)))
            if args.policy == "random_safe" and not emergency:
                selected_index = rng.choice(valid_indices)
            else:
                selected_index = _age_priority_index(
                    actions, state_payload["ages_s"], valid_indices
                )
            selected = actions[selected_index]

        frame_predictions, used_ages = _build_predictions(
            legacy, token, timestamp, selected, memory, caches, args
        )
        predictions[token] = frame_predictions
        selections[token] = list(selected)
        memory_ages.extend(used_ages)
        available_ages = [
            age
            for camera, age in zip(CAMERAS, state_payload["ages_s"])
            if availability[camera]
        ]
        initialized_before = [
            value >= 0 for value in state_payload["source_timestamps"]
        ]
        sample = nusc.get("sample", token)
        actual_next_interval_s = 0.0
        if sample["next"]:
            actual_next_timestamp = int(
                nusc.get("sample", sample["next"])["timestamp"]
            )
            actual_next_interval_s = max(
                0.0, (actual_next_timestamp - timestamp) / 1_000_000.0
            )
        # This audit deliberately includes cameras that are currently in
        # outage. The shield cannot select them, but the resulting loss of the
        # deadline guarantee must remain visible in the paper table.
        deadline_violating_cameras = deadline_violations_after_action(
            ages_s=state_payload["ages_s"],
            initialized=initialized_before,
            selected=selected,
            next_interval_s=actual_next_interval_s,
            max_age_s=args.max_age_s,
            tolerance_s=args.deadline_tolerance_s,
        )
        deadline_violation_frames += int(bool(deadline_violating_cameras))
        deadline_violating_camera_frames += len(deadline_violating_cameras)
        route_rows.append({
            "sample_token": token,
            "scene_token": scene_token,
            "scene_frame_index": scene_frame_index,
            "requested_budget": requested_budget,
            "effective_budget": effective_budget,
            "capture_available": capture_available,
            "available_cameras": list(available_cameras),
            "selected_index": selected_index,
            "selected": list(selected),
            "feasible_count": feasible_count,
            "emergency_override": emergency,
            "observed_previous_interval_s": observed_interval_s,
            "shield_planned_interval_s": shield_interval_s,
            "actual_next_interval_s": actual_next_interval_s,
            "max_available_pre_action_age_s": max(available_ages, default=0.0),
            "deadline_violation_after_action": bool(deadline_violating_cameras),
            "deadline_violating_cameras": list(deadline_violating_cameras),
            "switch_fraction": switch_fraction(selected, previous_action),
        })
        for camera in selected:
            memory[camera] = (timestamp, caches[camera][token])
        previous_action = selected
        previous_timestamp = timestamp
        scene_frame_index += 1
        if frame_number % 500 == 0:
            print(f"[PROGRESS] frames={frame_number}/{len(ordered_frames)}", flush=True)

    sequence_label = "-".join(str(value) for value in args.budget_sequence)
    method = f"robust_{scenario_name}_{args.policy}_k{sequence_label}"
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
    exact_camera_passes = sum(row["effective_budget"] for row in route_rows)
    diagnostics = {
        "method": method,
        "policy": args.policy,
        "scenario_name": scenario_name,
        "eval_set": args.eval_set,
        "frame_count": len(ordered_frames),
        "scene_count": len(split_names),
        "budget_sequence": list(args.budget_sequence),
        "budget_phase_offset": args.phase_offset % len(args.budget_sequence),
        "capture_sequence": [int(value) for value in args.capture_sequence],
        "outages": [
            {
                "camera": spec.camera,
                "start": spec.start,
                "duration": spec.duration,
                "period": spec.period,
            }
            for spec in args.outage
        ],
        "requested_budget_counts": dict(sorted(budget_counts.items())),
        "effective_budget_counts": dict(sorted(effective_budget_counts.items())),
        "average_requested_camera_count": sum(
            row["requested_budget"] for row in route_rows
        ) / max(1, len(route_rows)),
        "average_effective_camera_count": exact_camera_passes / max(1, len(route_rows)),
        "exact_camera_passes": exact_camera_passes,
        "budget_shortfall_passes": budget_shortfall_passes,
        "capture_drop_frames": dropped_frames,
        "capture_drop_rate": dropped_frames / max(1, len(route_rows)),
        "camera_unavailable_frame_counts": dict(sorted(unavailable_counts.items())),
        "emergency_overrides": emergency_overrides,
        "scheduler_infeasible_decisions": scheduler_infeasible_decisions,
        "scheduler_infeasible_rate": scheduler_infeasible_decisions / max(1, len(route_rows)),
        "shield_violation_rate": emergency_overrides / max(1, len(route_rows)),
        "deadline_violation_frames": deadline_violation_frames,
        "deadline_violation_rate": deadline_violation_frames / max(1, len(route_rows)),
        "deadline_violating_camera_frames": deadline_violating_camera_frames,
        "memory_age_mean_s": statistics.fmean(memory_ages) if memory_ages else 0.0,
        "memory_age_p95_s": _percentile(memory_ages, 0.95),
        "memory_age_max_s": max(memory_ages, default=0.0),
        "observed_interval_mean_s": statistics.fmean(observed_intervals) if observed_intervals else 0.0,
        "observed_interval_p95_s": _percentile(observed_intervals, 0.95),
        "observed_interval_max_s": max(observed_intervals, default=0.0),
        "mean_switch_fraction": statistics.fmean(
            row["switch_fraction"] for row in route_rows
        ),
        "free_full6_warm_start": False,
        "metrics": metrics,
        "elapsed_seconds_total": time.time() - started,
        "protocol": {
            "perturbation_reset": "each scene",
            "frame_drop_semantics": "no current cache read; predict from strictly past memory",
            "camera_outage_semantics": "unavailable camera cannot refresh but its past memory is retained",
            "budget_outage_semantics": "requested budget is a maximum; effective K is capped by available cameras",
            "interval_policy": args.interval_policy,
            "uses_future_interval": False,
            "deadline_audit": (
                "ex-post actual next timestamp; not used by the causal scheduler"
            ),
            "uses_current_unselected_detection_or_gt": False,
            "memory_mode": args.memory_mode,
            "memory_tau_s": args.memory_tau_s,
            "memory_rule": (
                "constant-velocity motion propagation without practical score decay"
                if args.memory_mode == "motion" and args.memory_tau_s >= 1.0e8
                else "non-frozen custom memory rule"
            ),
            "max_age_s": args.max_age_s,
            "deadline_tolerance_s": args.deadline_tolerance_s,
            "deadline_audit_scope": (
                "all initialized cameras, including outage cameras; age after action plus causal next interval"
            ),
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
