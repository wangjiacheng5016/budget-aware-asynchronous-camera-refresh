#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    from .activecam_core import (
        CAMERAS,
        local_detection_utility,
        local_nds_like_utility,
        propagate_detection,
    )
    from .mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        load_ground_truth,
        merge_sources,
        ordered_split_frames,
    )
    from .router_core import STATE_FEATURES, build_pre_action_state, enumerate_actions, feasible_actions
except ImportError:
    from activecam_core import CAMERAS, local_detection_utility, local_nds_like_utility, propagate_detection
    from mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        load_ground_truth,
        merge_sources,
        ordered_split_frames,
    )
    from router_core import STATE_FEATURES, build_pre_action_state, enumerate_actions, feasible_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privileged H=2 residual-teacher gate; never a deployable result."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--override-threshold", type=float, default=0.0)
    parser.add_argument("--max-stale-displacement-m", type=float)
    parser.add_argument("--planned-interval-s", type=float, default=0.5)
    parser.add_argument("--max-age-s", type=float, default=1.5)
    parser.add_argument("--uncertainty-tau-s", type=float, default=1.0)
    parser.add_argument("--memory-tau-s", type=float, default=1.0)
    parser.add_argument("--proxy-score-threshold", type=float, default=0.05)
    parser.add_argument("--proxy", choices=("detection", "nds_like"), default="detection")
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--skip-official-eval", action="store_true")
    return parser.parse_args()


def age_priority_index(actions, ages_s, valid_indices) -> int:
    def key(index):
        selected = set(actions[index])
        next_ages = [
            0.0 if camera in selected else age
            for camera, age in zip(CAMERAS, ages_s)
        ]
        return max(next_ages), sum(next_ages), index
    return min(valid_indices, key=key)


def stale_displacement_risk_m(action, payload, planned_interval_s: float) -> float:
    selected = set(action)
    max_speed_index = STATE_FEATURES.index("max_speed_ratio")
    risks = []
    for camera, age_s, state_row in zip(CAMERAS, payload["ages_s"], payload["state"]):
        if camera in selected:
            continue
        max_speed_mps = float(state_row[max_speed_index]) * 30.0
        risks.append(max_speed_mps * (float(age_s) + planned_interval_s))
    return max(risks, default=0.0)


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
    ordered = ordered_split_frames(nusc, split_names)
    tokens = [token for token, _scene in ordered]
    ground_truth = load_ground_truth(nusc, legacy, tokens)
    cache_dir = Path(args.cache_dir)
    caches = {
        camera: legacy.read_cache(cache_dir / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    expected = set(tokens)
    for camera, cache in caches.items():
        missing = expected.difference(cache)
        if missing:
            raise RuntimeError(f"{camera} is missing {len(missing)} tokens")
    actions = enumerate_actions(args.camera_count)

    def state_payload(memory, timestamp, previous_action):
        return build_pre_action_state(
            memory=memory,
            decision_timestamp=timestamp,
            previous_action=previous_action,
            max_age_s=args.max_age_s,
            uncertainty_tau_s=args.uncertainty_tau_s,
            strict_past=True,
        )

    def valid_action_indices(memory, timestamp, previous_action):
        payload = state_payload(memory, timestamp, previous_action)
        initialized = [value >= 0 for value in payload["source_timestamps"]]
        valid = feasible_actions(
            actions=actions,
            ages_s=payload["ages_s"],
            planned_interval_s=args.planned_interval_s,
            max_age_s=args.max_age_s,
            initialized=initialized,
        )
        indices = [index for index, value in enumerate(valid) if value]
        if not indices:
            raise RuntimeError("H2 teacher encountered an infeasible shield state")
        return payload, indices

    def simulate(token, timestamp, action, memory):
        selected = set(action)
        sources = []
        updated = dict(memory)
        for camera in CAMERAS:
            if camera in selected:
                records = caches[camera][token]
                updated[camera] = (timestamp, records)
            elif camera in memory:
                source_timestamp, stored = memory[camera]
                age_s = max(0.0, (timestamp - source_timestamp) / 1_000_000.0)
                records = [
                    propagate_detection(
                        record,
                        age_s=age_s,
                        tau_s=args.memory_tau_s,
                        use_velocity=True,
                    )
                    for record in stored
                ]
            else:
                records = []
            sources.append((camera, records))
        predictions = merge_sources(
            legacy, token, sources, args.merge_iou, args.max_boxes
        )
        utility_function = (
            local_nds_like_utility if args.proxy == "nds_like" else local_detection_utility
        )
        utility = utility_function(
            predictions,
            ground_truth[token],
            score_threshold=args.proxy_score_threshold,
        )
        return predictions, updated, utility

    predictions = {}
    selections = {}
    trace = []
    memory = {}
    previous_action = ()
    active_scene = None
    override_advantages = []
    candidate_advantages = []
    override_count = 0
    motion_guard_fallback_count = 0
    started = time.time()
    for frame_index, (token, scene_token) in enumerate(ordered):
        if scene_token != active_scene:
            active_scene = scene_token
            memory = {}
            previous_action = ()
        sample = nusc.get("sample", token)
        timestamp = int(sample["timestamp"])
        payload, valid_indices = valid_action_indices(memory, timestamp, previous_action)
        baseline_index = age_priority_index(actions, payload["ages_s"], valid_indices)
        baseline_predictions, baseline_memory, baseline_now = simulate(
            token, timestamp, actions[baseline_index], memory
        )
        next_token = sample["next"]

        def continuation_value(updated_memory, first_action):
            if not next_token:
                return 0.0
            next_sample = nusc.get("sample", next_token)
            next_timestamp = int(next_sample["timestamp"])
            next_payload, next_valid = valid_action_indices(
                updated_memory, next_timestamp, first_action
            )
            next_index = age_priority_index(actions, next_payload["ages_s"], next_valid)
            _predictions, _memory, utility = simulate(
                next_token,
                next_timestamp,
                actions[next_index],
                updated_memory,
            )
            return utility

        baseline_return = baseline_now + args.gamma * continuation_value(
            baseline_memory, actions[baseline_index]
        )
        candidates = []
        for index in valid_indices:
            if index == baseline_index:
                candidates.append((0.0, index, baseline_predictions, baseline_memory))
                continue
            current_predictions, updated_memory, current_utility = simulate(
                token, timestamp, actions[index], memory
            )
            value = current_utility + args.gamma * continuation_value(
                updated_memory, actions[index]
            )
            candidates.append((
                value - baseline_return,
                index,
                current_predictions,
                updated_memory,
            ))
        best_advantage, best_index, best_predictions, best_memory = max(
            candidates, key=lambda item: (item[0], -item[1])
        )
        candidate_advantages.append(best_advantage)
        motion_risk_m = stale_displacement_risk_m(
            actions[best_index], payload, args.planned_interval_s
        )
        motion_guarded = (
            args.max_stale_displacement_m is not None
            and best_index != baseline_index
            and motion_risk_m > args.max_stale_displacement_m
        )
        if best_advantage > args.override_threshold and not motion_guarded:
            selected_index = best_index
            selected_predictions = best_predictions
            selected_memory = best_memory
            override_count += int(best_index != baseline_index)
            if best_index != baseline_index:
                override_advantages.append(best_advantage)
        else:
            selected_index = baseline_index
            selected_predictions = baseline_predictions
            selected_memory = baseline_memory
            motion_guard_fallback_count += int(motion_guarded)
        predictions[token] = selected_predictions
        selections[token] = list(actions[selected_index])
        trace.append({
            "sample_token": token,
            "baseline_index": baseline_index,
            "selected_index": selected_index,
            "best_h2_advantage": best_advantage,
            "overrode_baseline": selected_index != baseline_index,
            "best_action_stale_displacement_risk_m": motion_risk_m,
            "motion_guarded": motion_guarded,
            "feasible_count": len(valid_indices),
        })
        memory = selected_memory
        previous_action = actions[selected_index]
        if (frame_index + 1) % 250 == 0:
            print(f"[PROGRESS] {frame_index + 1}/{len(ordered)}", flush=True)

    method = f"privileged_h2_residual_threshold{args.override_threshold:g}"
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
    summary = {
        "method": method,
        "diagnostic_privileged_teacher": True,
        "deployable": False,
        "uses_current_and_next_ground_truth": True,
        "frame_count": len(ordered),
        "scene_count": len(split_names),
        "gamma": args.gamma,
        "proxy": args.proxy,
        "override_threshold": args.override_threshold,
        "override_count": override_count,
        "override_rate": override_count / max(1, len(ordered)),
        "candidate_advantage_mean": statistics.fmean(candidate_advantages),
        "candidate_advantage_p50": statistics.median(candidate_advantages),
        "positive_override_advantage_mean": (
            statistics.fmean(override_advantages) if override_advantages else 0.0
        ),
        "max_stale_displacement_m": args.max_stale_displacement_m,
        "motion_guard_fallback_count": motion_guard_fallback_count,
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
    }
    (output_dir / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "teacher_trace.json").write_text(
        json.dumps(trace), encoding="utf-8"
    )
    (output_dir / "selections.json").write_text(
        json.dumps(selections), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
