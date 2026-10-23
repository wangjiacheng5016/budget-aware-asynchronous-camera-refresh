#!/usr/bin/env python3
"""Privileged H=4 one-swap ceiling under the frozen final memory protocol.

This is a diagnostic teacher, never a deployable policy.  Candidate ranking
uses current/future all-camera caches and ground truth.  The emitted route still
has exactly K current camera passes; unselected current cameras are never used
in its predictions, only as privileged counterfactual teacher information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .activecam_core import CAMERAS, local_detection_utility, propagate_detection
    from .mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        load_ground_truth,
        merge_sources,
        ordered_split_frames,
    )
    from .router_core import build_pre_action_state, enumerate_actions, feasible_actions
    from .robustness_protocol import audit_actual_deadline_after_action
except ImportError:
    from activecam_core import CAMERAS, local_detection_utility, propagate_detection
    from mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        load_ground_truth,
        merge_sources,
        ordered_split_frames,
    )
    from router_core import build_pre_action_state, enumerate_actions, feasible_actions
    from robustness_protocol import audit_actual_deadline_after_action


FINAL_MEMORY_TAU_S = 1.0e9
_CONTEXT: dict[str, Any] | None = None


def one_swap_candidates(
    baseline: Sequence[str],
    feasible_action_set: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Return abstain/baseline followed by feasible actions differing by one swap."""
    baseline_tuple = tuple(baseline)
    baseline_set = set(baseline_tuple)
    output = [baseline_tuple]
    for action in feasible_action_set:
        candidate = tuple(action)
        chosen = set(candidate)
        if candidate == baseline_tuple:
            continue
        if len(baseline_set - chosen) == 1 and len(chosen - baseline_set) == 1:
            output.append(candidate)
    return tuple(output)


def future_two_step_recoverable(
    actions: Sequence[Sequence[str]],
    is_feasible: Callable[[int, tuple[str, ...]], bool],
) -> bool:
    """Whether a feasible action exists at each of the next two recovery steps."""
    return all(
        any(is_feasible(step, tuple(action)) for action in actions)
        for step in (1, 2)
    )


@dataclass(frozen=True)
class CandidateValue:
    action: tuple[str, ...]
    detection_gain: float
    velocity_harm: float
    recoverable: bool


def choose_pareto_candidate(
    candidates: Sequence[CandidateValue],
    *,
    min_detection_gain: float,
    max_velocity_harm: float,
) -> tuple[CandidateValue, tuple[CandidateValue, ...]]:
    """Choose an admissible point and report the complete non-dominated front."""
    if not candidates:
        raise ValueError("at least the abstain candidate is required")
    front = tuple(
        candidate
        for candidate in candidates
        if not any(
            other.detection_gain >= candidate.detection_gain
            and other.velocity_harm <= candidate.velocity_harm
            and (
                other.detection_gain > candidate.detection_gain
                or other.velocity_harm < candidate.velocity_harm
            )
            for other in candidates
        )
    )
    eligible = [
        item
        for item in candidates[1:]
        if item.recoverable
        and item.detection_gain >= min_detection_gain
        and item.velocity_harm <= max_velocity_harm
    ]
    chosen = max(
        eligible,
        key=lambda item: (item.detection_gain, -item.velocity_harm, item.action),
        default=candidates[0],
    )
    return chosen, front


class OverrideLimiter:
    def __init__(
        self, *, total_frames: int, cap_fraction: float, cooldown_frames: int
    ) -> None:
        if total_frames < 0 or not 0 <= cap_fraction <= 1 or cooldown_frames < 0:
            raise ValueError("invalid override limiter")
        self.maximum = math.floor(total_frames * cap_fraction)
        self.cooldown_frames = cooldown_frames
        self.override_count = 0
        self.last_override_frame: int | None = None

    def allow(self, *, frame_index: int) -> bool:
        cooldown_ok = (
            self.last_override_frame is None
            or frame_index - self.last_override_frame > self.cooldown_frames
        )
        return self.override_count < self.maximum and cooldown_ok

    def record(self, *, frame_index: int) -> None:
        if not self.allow(frame_index=frame_index):
            raise RuntimeError("override is forbidden by cooldown or cap")
        self.override_count += 1
        self.last_override_frame = frame_index


def formal_stop_gate(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    override_rate: float,
    deadline_violations: int,
    emergency_overrides: int,
) -> dict[str, Any]:
    """Apply the pre-registered formal teacher STOP gate."""
    deltas = {
        "NDS": float(metrics["NDS"]) - float(baseline["NDS"]),
        "mAP": float(metrics["mAP"]) - float(baseline["mAP"]),
        "mAVE": float(metrics["mAVE"]) - float(baseline["mAVE"]),
    }
    checks = {
        "nds_gain_at_least_0p003": deltas["NDS"] >= 0.003 - 1e-12,
        "map_gain_at_least_0p005": deltas["mAP"] >= 0.005 - 1e-12,
        "mave_harm_at_most_0p005": deltas["mAVE"] <= 0.005 + 1e-12,
        "zero_deadline_violations": deadline_violations == 0,
        "zero_emergency_overrides": emergency_overrides == 0,
        "override_rate_at_most_0p08": override_rate <= 0.08 + 1e-12,
    }
    return {
        "passed": all(checks.values()),
        "decision": "GO_TO_STUDENT_TRAINING" if all(checks.values()) else "STOP_NO_STUDENT",
        "deltas": deltas,
        "checks": checks,
    }


def local_velocity_error(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float = 0.05,
    match_distance_m: float = 4.0,
) -> float:
    """Mean velocity error of greedy class-consistent local TP matches."""
    remaining = set(range(len(ground_truth)))
    errors: list[float] = []
    for prediction in sorted(
        predictions,
        key=lambda item: float(item.get("detection_score", 0.0)),
        reverse=True,
    ):
        if float(prediction.get("detection_score", 0.0)) < score_threshold:
            continue
        px, py = (float(value) for value in prediction["translation"][:2])
        name = str(prediction.get("detection_name", ""))
        options = []
        for index in remaining:
            target = ground_truth[index]
            if str(target.get("detection_name", "")) != name:
                continue
            gx, gy = (float(value) for value in target["translation"][:2])
            distance = math.hypot(px - gx, py - gy)
            if distance <= match_distance_m:
                options.append((distance, index))
        if not options:
            continue
        _distance, match = min(options)
        remaining.remove(match)
        pv = prediction.get("velocity", (0.0, 0.0))
        tv = ground_truth[match].get("velocity", (0.0, 0.0))
        errors.append(
            math.hypot(float(pv[0]) - float(tv[0]), float(pv[1]) - float(tv[1]))
        )
    return statistics.fmean(errors) if errors else 0.0


def group_ordered_frames(
    ordered: Sequence[tuple[str, str]],
) -> list[tuple[int, str, tuple[str, ...]]]:
    grouped: list[tuple[int, str, tuple[str, ...]]] = []
    active_scene = None
    tokens: list[str] = []
    seen: set[str] = set()
    for token, scene in ordered:
        if scene != active_scene:
            if active_scene is not None:
                grouped.append((len(grouped), active_scene, tuple(tokens)))
                seen.add(active_scene)
            if scene in seen:
                raise ValueError(f"scene {scene} is not contiguous")
            active_scene, tokens = scene, []
        tokens.append(token)
    if active_scene is not None:
        grouped.append((len(grouped), active_scene, tuple(tokens)))
    return grouped


def _age_priority_index(
    actions: Sequence[Sequence[str]], ages_s: Sequence[float], valid: Sequence[int]
) -> int:
    def key(index: int):
        selected = set(actions[index])
        next_ages = [
            0.0 if camera in selected else age
            for camera, age in zip(CAMERAS, ages_s)
        ]
        return max(next_ages), sum(next_ages), index

    return min(valid, key=key)


def _ctx() -> dict[str, Any]:
    if _CONTEXT is None:
        raise RuntimeError("teacher worker context is not initialized")
    return _CONTEXT


def _valid(memory, timestamp: int, previous_action):
    context = _ctx()
    args = context["args"]
    payload = build_pre_action_state(
        memory=memory,
        decision_timestamp=timestamp,
        previous_action=previous_action,
        max_age_s=args.max_age_s,
        uncertainty_tau_s=1.0,
        strict_past=True,
    )
    mask = feasible_actions(
        actions=context["actions"],
        ages_s=payload["ages_s"],
        planned_interval_s=args.planned_interval_s,
        max_age_s=args.max_age_s,
        initialized=[value >= 0 for value in payload["source_timestamps"]],
    )
    return payload, [index for index, allowed in enumerate(mask) if allowed]


def _simulate(token: str, timestamp: int, action, memory):
    context = _ctx()
    args = context["args"]
    selected = set(action)
    sources = []
    updated = dict(memory)
    for camera in CAMERAS:
        if camera in selected:
            records = context["caches"][camera][token]
            updated[camera] = (timestamp, records)
        elif camera in memory:
            source_timestamp, stored = memory[camera]
            age_s = max(0.0, (timestamp - source_timestamp) / 1_000_000.0)
            records = [
                propagate_detection(
                    record,
                    age_s=age_s,
                    tau_s=FINAL_MEMORY_TAU_S,
                    use_velocity=True,
                )
                for record in stored
            ]
        else:
            records = []
        sources.append((camera, records))
    predictions = merge_sources(
        context["legacy"], token, sources, args.merge_iou, args.max_boxes
    )
    target = context["ground_truth"][token]
    detection = local_detection_utility(
        predictions, target, score_threshold=args.proxy_score_threshold
    )
    velocity = local_velocity_error(
        predictions, target, score_threshold=args.proxy_score_threshold
    )
    return predictions, updated, detection, velocity


def _rollout(
    tokens,
    start: int,
    first_action,
    initial_memory,
    previous_action,
    required_recovery_cameras: Sequence[str] = (),
):
    context = _ctx()
    gamma = context["args"].gamma
    memory = dict(initial_memory)
    action = tuple(first_action)
    detection_return = 0.0
    velocity_return = 0.0
    weight = 0.0
    first_predictions = None
    first_memory = None
    recovery_steps = 0
    recovered_cameras: set[str] = set()
    for offset in range(4):
        position = start + offset
        if position >= len(tokens):
            break
        token = tokens[position]
        timestamp = context["sample_info"][token]
        if offset:
            payload, valid = _valid(memory, timestamp, action)
            if not valid:
                return None
            selected_index = _age_priority_index(
                context["actions"], payload["ages_s"], valid
            )
            action = tuple(context["actions"][selected_index])
            if offset <= 2:
                recovery_steps += 1
                recovered_cameras.update(action)
        predictions, memory, detection, velocity = _simulate(
            token, timestamp, action, memory
        )
        if offset == 0:
            first_predictions, first_memory = predictions, dict(memory)
        discount = gamma**offset
        detection_return += discount * detection
        velocity_return += discount * velocity
        weight += discount
    return {
        "predictions": first_predictions,
        "memory": first_memory,
        "detection_return": detection_return / max(weight, 1e-12),
        "velocity_return": velocity_return / max(weight, 1e-12),
        "recoverable": recovery_steps == 2 and set(required_recovery_cameras).issubset(
            recovered_cameras
        ),
    }


def _process_scene(task: tuple[int, str, tuple[str, ...]]) -> dict[str, Any]:
    scene_index, scene_token, tokens = task
    context = _ctx()
    args = context["args"]
    limiter = OverrideLimiter(
        total_frames=len(tokens),
        cap_fraction=args.override_cap,
        cooldown_frames=args.cooldown_frames,
    )
    memory = {}
    previous_action: tuple[str, ...] = ()
    predictions: dict[str, list[dict[str, Any]]] = {}
    selections: dict[str, list[str]] = {}
    trace = []
    infeasible = 0
    deadline_violations = 0
    recoverability_rejections = 0
    cap_or_cooldown_rejections = 0
    selected_detection_gains = []
    selected_velocity_harms = []
    pareto_sizes = []
    max_age_seen = 0.0

    for position, token in enumerate(tokens):
        timestamp = context["sample_info"][token]
        payload, valid_indices = _valid(memory, timestamp, previous_action)
        max_age_seen = max(max_age_seen, max(payload["ages_s"], default=0.0))
        if not valid_indices:
            infeasible += 1
            valid_indices = list(range(len(context["actions"])))
        baseline_index = _age_priority_index(
            context["actions"], payload["ages_s"], valid_indices
        )
        baseline_action = tuple(context["actions"][baseline_index])
        baseline_rollout = _rollout(
            tokens, position, baseline_action, memory, previous_action
        )
        if baseline_rollout is None:
            raise RuntimeError("oldest baseline rollout became infeasible")
        feasible_set = [context["actions"][index] for index in valid_indices]
        actions = one_swap_candidates(baseline_action, feasible_set)
        values = [CandidateValue(baseline_action, 0.0, 0.0, True)]
        rollout_by_action = {baseline_action: baseline_rollout}
        for action in actions[1:]:
            rollout = _rollout(
                tokens,
                position,
                action,
                memory,
                previous_action,
                tuple(set(baseline_action) - set(action)),
            )
            recoverable = bool(rollout and rollout["recoverable"])
            if not recoverable:
                recoverability_rejections += 1
            if rollout is None:
                continue
            rollout_by_action[action] = rollout
            values.append(
                CandidateValue(
                    action,
                    rollout["detection_return"]
                    - baseline_rollout["detection_return"],
                    rollout["velocity_return"]
                    - baseline_rollout["velocity_return"],
                    recoverable,
                )
            )
        proposed, pareto = choose_pareto_candidate(
            values,
            min_detection_gain=args.min_h4_detection_gain,
            max_velocity_harm=args.max_h4_velocity_harm,
        )
        pareto_sizes.append(len(pareto))
        can_override = proposed.action != baseline_action and limiter.allow(
            frame_index=position
        )
        if proposed.action != baseline_action and not can_override:
            cap_or_cooldown_rejections += 1
        selected = proposed if can_override else values[0]
        if can_override:
            limiter.record(frame_index=position)
            selected_detection_gains.append(selected.detection_gain)
            selected_velocity_harms.append(selected.velocity_harm)
        chosen_rollout = rollout_by_action[selected.action]
        next_timestamp = (
            context["sample_info"][tokens[position + 1]]
            if position + 1 < len(tokens)
            else None
        )
        _interval, violating_cameras = audit_actual_deadline_after_action(
            ages_s=payload["ages_s"],
            source_timestamps=payload["source_timestamps"],
            selected=selected.action,
            current_timestamp_us=timestamp,
            next_timestamp_us=next_timestamp,
            max_age_s=args.max_age_s,
        )
        deadline_violations += int(bool(violating_cameras))
        predictions[token] = chosen_rollout["predictions"]
        selections[token] = list(selected.action)
        trace.append(
            {
                "sample_token": token,
                "baseline_action": list(baseline_action),
                "selected_action": list(selected.action),
                "overrode_baseline": selected.action != baseline_action,
                "h4_detection_gain": selected.detection_gain,
                "h4_velocity_harm": selected.velocity_harm,
                "one_swap_candidate_count": len(actions) - 1,
                "pareto_front_size": len(pareto),
                "pareto_front": [
                    {
                        "action": list(item.action),
                        "detection_gain": item.detection_gain,
                        "velocity_harm": item.velocity_harm,
                        "future_two_step_recoverable": item.recoverable,
                    }
                    for item in pareto
                ],
                "future_two_step_recoverable": selected.recoverable,
                "cap_or_cooldown_blocked": proposed.action != baseline_action and not can_override,
                "actual_deadline_violating_cameras": list(violating_cameras),
            }
        )
        memory = chosen_rollout["memory"]
        previous_action = selected.action
    return {
        "scene_index": scene_index,
        "scene_token": scene_token,
        "predictions": predictions,
        "selections": selections,
        "trace": trace,
        "override_count": limiter.override_count,
        "override_cap_count": limiter.maximum,
        "infeasible": infeasible,
        "deadline_violations": deadline_violations,
        "recoverability_rejections": recoverability_rejections,
        "cap_or_cooldown_rejections": cap_or_cooldown_rejections,
        "selected_detection_gains": selected_detection_gains,
        "selected_velocity_harms": selected_velocity_harms,
        "pareto_sizes": pareto_sizes,
        "max_age_seen": max_age_seen,
    }


def _load_baseline(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "NDS" in payload:
        return {name: float(payload[name]) for name in ("NDS", "mAP", "mAVE")}
    return {
        "NDS": float(payload["nd_score"]),
        "mAP": float(payload["mean_ap"]),
        "mAVE": float(payload["tp_errors"]["vel_err"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-metrics-json")
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--planned-interval-s", type=float, default=0.5)
    parser.add_argument("--max-age-s", type=float, default=1.5)
    parser.add_argument("--override-cap", type=float, default=0.08)
    parser.add_argument("--cooldown-frames", type=int, default=1)
    parser.add_argument("--min-h4-detection-gain", type=float, default=0.20)
    parser.add_argument("--max-h4-velocity-harm", type=float, default=0.0)
    parser.add_argument("--proxy-score-threshold", type=float, default=0.05)
    parser.add_argument("--merge-iou", type=float, default=0.05)
    parser.add_argument("--max-boxes", type=int, default=500)
    parser.add_argument("--skip-official-eval", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.camera_count != 3:
        parser.error("this frozen final ceiling is defined only for K=3")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.override_cap > 0.08:
        parser.error("--override-cap cannot exceed the pre-registered 0.08")
    if args.summary_only and not args.skip_official_eval:
        parser.error("--summary-only requires --skip-official-eval")
    return args


def main() -> None:
    global _CONTEXT
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.cache_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != "v1.0-trainval"
        or manifest.get("split") != "val"
        or manifest.get("expected_samples") != 6019
        or manifest.get("batch_size") != 1
        or len(manifest.get("cameras", {})) != 6
    ):
        raise RuntimeError("H4 final ceiling requires the formal FP32 batch=1 val cache")
    legacy, NuScenes, config_factory, NuScenesEval, create_splits_scenes = load_dependencies(
        Path(args.legacy_src)
    )
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=False)
    scenes = sorted(create_splits_scenes()[args.eval_set])
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    ordered = ordered_split_frames(nusc, scenes)
    tokens = [token for token, _scene in ordered]
    ground_truth = load_ground_truth(nusc, legacy, tokens)
    caches = {
        camera: legacy.read_cache(Path(args.cache_dir) / camera / "results_nusc.json")
        for camera in CAMERAS
    }
    expected = set(tokens)
    for camera, cache in caches.items():
        missing = expected.difference(cache)
        if missing:
            raise RuntimeError(f"{camera} cache is missing {len(missing)} tokens")
    _CONTEXT = {
        "args": args,
        "legacy": legacy,
        "ground_truth": ground_truth,
        "caches": caches,
        "actions": enumerate_actions(args.camera_count),
        "sample_info": {
            token: int(nusc.get("sample", token)["timestamp"]) for token in tokens
        },
    }
    tasks = group_ordered_frames(ordered)
    started = time.time()
    worker_count = min(args.workers, len(tasks))
    if worker_count == 1:
        results = [_process_scene(task) for task in tasks]
    else:
        try:
            fork = mp.get_context("fork")
        except ValueError as error:
            raise RuntimeError("parallel H4 teacher requires multiprocessing fork") from error
        with fork.Pool(worker_count) as pool:
            results = list(pool.imap_unordered(_process_scene, tasks, chunksize=1))
    results.sort(key=lambda item: item["scene_index"])
    predictions: dict[str, list[dict[str, Any]]] = {}
    selections = {}
    trace = []
    for result in results:
        predictions.update(result["predictions"])
        selections.update(result["selections"])
        trace.extend(result["trace"])
    if list(predictions) != tokens or [row["sample_token"] for row in trace] != tokens:
        raise AssertionError("scene aggregation changed the official frame order")
    method = "privileged_h4_one_swap_pareto_final_no_decay"
    metrics = None
    if not args.skip_official_eval:
        metrics = evaluate_predictions(
            method=method,
            predictions=predictions,
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output,
        )
    override_count = sum(item["override_count"] for item in results)
    detection_gains = [
        value for item in results for value in item["selected_detection_gains"]
    ]
    velocity_harms = [
        value for item in results for value in item["selected_velocity_harms"]
    ]
    deadline_violations = sum(item["deadline_violations"] for item in results)
    emergency = sum(item["infeasible"] for item in results)
    summary: dict[str, Any] = {
        "method": method,
        "protocol_name": "privileged_h4_one_swap_teacher_ceiling",
        "diagnostic_privileged_teacher": True,
        "deployable": False,
        "student_training_authorized": False,
        "uses_ground_truth_for_teacher_ranking": True,
        "teacher_uses_current_and_future_all_camera_cache": True,
        "route_predictions_use_only_selected_current_cameras": True,
        "frame_count": len(tokens),
        "scene_count": len(scenes),
        "camera_count_per_frame": 3,
        "exact_camera_passes": 3 * len(tokens),
        "free_full6_warm_start": False,
        "horizon": 4,
        "future_recoverability_steps": 2,
        "candidate_family": "oldest abstain plus feasible exact one-swap actions",
        "override_count": override_count,
        "override_rate": override_count / max(1, len(tokens)),
        "override_cap": args.override_cap,
        "override_cap_scope": "per_scene_floor_guarantees_global_cap",
        "cooldown_frames": args.cooldown_frames,
        "scheduler_infeasible_decisions": emergency,
        "emergency_overrides": emergency,
        "actual_deadline_violation_frames": deadline_violations,
        "recoverability_rejections": sum(
            item["recoverability_rejections"] for item in results
        ),
        "cap_or_cooldown_rejections": sum(
            item["cap_or_cooldown_rejections"] for item in results
        ),
        "selected_h4_detection_gain_mean": statistics.fmean(detection_gains)
        if detection_gains
        else 0.0,
        "selected_h4_detection_gain_sum": sum(detection_gains),
        "selected_h4_velocity_harm_mean": statistics.fmean(velocity_harms)
        if velocity_harms
        else 0.0,
        "selected_h4_velocity_harm_max": max(velocity_harms, default=0.0),
        "pareto_front_size_mean": statistics.fmean(
            value for item in results for value in item["pareto_sizes"]
        ),
        "memory_age_max_s": max((item["max_age_seen"] for item in results), default=0.0),
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "protocol": {
            "memory": "global-frame constant-velocity propagation without practical score decay",
            "memory_tau_s": FINAL_MEMORY_TAU_S,
            "baseline": "deadline-feasible oldest-first exact K=3",
            "merge": "production global tuple conversion, classwise BEV NMS, top-500",
            "ground_truth_scope": "privileged teacher candidate ranking only",
            "not_a_deployable_router": True,
            "cache_manifest": str(manifest_path),
            "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }
    if metrics is not None and args.baseline_metrics_json:
        baseline = _load_baseline(Path(args.baseline_metrics_json))
        summary["formal_baseline"] = baseline
        summary["formal_stop_gate"] = formal_stop_gate(
            metrics,
            baseline,
            override_rate=summary["override_rate"],
            deadline_violations=deadline_violations,
            emergency_overrides=emergency,
        )
        # This only authorizes a later, separately reviewed student experiment;
        # this privileged diagnostic never trains or exports a deployable model.
        summary["student_training_authorized"] = bool(
            summary["formal_stop_gate"]["passed"]
        )
    (output / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not args.summary_only:
        (output / "teacher_trace.json").write_text(
            json.dumps(trace), encoding="utf-8"
        )
        (output / "selections.json").write_text(
            json.dumps(selections), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
