#!/usr/bin/env python3
"""Fork-parallel privileged H=2 teacher gate.

Each task owns one complete scene so the router memory and previous action have
exactly the same lifetime as in ``h2_teacher_gate.py``.  Large read-only inputs
(nuScenes detector caches and ground truth) are loaded before the fork and are
therefore shared copy-on-write on Linux.  Only the parent process aggregates
predictions and invokes the official nuScenes evaluator.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from .activecam_core import (
        CAMERAS,
        local_detection_utility,
        local_nds_like_utility,
        propagate_detection,
    )
    from .h2_teacher_gate import age_priority_index, stale_displacement_risk_m
    from .mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        load_ground_truth,
        merge_sources,
        ordered_split_frames,
    )
    from .router_core import build_pre_action_state, enumerate_actions, feasible_actions
except ImportError:
    from activecam_core import (
        CAMERAS,
        local_detection_utility,
        local_nds_like_utility,
        propagate_detection,
    )
    from h2_teacher_gate import age_priority_index, stale_displacement_risk_m
    from mini_preflight import (
        evaluate_predictions,
        load_dependencies,
        load_ground_truth,
        merge_sources,
        ordered_split_frames,
    )
    from router_core import build_pre_action_state, enumerate_actions, feasible_actions


# These objects are populated in the parent immediately before creating the
# fork pool.  Workers only read them; per-scene memory remains local.
_WORKER_CONTEXT: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fork-parallel privileged H=2 residual-teacher gate; never deployable."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of fork workers. Use --workers 8 on the target server.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def group_ordered_frames(
    ordered: list[tuple[str, str]],
) -> list[tuple[int, str, tuple[str, ...]]]:
    """Partition contiguous ordered frames without changing scene/frame order."""
    grouped: list[tuple[int, str, tuple[str, ...]]] = []
    active_scene: str | None = None
    active_tokens: list[str] = []
    seen: set[str] = set()
    for token, scene_token in ordered:
        if scene_token != active_scene:
            if active_scene is not None:
                grouped.append((len(grouped), active_scene, tuple(active_tokens)))
                seen.add(active_scene)
            if scene_token in seen:
                raise ValueError(f"scene {scene_token} is not contiguous")
            active_scene = scene_token
            active_tokens = []
        active_tokens.append(token)
    if active_scene is not None:
        grouped.append((len(grouped), active_scene, tuple(active_tokens)))
    return grouped


def _context() -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context was not initialized before scene processing")
    return _WORKER_CONTEXT


def _state_payload(memory, timestamp: int, previous_action):
    context = _context()
    args = context["args"]
    return build_pre_action_state(
        memory=memory,
        decision_timestamp=timestamp,
        previous_action=previous_action,
        max_age_s=args.max_age_s,
        uncertainty_tau_s=args.uncertainty_tau_s,
        strict_past=True,
    )


def _valid_action_indices(memory, timestamp: int, previous_action):
    context = _context()
    args = context["args"]
    actions = context["actions"]
    payload = _state_payload(memory, timestamp, previous_action)
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


def _simulate(token: str, timestamp: int, action, memory):
    context = _context()
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
                    tau_s=args.memory_tau_s,
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
    utility_function = (
        local_nds_like_utility if args.proxy == "nds_like" else local_detection_utility
    )
    utility = utility_function(
        predictions,
        context["ground_truth"][token],
        score_threshold=args.proxy_score_threshold,
    )
    return predictions, updated, utility


def _continuation_value(next_token: str, updated_memory, first_action) -> float:
    if not next_token:
        return 0.0
    context = _context()
    next_timestamp = context["sample_info"][next_token]["timestamp"]
    next_payload, next_valid = _valid_action_indices(
        updated_memory, next_timestamp, first_action
    )
    next_index = age_priority_index(
        context["actions"], next_payload["ages_s"], next_valid
    )
    _predictions, _memory, utility = _simulate(
        next_token,
        next_timestamp,
        context["actions"][next_index],
        updated_memory,
    )
    return utility


def _process_scene(task: tuple[int, str, tuple[str, ...]]) -> dict[str, Any]:
    """Process one scene with fresh memory, exactly matching the serial loop."""
    scene_index, scene_token, tokens = task
    context = _context()
    args = context["args"]
    actions = context["actions"]

    predictions: dict[str, list[dict[str, Any]]] = {}
    selections: dict[str, list[str]] = {}
    trace: list[dict[str, Any]] = []
    memory = {}
    previous_action = ()
    override_advantages: list[float] = []
    candidate_advantages: list[float] = []
    override_count = 0
    motion_guard_fallback_count = 0

    for token in tokens:
        sample = context["sample_info"][token]
        timestamp = sample["timestamp"]
        payload, valid_indices = _valid_action_indices(memory, timestamp, previous_action)
        baseline_index = age_priority_index(actions, payload["ages_s"], valid_indices)
        baseline_predictions, baseline_memory, baseline_now = _simulate(
            token, timestamp, actions[baseline_index], memory
        )
        next_token = sample["next"]
        baseline_return = baseline_now + args.gamma * _continuation_value(
            next_token, baseline_memory, actions[baseline_index]
        )
        candidates = []
        for index in valid_indices:
            if index == baseline_index:
                candidates.append((0.0, index, baseline_predictions, baseline_memory))
                continue
            current_predictions, updated_memory, current_utility = _simulate(
                token, timestamp, actions[index], memory
            )
            value = current_utility + args.gamma * _continuation_value(
                next_token, updated_memory, actions[index]
            )
            candidates.append(
                (value - baseline_return, index, current_predictions, updated_memory)
            )
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
        trace.append(
            {
                "sample_token": token,
                "baseline_index": baseline_index,
                "selected_index": selected_index,
                "best_h2_advantage": best_advantage,
                "overrode_baseline": selected_index != baseline_index,
                "best_action_stale_displacement_risk_m": motion_risk_m,
                "motion_guarded": motion_guarded,
                "feasible_count": len(valid_indices),
            }
        )
        memory = selected_memory
        previous_action = actions[selected_index]

    return {
        "scene_index": scene_index,
        "scene_token": scene_token,
        "predictions": predictions,
        "selections": selections,
        "trace": trace,
        "candidate_advantages": candidate_advantages,
        "override_advantages": override_advantages,
        "override_count": override_count,
        "motion_guard_fallback_count": motion_guard_fallback_count,
    }


def merge_scene_results(
    scene_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Restore serial scene/frame insertion order and aggregate diagnostics."""
    predictions: dict[str, list[dict[str, Any]]] = {}
    selections: dict[str, list[str]] = {}
    trace: list[dict[str, Any]] = []
    candidate_advantages: list[float] = []
    override_advantages: list[float] = []
    override_count = 0
    motion_guard_fallback_count = 0
    for result in sorted(scene_results, key=lambda item: item["scene_index"]):
        predictions.update(result["predictions"])
        selections.update(result["selections"])
        trace.extend(result["trace"])
        candidate_advantages.extend(result["candidate_advantages"])
        override_advantages.extend(result["override_advantages"])
        override_count += result["override_count"]
        motion_guard_fallback_count += result["motion_guard_fallback_count"]
    return {
        "predictions": predictions,
        "selections": selections,
        "trace": trace,
        "candidate_advantages": candidate_advantages,
        "override_advantages": override_advantages,
        "override_count": override_count,
        "motion_guard_fallback_count": motion_guard_fallback_count,
    }


def main() -> None:
    global _WORKER_CONTEXT

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
    if not ordered:
        raise RuntimeError("no frames were selected for the H2 teacher gate")
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
    sample_info = {}
    for token in tokens:
        sample = nusc.get("sample", token)
        sample_info[token] = {
            "timestamp": int(sample["timestamp"]),
            "next": sample["next"],
        }
    scene_tasks = group_ordered_frames(ordered)
    _WORKER_CONTEXT = {
        "args": args,
        "legacy": legacy,
        "ground_truth": ground_truth,
        "caches": caches,
        "actions": actions,
        "sample_info": sample_info,
    }

    started = time.time()
    worker_count = min(args.workers, len(scene_tasks))
    if worker_count == 1:
        scene_results = []
        completed_frames = 0
        for task in scene_tasks:
            result = _process_scene(task)
            scene_results.append(result)
            completed_frames += len(result["trace"])
            print(
                f"[PROGRESS] {completed_frames}/{len(ordered)} "
                f"scenes={len(scene_results)}/{len(scene_tasks)}",
                flush=True,
            )
    else:
        try:
            fork_context = mp.get_context("fork")
        except ValueError as error:
            raise RuntimeError(
                "--workers > 1 requires a platform with multiprocessing fork support"
            ) from error
        scene_results = []
        completed_frames = 0
        with fork_context.Pool(processes=worker_count) as pool:
            for result in pool.imap_unordered(_process_scene, scene_tasks, chunksize=1):
                scene_results.append(result)
                completed_frames += len(result["trace"])
                print(
                    f"[PROGRESS] {completed_frames}/{len(ordered)} "
                    f"scenes={len(scene_results)}/{len(scene_tasks)}",
                    flush=True,
                )

    combined = merge_scene_results(scene_results)
    if list(combined["predictions"]) != tokens:
        raise AssertionError("parallel aggregation changed frame order or lost predictions")
    if list(combined["selections"]) != tokens:
        raise AssertionError("parallel aggregation changed selection order or lost frames")
    if [row["sample_token"] for row in combined["trace"]] != tokens:
        raise AssertionError("parallel aggregation changed trace order or lost frames")

    method = f"privileged_h2_residual_threshold{args.override_threshold:g}"
    metrics = None
    if not args.skip_official_eval:
        metrics = evaluate_predictions(
            method=method,
            predictions=combined["predictions"],
            nusc=nusc,
            config_factory=config_factory,
            NuScenesEval=NuScenesEval,
            eval_set=args.eval_set,
            output_dir=output_dir,
        )
    candidate_advantages = combined["candidate_advantages"]
    override_advantages = combined["override_advantages"]
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
        "override_count": combined["override_count"],
        "override_rate": combined["override_count"] / max(1, len(ordered)),
        "candidate_advantage_mean": statistics.fmean(candidate_advantages),
        "candidate_advantage_p50": statistics.median(candidate_advantages),
        "positive_override_advantage_mean": (
            statistics.fmean(override_advantages) if override_advantages else 0.0
        ),
        "max_stale_displacement_m": args.max_stale_displacement_m,
        "motion_guard_fallback_count": combined["motion_guard_fallback_count"],
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "workers": worker_count,
    }
    (output_dir / "teacher_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "teacher_trace.json").write_text(
        json.dumps(combined["trace"]), encoding="utf-8"
    )
    (output_dir / "selections.json").write_text(
        json.dumps(combined["selections"]), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
