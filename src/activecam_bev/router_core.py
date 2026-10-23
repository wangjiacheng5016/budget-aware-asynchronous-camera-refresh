from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .activecam_core import CAMERAS, refresh_deadline_exceeded
except ImportError:  # Direct script/module execution on the experiment server.
    from activecam_core import CAMERAS, refresh_deadline_exceeded


DETECTION_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)

STATE_FEATURES = (
    "age_ratio",
    "log_box_count",
    "mean_score",
    "max_score",
    *(f"class_fraction_{name}" for name in DETECTION_CLASSES),
    "mean_speed_ratio",
    "max_speed_ratio",
    "uncertainty",
    "previous_selected",
    "initialized",
    "available",
)


def enumerate_actions(
    camera_count: int,
    cameras: Sequence[str] = CAMERAS,
) -> tuple[tuple[str, ...], ...]:
    if not 1 <= camera_count <= len(cameras):
        raise ValueError("camera_count must be between one and the camera count")
    return tuple(itertools.combinations(cameras, camera_count))


def action_mask(
    action: Sequence[str],
    cameras: Sequence[str] = CAMERAS,
) -> tuple[int, ...]:
    chosen = set(action)
    if not chosen.issubset(cameras):
        raise ValueError("action contains an unknown camera")
    return tuple(int(camera in chosen) for camera in cameras)


def detection_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    age_s: float,
    max_age_s: float,
    uncertainty_tau_s: float,
) -> dict[str, Any]:
    if age_s < 0 or max_age_s <= 0 or uncertainty_tau_s <= 0:
        raise ValueError("ages and uncertainty time constant are invalid")
    count = len(records)
    scores = [float(record.get("detection_score", 0.0)) for record in records]
    speeds = []
    histogram = {name: 0 for name in DETECTION_CLASSES}
    for record in records:
        velocity = record.get("velocity", (0.0, 0.0))
        vx, vy = float(velocity[0]), float(velocity[1])
        speeds.append(math.hypot(vx, vy))
        name = str(record.get("detection_name", ""))
        if name in histogram:
            histogram[name] += 1
    mean_score = sum(scores) / count if count else 0.0
    max_score = max(scores, default=0.0)
    mean_speed = sum(speeds) / count if count else 0.0
    max_speed = max(speeds, default=0.0)
    class_fraction = [histogram[name] / max(1, count) for name in DETECTION_CLASSES]
    age_uncertainty = 1.0 - math.exp(-age_s / uncertainty_tau_s)
    if count:
        uncertainty = min(
            2.0,
            (1.0 - mean_score)
            + 0.25 * min(1.0, mean_speed / 10.0)
            + 0.50 * age_uncertainty,
        )
    else:
        # An empty detector memory is not evidence that the view is safe.
        uncertainty = min(2.0, 1.0 + 0.50 * age_uncertainty)
    return {
        "box_count": count,
        "mean_score": mean_score,
        "max_score": max_score,
        "class_fraction": class_fraction,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "uncertainty": uncertainty,
        "features": [
            min(age_s / max_age_s, 4.0),
            math.log1p(count) / math.log(501.0),
            mean_score,
            max_score,
            *class_fraction,
            min(mean_speed / 20.0, 2.0),
            min(max_speed / 30.0, 2.0),
            uncertainty,
        ],
    }


def build_pre_action_state(
    *,
    memory: Mapping[str, tuple[int, Sequence[Mapping[str, Any]]]],
    decision_timestamp: int,
    previous_action: Sequence[str],
    max_age_s: float,
    uncertainty_tau_s: float,
    availability: Mapping[str, bool] | None = None,
    strict_past: bool = True,
    cameras: Sequence[str] = CAMERAS,
) -> dict[str, Any]:
    """Build deployable state using only memory that existed before the action."""
    available = availability or {camera: True for camera in cameras}
    previous = set(previous_action)
    rows = []
    ages = []
    source_timestamps = []
    uncertainties = []
    for camera in cameras:
        initialized = camera in memory
        if initialized:
            source_timestamp, records = memory[camera]
            if source_timestamp > decision_timestamp:
                raise AssertionError("memory comes from the future")
            if strict_past and source_timestamp >= decision_timestamp:
                raise AssertionError("router state includes a current-frame cache")
            age_s = max(0.0, (decision_timestamp - source_timestamp) / 1_000_000.0)
            summary = detection_summary(
                records,
                age_s=age_s,
                max_age_s=max_age_s,
                uncertainty_tau_s=uncertainty_tau_s,
            )
            base_features = summary["features"]
            uncertainty = float(summary["uncertainty"])
        else:
            source_timestamp = -1
            age_s = max_age_s
            summary = None
            base_features = detection_summary(
                [],
                age_s=age_s,
                max_age_s=max_age_s,
                uncertainty_tau_s=uncertainty_tau_s,
            )["features"]
            uncertainty = 2.0
        rows.append([
            *base_features,
            float(camera in previous),
            float(initialized),
            float(bool(available.get(camera, False))),
        ])
        ages.append(age_s)
        source_timestamps.append(source_timestamp)
        uncertainties.append(uncertainty)
    if rows and len(rows[0]) != len(STATE_FEATURES):
        raise AssertionError("state feature schema is inconsistent")
    return {
        "state": rows,
        "ages_s": ages,
        "uncertainties": uncertainties,
        "source_timestamps": source_timestamps,
    }


def feasible_actions(
    *,
    actions: Sequence[Sequence[str]],
    ages_s: Sequence[float],
    planned_interval_s: float,
    max_age_s: float,
    availability: Sequence[bool] | None = None,
    initialized: Sequence[bool] | None = None,
    timestamp_tolerance_s: float = 0.01,
    cameras: Sequence[str] = CAMERAS,
) -> tuple[bool, ...]:
    if len(ages_s) != len(cameras):
        raise ValueError("one age is required per camera")
    available = tuple(availability) if availability is not None else (True,) * len(cameras)
    if len(available) != len(cameras):
        raise ValueError("one availability flag is required per camera")
    is_initialized = (
        tuple(initialized) if initialized is not None else (True,) * len(cameras)
    )
    if len(is_initialized) != len(cameras):
        raise ValueError("one initialization flag is required per camera")
    unseen = {
        camera
        for camera, is_up, was_seen in zip(cameras, available, is_initialized)
        if is_up and not was_seen
    }
    output = []
    for action in actions:
        selected = set(action)
        # An unavailable camera cannot be selected and is exempt from refresh debt.
        valid = all(
            camera not in selected
            for camera, is_up in zip(cameras, available)
            if not is_up
        )
        if valid and unseen:
            # Exact-budget bootstrap: refresh as many never-seen views as possible.
            valid = len(selected.intersection(unseen)) == min(len(selected), len(unseen))
        if valid:
            valid = all(
                not refresh_deadline_exceeded(
                    age_s=ages_s[index],
                    next_interval_s=planned_interval_s,
                    max_age_s=max_age_s,
                    timestamp_tolerance_s=timestamp_tolerance_s,
                )
                for index, camera in enumerate(cameras)
                if available[index] and is_initialized[index] and camera not in selected
            )
        output.append(valid)
    return tuple(output)


def uncertainty_weighted_age_debt(
    *,
    action: Sequence[str],
    ages_s: Sequence[float],
    uncertainties: Sequence[float],
    planned_interval_s: float,
    soft_age_s: float,
    exponent: float = 2.0,
    cameras: Sequence[str] = CAMERAS,
) -> float:
    if len(ages_s) != len(cameras) or len(uncertainties) != len(cameras):
        raise ValueError("age and uncertainty vectors must match the cameras")
    if planned_interval_s < 0 or soft_age_s <= 0 or exponent <= 0:
        raise ValueError("invalid debt parameters")
    selected = set(action)
    debt = 0.0
    for camera, age_s, uncertainty in zip(cameras, ages_s, uncertainties):
        next_age = planned_interval_s if camera in selected else age_s + planned_interval_s
        debt += max(0.0, uncertainty) * (next_age / soft_age_s) ** exponent
    return debt / len(cameras)


def switch_fraction(
    action: Sequence[str],
    previous_action: Sequence[str],
) -> float:
    chosen = set(action)
    previous = set(previous_action)
    denominator = max(1, len(chosen) + len(previous))
    return len(chosen.symmetric_difference(previous)) / denominator


def teacher_q_values(
    *,
    actions: Sequence[Sequence[str]],
    current_utilities: Sequence[float],
    feasible: Sequence[bool],
    ages_s: Sequence[float],
    uncertainties: Sequence[float],
    previous_action: Sequence[str],
    planned_interval_s: float,
    soft_age_s: float,
    age_weight: float,
    switch_weight: float,
    cameras: Sequence[str] = CAMERAS,
) -> tuple[list[float], list[float]]:
    if not (len(actions) == len(current_utilities) == len(feasible)):
        raise ValueError("candidate vectors have different lengths")
    q_values, debts = [], []
    for action, utility, is_feasible in zip(actions, current_utilities, feasible):
        debt = uncertainty_weighted_age_debt(
            action=action,
            ages_s=ages_s,
            uncertainties=uncertainties,
            planned_interval_s=planned_interval_s,
            soft_age_s=soft_age_s,
            cameras=cameras,
        )
        q = (
            float(utility)
            - age_weight * debt
            - switch_weight * switch_fraction(action, previous_action)
        )
        q_values.append(q if is_feasible else float("-inf"))
        debts.append(debt)
    return q_values, debts


def best_feasible_action(q_values: Sequence[float]) -> int:
    candidates = [index for index, value in enumerate(q_values) if math.isfinite(value)]
    if not candidates:
        raise RuntimeError("the hard safety shield has no feasible action")
    return max(candidates, key=lambda index: (q_values[index], -index))
