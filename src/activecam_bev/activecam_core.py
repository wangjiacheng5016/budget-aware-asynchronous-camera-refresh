from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable, Mapping, Sequence


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

DETECTION_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def evaluation_scope(version: str, eval_set: str) -> tuple[bool, str]:
    """Describe whether an evaluation is formal full-validation evidence."""
    if version == "v1.0-trainval" and eval_set == "val":
        return False, (
            "Official nuScenes v1.0 validation split; suitable for formal evaluation "
            "when the detector and protocol remain frozen."
        )
    return True, (
        "Non-full-validation pipeline preflight; do not use its scores as a paper "
        "main result."
    )


def refresh_deadline_exceeded(
    *,
    age_s: float,
    next_interval_s: float,
    max_age_s: float,
    timestamp_tolerance_s: float = 0.01,
) -> bool:
    """Return whether leaving a camera stale would miss the next refresh deadline."""
    if min(age_s, next_interval_s, max_age_s, timestamp_tolerance_s) < 0:
        raise ValueError("age, interval, limit, and tolerance must be non-negative")
    return age_s + next_interval_s > max_age_s + timestamp_tolerance_s


def propagate_detection(
    record: Mapping[str, object],
    *,
    age_s: float,
    tau_s: float,
    use_velocity: bool,
    velocity_decay_scale_mps: float | None = None,
) -> dict[str, object]:
    """Return an age-decayed copy of a global-frame nuScenes detection."""
    if age_s < 0:
        raise ValueError("age_s must be non-negative")
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")
    if velocity_decay_scale_mps is not None and velocity_decay_scale_mps <= 0:
        raise ValueError("velocity_decay_scale_mps must be positive")
    output = copy.deepcopy(dict(record))
    translation = [float(value) for value in output["translation"]]
    velocity = [float(value) for value in output.get("velocity", (0.0, 0.0))]
    if use_velocity:
        translation[0] += velocity[0] * age_s
        translation[1] += velocity[1] * age_s
    output["translation"] = translation
    effective_tau_s = tau_s
    if velocity_decay_scale_mps is not None:
        speed_mps = math.hypot(velocity[0], velocity[1])
        effective_tau_s = tau_s / (1.0 + speed_mps / velocity_decay_scale_mps)
    output["detection_score"] = float(output["detection_score"]) * math.exp(
        -age_s / effective_tau_s
    )
    return output


def _match_counts(
    predictions: Sequence[Mapping[str, object]],
    ground_truth: Sequence[Mapping[str, object]],
    distance_threshold: float,
) -> tuple[int, int, int]:
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    ordered = sorted(
        predictions,
        key=lambda item: float(item.get("detection_score", 0.0)),
        reverse=True,
    )
    for prediction in ordered:
        px, py = (float(value) for value in prediction["translation"][:2])
        name = str(prediction["detection_name"])
        candidates = []
        for index in unmatched:
            target = ground_truth[index]
            if str(target["detection_name"]) != name:
                continue
            gx, gy = (float(value) for value in target["translation"][:2])
            distance = math.hypot(px - gx, py - gy)
            if distance <= distance_threshold:
                candidates.append((distance, index))
        if candidates:
            _, matched = min(candidates)
            unmatched.remove(matched)
            true_positives += 1
    false_positives = len(ordered) - true_positives
    false_negatives = len(unmatched)
    return true_positives, false_positives, false_negatives


def local_detection_utility(
    predictions: Sequence[Mapping[str, object]],
    ground_truth: Sequence[Mapping[str, object]],
    *,
    score_threshold: float = 0.05,
    distance_thresholds: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
    false_positive_weight: float = 0.10,
) -> float:
    """Dataset-metric-aligned local proxy used only to rank per-frame actions."""
    kept = [
        record for record in predictions
        if float(record.get("detection_score", 0.0)) >= score_threshold
    ]
    total = 0.0
    thresholds = tuple(float(value) for value in distance_thresholds)
    if not thresholds:
        raise ValueError("at least one distance threshold is required")
    normalizer = max(1, len(ground_truth))
    for threshold in thresholds:
        tp, fp, fn = _match_counts(kept, ground_truth, threshold)
        total += (tp - fn - false_positive_weight * fp) / normalizer
    return total / len(thresholds)


def _quaternion_yaw(rotation: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in rotation)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped_angle_error(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def local_nds_like_utility(
    predictions: Sequence[Mapping[str, object]],
    ground_truth: Sequence[Mapping[str, object]],
    *,
    score_threshold: float = 0.05,
    match_distance_m: float = 4.0,
    false_positive_weight: float = 0.25,
    false_negative_weight: float = 1.0,
) -> float:
    """Class-balanced local proxy including NDS translation/velocity/yaw/scale quality."""
    kept = [
        record for record in predictions
        if float(record.get("detection_score", 0.0)) >= score_threshold
    ]
    active_classes = [
        name for name in DETECTION_CLASSES
        if any(str(item.get("detection_name")) == name for item in kept)
        or any(str(item.get("detection_name")) == name for item in ground_truth)
    ]
    if not active_classes:
        return 0.0
    class_scores = []
    for name in active_classes:
        predictions_for_class = [
            item for item in kept if str(item.get("detection_name")) == name
        ]
        targets = [
            item for item in ground_truth if str(item.get("detection_name")) == name
        ]
        unmatched = set(range(len(targets)))
        quality_sum = 0.0
        matches = 0
        ordered = sorted(
            predictions_for_class,
            key=lambda item: float(item.get("detection_score", 0.0)),
            reverse=True,
        )
        for prediction in ordered:
            px, py = (float(value) for value in prediction["translation"][:2])
            candidates = []
            for index in unmatched:
                target = targets[index]
                gx, gy = (float(value) for value in target["translation"][:2])
                distance = math.hypot(px - gx, py - gy)
                if distance <= match_distance_m:
                    candidates.append((distance, index))
            if not candidates:
                continue
            distance, matched_index = min(candidates)
            unmatched.remove(matched_index)
            target = targets[matched_index]
            center_quality = max(0.0, 1.0 - distance / match_distance_m)
            prediction_velocity = prediction.get("velocity", (0.0, 0.0))
            target_velocity = target.get("velocity", (0.0, 0.0))
            velocity_error = math.hypot(
                float(prediction_velocity[0]) - float(target_velocity[0]),
                float(prediction_velocity[1]) - float(target_velocity[1]),
            )
            velocity_quality = math.exp(-velocity_error / 2.0)
            if "rotation" in prediction and "rotation" in target:
                yaw_error = _wrapped_angle_error(
                    _quaternion_yaw(prediction["rotation"]),
                    _quaternion_yaw(target["rotation"]),
                )
                yaw_quality = math.exp(-yaw_error)
            else:
                yaw_quality = 0.5
            if "size" in prediction and "size" in target:
                log_errors = [
                    abs(math.log(max(float(p), 1e-3) / max(float(t), 1e-3)))
                    for p, t in zip(prediction["size"], target["size"])
                ]
                scale_quality = math.exp(-sum(log_errors) / len(log_errors))
            else:
                scale_quality = 0.5
            quality_sum += (
                0.50
                + 0.20 * center_quality
                + 0.15 * velocity_quality
                + 0.10 * yaw_quality
                + 0.05 * scale_quality
            )
            matches += 1
        false_positives = len(ordered) - matches
        false_negatives = len(unmatched)
        class_scores.append(
            (
                quality_sum
                - false_positive_weight * false_positives
                - false_negative_weight * false_negatives
            )
            / max(1, len(targets))
        )
    return sum(class_scores) / len(class_scores)


def oldest_first_selections(
    ordered_frames: Sequence[tuple[str, str]],
    *,
    camera_count: int,
    cameras: Sequence[str] = CAMERAS,
) -> dict[str, tuple[str, ...]]:
    """Deterministic max-age scheduler, resetting its state at scene boundaries."""
    if not 1 <= camera_count <= len(cameras):
        raise ValueError("camera_count is outside the available camera range")
    selections: dict[str, tuple[str, ...]] = {}
    active_scene = None
    ages: dict[str, int] = {}
    camera_order = {camera: index for index, camera in enumerate(cameras)}
    for token, scene_token in ordered_frames:
        if scene_token != active_scene:
            active_scene = scene_token
            ages = {camera: 1 for camera in cameras}
        ranked = sorted(cameras, key=lambda camera: (-ages[camera], camera_order[camera]))
        selected = tuple(ranked[:camera_count])
        selections[token] = selected
        for camera in cameras:
            ages[camera] = 0 if camera in selected else ages[camera] + 1
    return selections


def random_selections(
    ordered_frames: Sequence[tuple[str, str]],
    *,
    camera_count: int,
    seed: int,
    cameras: Sequence[str] = CAMERAS,
) -> dict[str, tuple[str, ...]]:
    if not 1 <= camera_count <= len(cameras):
        raise ValueError("camera_count is outside the available camera range")
    rng = random.Random(seed)
    order = {camera: index for index, camera in enumerate(cameras)}
    return {
        token: tuple(sorted(rng.sample(list(cameras), camera_count), key=order.get))
        for token, _scene_token in ordered_frames
    }
