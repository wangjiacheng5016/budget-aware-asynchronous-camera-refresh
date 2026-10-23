"""Pure, dependency-light helpers for deterministic robustness schedules."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

try:
    from .activecam_core import CAMERAS
except ImportError:
    from activecam_core import CAMERAS


@dataclass(frozen=True)
class OutageSpec:
    camera: str
    start: int
    duration: int
    period: int | None = None

    def unavailable(self, scene_frame_index: int) -> bool:
        if scene_frame_index < self.start:
            return False
        offset = scene_frame_index - self.start
        if self.period is None:
            return offset < self.duration
        return offset % self.period < self.duration


def parse_capture_sequence(value: str) -> tuple[bool, ...]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw or any(item not in {"0", "1"} for item in raw):
        raise argparse.ArgumentTypeError("capture sequence must contain only 0 and 1")
    return tuple(item == "1" for item in raw)


def parse_outage(value: str) -> OutageSpec:
    fields = [item.strip() for item in value.split(":")]
    if len(fields) not in {3, 4}:
        raise argparse.ArgumentTypeError(
            "outage must be CAMERA:START:DURATION[:PERIOD]"
        )
    camera = fields[0]
    if camera not in CAMERAS:
        raise argparse.ArgumentTypeError(f"unknown camera: {camera}")
    try:
        start, duration = int(fields[1]), int(fields[2])
        period = int(fields[3]) if len(fields) == 4 else None
    except ValueError as error:
        raise argparse.ArgumentTypeError("outage positions must be integers") from error
    if start < 0 or duration < 1 or (period is not None and period < duration):
        raise argparse.ArgumentTypeError(
            "outage requires start >= 0, duration >= 1, and period >= duration"
        )
    return OutageSpec(camera, start, duration, period)


def availability_at_step(
    scene_frame_index: int,
    outages: tuple[OutageSpec, ...],
) -> dict[str, bool]:
    return {
        camera: not any(
            spec.camera == camera and spec.unavailable(scene_frame_index)
            for spec in outages
        )
        for camera in CAMERAS
    }


def deadline_violations_after_action(
    *,
    ages_s: tuple[float, ...] | list[float],
    initialized: tuple[bool, ...] | list[bool],
    selected: tuple[str, ...] | list[str],
    next_interval_s: float,
    max_age_s: float,
    tolerance_s: float = 0.01,
    cameras: tuple[str, ...] = CAMERAS,
) -> tuple[str, ...]:
    """Audit the next-cycle deadline independently of scheduler feasibility.

    Availability is intentionally absent: a camera outage may make refresh
    impossible, but it must not make the resulting deadline violation vanish.
    Previously unseen and still-unselected cameras are excluded because no
    source timestamp exists from which an age can be defined.
    """
    if len(ages_s) != len(cameras) or len(initialized) != len(cameras):
        raise ValueError("ages and initialization flags must match camera count")
    if next_interval_s < 0.0 or max_age_s <= 0.0 or tolerance_s < 0.0:
        raise ValueError("invalid deadline audit parameters")
    chosen = set(selected)
    violations = []
    for camera, age_s, was_initialized in zip(cameras, ages_s, initialized):
        if not was_initialized and camera not in chosen:
            continue
        next_age_s = next_interval_s if camera in chosen else float(age_s) + next_interval_s
        if next_age_s > max_age_s + tolerance_s:
            violations.append(camera)
    return tuple(violations)


def audit_actual_deadline_after_action(
    *,
    ages_s: tuple[float, ...] | list[float],
    source_timestamps: tuple[int, ...] | list[int],
    selected: tuple[str, ...] | list[str],
    current_timestamp_us: int,
    next_timestamp_us: int | None,
    max_age_s: float,
    tolerance_s: float = 0.01,
    cameras: tuple[str, ...] = CAMERAS,
) -> tuple[float, tuple[str, ...]]:
    """Audit a chosen action using the subsequently observed frame interval.

    This helper is deliberately post-decision: callers must select an action
    before reading ``next_timestamp_us``. A missing or non-forward timestamp is
    treated as a zero interval, matching scene-end handling in the evaluators.
    """
    if len(source_timestamps) != len(cameras):
        raise ValueError("source timestamps must match camera count")
    next_interval_s = (
        0.0
        if next_timestamp_us is None
        else max(0.0, (int(next_timestamp_us) - int(current_timestamp_us)) / 1_000_000.0)
    )
    violations = deadline_violations_after_action(
        ages_s=ages_s,
        initialized=[int(timestamp) >= 0 for timestamp in source_timestamps],
        selected=selected,
        next_interval_s=next_interval_s,
        max_age_s=max_age_s,
        tolerance_s=tolerance_s,
        cameras=cameras,
    )
    return next_interval_s, violations


def summarize_deadline_violations(
    violations_by_frame: tuple[tuple[str, ...], ...] | list[tuple[str, ...]],
) -> dict[str, float | int]:
    """Return frame- and camera-level deadline counts from a route trace."""
    frame_count = len(violations_by_frame)
    violation_frames = sum(bool(cameras) for cameras in violations_by_frame)
    camera_events = sum(len(cameras) for cameras in violations_by_frame)
    return {
        "actual_deadline_violation_frames": violation_frames,
        "actual_deadline_violation_rate": violation_frames / max(1, frame_count),
        "actual_deadline_violating_camera_events": camera_events,
    }
