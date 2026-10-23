#!/usr/bin/env python3
"""Fail-closed sensor-history rules for an official-test-compliant ActiveCam.

The nuScenes detection rules permit approximately 0.5 seconds of past sensor
data and no more than six past camera images.  Protocol A makes the temporal
V31 branch current-only and therefore counts one past image per valid reused
camera cache entry.  Age is measured from same-camera capture timestamps, not
from the nominal sample timestamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence


OFFICIAL_MAX_PAST_WINDOW_S = 0.5
OFFICIAL_MAX_PAST_CAMERA_IMAGES = 6


def camera_capture_timestamps_us(
    nusc: Any, token: str, cameras: Sequence[str]
) -> dict[str, int]:
    """Read real per-camera capture times, never the nominal sample anchor."""

    sample = nusc.get("sample", token)
    output = {}
    for camera in cameras:
        sample_data_token = sample["data"].get(camera)
        if not sample_data_token:
            raise KeyError(f"{token}: missing sample_data for {camera}")
        output[camera] = int(
            nusc.get("sample_data", sample_data_token)["timestamp"]
        )
    return output


@dataclass(frozen=True)
class MemoryProvenance:
    """Raw-image provenance for one cached current-only camera prediction."""

    raw_image_timestamp_us: int
    source_sample_timestamp_us: int


@dataclass(frozen=True)
class FrameHistoryAudit:
    selected: tuple[str, ...]
    valid_memory_cameras: tuple[str, ...]
    stale_memory_cameras: tuple[str, ...]
    missing_memory_cameras: tuple[str, ...]
    past_image_count: int
    max_past_image_age_s: float
    history_window_violations: int
    past_image_count_violations: int

    @property
    def compliant(self) -> bool:
        return not (
            self.history_window_violations or self.past_image_count_violations
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "compliant": self.compliant}


def audit_protocol_a_frame(
    *,
    cameras: Sequence[str],
    selected: Sequence[str],
    current_capture_timestamps_us: Mapping[str, int],
    memory: Mapping[str, MemoryProvenance],
    max_past_window_s: float = OFFICIAL_MAX_PAST_WINDOW_S,
    max_past_images: int = OFFICIAL_MAX_PAST_CAMERA_IMAGES,
) -> FrameHistoryAudit:
    """Select legal external memory under conservative protocol A.

    Entries with negative age or age strictly greater than 0.5 seconds are
    rejected.  The exact 0.5-second boundary remains valid.  Selected current
    images are not past images and selected inference must separately enforce
    ``prev_valid=0``.
    """

    cameras = tuple(cameras)
    selected = tuple(selected)
    if len(set(cameras)) != len(cameras):
        raise ValueError("camera order contains duplicates")
    if len(set(selected)) != len(selected) or not set(selected) <= set(cameras):
        raise ValueError("selected cameras must be a unique subset")
    if set(current_capture_timestamps_us) != set(cameras):
        raise ValueError("one current capture timestamp is required per camera")
    if max_past_window_s != OFFICIAL_MAX_PAST_WINDOW_S:
        raise ValueError("strict official protocol freezes the history window at 0.5 s")
    if max_past_images > OFFICIAL_MAX_PAST_CAMERA_IMAGES:
        raise ValueError("strict official protocol permits at most six past images")

    selected_set = set(selected)
    valid: list[str] = []
    stale: list[str] = []
    missing: list[str] = []
    valid_ages: list[float] = []
    for camera in cameras:
        if camera in selected_set:
            continue
        provenance = memory.get(camera)
        if provenance is None:
            missing.append(camera)
            continue
        age_s = (
            int(current_capture_timestamps_us[camera])
            - int(provenance.raw_image_timestamp_us)
        ) / 1_000_000.0
        if age_s < 0.0 or age_s > max_past_window_s:
            stale.append(camera)
            continue
        valid.append(camera)
        valid_ages.append(age_s)

    past_image_count = len(valid)
    return FrameHistoryAudit(
        selected=selected,
        valid_memory_cameras=tuple(valid),
        stale_memory_cameras=tuple(stale),
        missing_memory_cameras=tuple(missing),
        past_image_count=past_image_count,
        max_past_image_age_s=max(valid_ages, default=0.0),
        history_window_violations=sum(
            age > max_past_window_s for age in valid_ages
        ),
        past_image_count_violations=int(past_image_count > max_past_images),
    )


def build_protocol_a_predictions(
    *,
    token: str,
    current_sample_timestamp_us: int,
    cameras: Sequence[str],
    selected: Sequence[str],
    current_capture_timestamps_us: Mapping[str, int],
    current_only_caches: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    memory: Mapping[
        str, tuple[MemoryProvenance, list[dict[str, Any]]]
    ],
    propagate: Callable[
        [list[dict[str, Any]], float], list[dict[str, Any]]
    ],
    merge: Callable[
        [list[tuple[str, list[dict[str, Any]]]]], list[dict[str, Any]]
    ],
) -> tuple[list[dict[str, Any]], FrameHistoryAudit]:
    """Fuse selected current-only detections and only legally aged memory."""

    audit = audit_protocol_a_frame(
        cameras=cameras,
        selected=selected,
        current_capture_timestamps_us=current_capture_timestamps_us,
        memory={camera: value[0] for camera, value in memory.items()},
    )
    selected_set = set(selected)
    valid_memory = set(audit.valid_memory_cameras)
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    for camera in cameras:
        if camera in selected_set:
            # Never touch an unselected current cache: this is also the central
            # leakage guard used by the existing router simulator.
            records = current_only_caches[camera][token]
        elif camera in valid_memory:
            provenance, stored = memory[camera]
            # Raw-sensor eligibility is anchored to same-camera capture time;
            # global constant-velocity propagation is anchored to the source
            # and target sample coordinate frames.
            motion_age_s = (
                int(current_sample_timestamp_us)
                - int(provenance.source_sample_timestamp_us)
            ) / 1_000_000.0
            if motion_age_s < 0.0:
                raise ValueError("memory source sample cannot be from the future")
            records = propagate(stored, motion_age_s)
        else:
            records = []
        sources.append((camera, records))
    return merge(sources), audit


def official_val_stop_gate(
    *,
    metrics: Mapping[int, Mapping[str, float]],
    audits: Mapping[int, Mapping[str, int]],
    k4_min_nds_retention: float = 0.98,
    k4_min_map_retention: float = 0.94,
    k3_min_nds_retention: float = 0.97,
    k3_min_map_retention: float = 0.90,
) -> dict[str, Any]:
    """Frozen val gate before creating any official test result.

    K4 is the paper's primary operating point.  K3 is an optional aggressive
    point and failing it does not invalidate a successful K4 result.
    """

    if set(metrics) != {3, 4, 6} or set(audits) != {3, 4, 6}:
        raise ValueError("the minimal matrix must contain K6, K4, and K3")
    for k in (3, 4, 6):
        if not {"NDS", "mAP"} <= set(metrics[k]):
            raise ValueError(f"K{k} is missing NDS or mAP")
    if float(metrics[6]["NDS"]) <= 0.0 or float(metrics[6]["mAP"]) <= 0.0:
        raise ValueError("K6 reference metrics must be positive")

    forbidden = (
        "history_window_violations",
        "past_image_count_violations",
        "unselected_current_reads",
        "previous_branch_valid_events",
    )
    global_compliance = all(
        int(audits[k].get(field, -1)) == 0
        for k in (3, 4, 6)
        for field in forbidden
    )
    retention = {
        k: {
            "NDS": float(metrics[k]["NDS"]) / float(metrics[6]["NDS"]),
            "mAP": float(metrics[k]["mAP"]) / float(metrics[6]["mAP"]),
        }
        for k in (3, 4)
    }
    k4_gate = global_compliance and (
        retention[4]["NDS"] >= k4_min_nds_retention
        and retention[4]["mAP"] >= k4_min_map_retention
    )
    k3_gate = global_compliance and (
        retention[3]["NDS"] >= k3_min_nds_retention
        and retention[3]["mAP"] >= k3_min_map_retention
    )
    if not global_compliance:
        decision = "STOP_PROTOCOL_VIOLATION"
    elif k4_gate and k3_gate:
        decision = "PROCEED_K4_PRIMARY_K3_AGGRESSIVE"
    elif k4_gate:
        decision = "PROCEED_K4_ONLY"
    else:
        decision = "STOP_PERFORMANCE_NOT_ESTABLISHED"
    return {
        "global_compliance_gate": global_compliance,
        "k4_primary_gate": k4_gate,
        "k3_aggressive_gate": k3_gate,
        "retention_vs_current_only_k6": retention,
        "thresholds": {
            "k4_min_nds_retention": k4_min_nds_retention,
            "k4_min_map_retention": k4_min_map_retention,
            "k3_min_nds_retention": k3_min_nds_retention,
            "k3_min_map_retention": k3_min_map_retention,
        },
        "decision": decision,
    }
