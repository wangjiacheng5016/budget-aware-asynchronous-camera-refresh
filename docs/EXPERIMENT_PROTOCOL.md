# Interpretation and fixed comparison protocol

## Primary shared-checkpoint experiment

- nuScenes `v1.0-trainval`, official validation split, 6,019 frames in 150 scenes; official detection evaluator `detection_cvpr_2019`.
- One frozen MobileNetV3-based detector, 7,191,090 parameters; no additional learned parameters in the five compared mechanisms.
- Input resolution 640 x 360, FP32, batch one. Selected cameras use current and previous same-camera keyframe images. At a scene's first frame, current is cloned as previous and `prev_valid=0`. A selected detector call can read its previous same-camera raw image even if that camera was not refreshed on the preceding step.
- Identical deterministic oldest-first, frame-age refresh schedules at K=3 and K=4 for all five conditions; K=6/no-reuse is the full-refresh reference. Masks are serialized in canonical camera order. There is no free all-camera warm start.
- Canonical order: `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`.
- Identical available cached-observation context: at most 1.5 seconds by each camera's actual `sample_data` capture timestamps, reset between scenes. This is a cached-observation bound, not a bound on all raw-image ancestors. The dual-frame branch can extend available raw-image ancestry to approximately 2.15 seconds. Per-frame available and used ancestry are recorded separately.
- Detection input score threshold 0.05, native per-view NMS IoU 0.2, final class-aware NMS IoU 0.05, at most 500 detections per frame. Attribute assignment is common to all methods and uses the method's output velocity, including a 0.2 m/s moving/stationary threshold.

## Conditions

| Condition | Operation |
| --- | --- |
| `none` | Fuse only the K selected current-camera outputs |
| `static` | Add the newest eligible output from each unselected camera without motion compensation |
| `motion` | Add those outputs after compensation using detector-predicted velocity |
| `ab3dmot` | Rebuild the upstream association/Kalman mechanism from the common bounded observations |
| `simpletrack` | Rebuild the upstream tracking/prediction mechanism from the common bounded observations |

The tracker mechanisms cover the seven nuScenes tracking classes: bicycle, bus, car, motorcycle, pedestrian, trailer and truck. Construction vehicle, barrier and traffic cone use a shared static fallback in **all** reuse methods, including motion. Current observations are identical across conditions. Each tracker rebuilds state from only the common bounded context, with immediate birth and survival adapted to that context; these adaptations differ from their original full tracking protocols. Native association settings and native tracker confidence behavior are retained. This is a comparison of adapted temporal mechanisms under a common detector and observation budget, not evidence that the complete proposed detector outperforms the original external models.

AB3DMOT uses the upstream nuScenes/megvii association settings and timestamp-adjusted transition intervals. Its visualization, logging and unused KITTI I/O imports are omitted at runtime by AST filtering. The official rotation function bodies execute in Python to accommodate the removal of an old Numba object-mode fallback. SimpleTrack uses its upstream GIoU configuration with the common input-score threshold and bounded-context birth/survival adaptations. See `mechanism_adapters.py` for the exact transformations and upstream commits in `THIRD_PARTY.md`.

## Scope limits

1. This dual-frame protocol is not a claim of compliance with a standard 0.5-second raw-history challenge restriction. The code explicitly records `official_history_compliant=False`. Official detection **scoring** and challenge input-history eligibility are separate issues.
2. Shared-cache replay audits which observations each mechanism receives. Its CPU runtime excludes native detector generation and does not measure online latency, energy or a reduction in raw image accesses of K/6. Each selected detector call has a previous-image branch.
3. `eval_official_history_protocol_b.py` is a cache-level upper-bound analysis. A shared current-only output seam is required before interpreting it as K online detector calls. It is not the primary deployed implementation.
4. Current-only export is included as an explicitly separate auxiliary path. Do not substitute its cache for the primary dual-frame cache.
5. Older routing/staleness/budget/robustness evaluations can have different cache and attribute protocols. They must be labeled by their own configuration rather than pooled with the adapted comparison as though all protocols were identical.
6. The frozen repeatability envelope is an implementation smoke guard established before the release. It is neither bitwise inference equality nor evidence of task-quality equivalence. Twenty-four fixed inference spot checks and 80-frame serial/parallel replay checks do not establish full 6,019-frame online-forward equivalence.
7. Desktop end-to-end timing includes actual dual-frame I/O and inference. Jetson temporal graph timing uses placeholder previous images and identity motion; CPU scheduling overhead is measured separately. These measurements do not establish an online Jetson end-to-end latency or real-time guarantee.
8. Dataset, weights and cached predictions are not shipped. The code-only release cannot independently recreate the frozen weights or the exact paper numbers without these inputs.

All settings are fixed for a comparison. Do not select tracker settings or observation windows on the validation results to favor one condition. Preserve original runs and inspect source manifests and complete frame coverage before reporting metrics.
