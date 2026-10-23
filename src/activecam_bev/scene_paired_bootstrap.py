#!/usr/bin/env python3
"""Paired scene-cluster bootstrap for nuScenes detection results.

The script consumes the ``results_nusc.json`` files already produced by the
router/fixed-subset evaluators.  It keeps every frame from a scene together,
draws scenes with replacement, and recomputes the official AP/TP/NDS
aggregation for every draw.  No detector forward pass is required.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import random
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .mini_preflight import load_dependencies
except ImportError:  # Direct script execution on the experiment server.
    from mini_preflight import load_dependencies


DIAGNOSTIC_NAMES = (
    "router_diagnostics.json",
    "fixed_subset_diagnostics.json",
    "diagnostics.json",
)
METRIC_NAMES = ("NDS", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE")
ERROR_METRICS = {
    "trans_err": "mATE",
    "scale_err": "mASE",
    "orient_err": "mAOE",
    "vel_err": "mAVE",
    "attr_err": "mAAE",
}


@dataclass(frozen=True)
class MethodArtifact:
    label: str
    result_path: Path
    diagnostics_path: Path | None
    reported_metrics: Mapping[str, float]


@dataclass
class MatchAccumulator:
    """Sufficient statistics for one class/distance threshold."""

    npos_by_scene: np.ndarray
    prediction_scene: np.ndarray
    true_positive: np.ndarray
    confidence: np.ndarray
    trans_err: np.ndarray
    vel_err: np.ndarray
    scale_err: np.ndarray
    orient_err: np.ndarray
    attr_err: np.ndarray


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("method must be LABEL=PATH")
    return label, Path(raw_path).expanduser()


def parse_named_group(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("group must be LABEL=METHOD1,METHOD2,...")
    label, raw_members = value.split("=", 1)
    members = tuple(item.strip() for item in raw_members.split(",") if item.strip())
    if not label.strip() or not members:
        raise argparse.ArgumentTypeError("group must have a label and at least one member")
    return label.strip(), members


def parse_comparison(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("comparison must be LEFT:RIGHT")
    left, right = (item.strip() for item in value.split(":", 1))
    if not left or not right or left == right:
        raise argparse.ArgumentTypeError("comparison labels must be distinct")
    return left, right


def _read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".json.gz") else Path.open
    if opener is gzip.open:
        handle_context = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle_context = path.open("r", encoding="utf-8")
    with handle_context as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _diagnostic_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    output = {}
    for name in METRIC_NAMES:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            output[name] = float(value)
    return output


def _summary_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    mapping = {
        "NDS": payload.get("nd_score"),
        "mAP": payload.get("mean_ap"),
        "mATE": payload.get("tp_errors", {}).get("trans_err"),
        "mASE": payload.get("tp_errors", {}).get("scale_err"),
        "mAOE": payload.get("tp_errors", {}).get("orient_err"),
        "mAVE": payload.get("tp_errors", {}).get("vel_err"),
        "mAAE": payload.get("tp_errors", {}).get("attr_err"),
    }
    return {
        name: float(value)
        for name, value in mapping.items()
        if isinstance(value, (int, float))
    }


def resolve_method_artifact(label: str, supplied_path: Path) -> MethodArtifact:
    """Resolve a result file from a result, diagnostics, or run directory path."""
    path = supplied_path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    diagnostics_path = None
    diagnostic_payload: dict[str, Any] = {}
    if path.is_file() and path.name in DIAGNOSTIC_NAMES:
        diagnostics_path = path
        diagnostic_payload = _read_json(path)
        search_root = path.parent
    elif path.is_dir():
        search_root = path
        diagnostics = [path / name for name in DIAGNOSTIC_NAMES if (path / name).exists()]
        if len(diagnostics) == 1:
            diagnostics_path = diagnostics[0]
            diagnostic_payload = _read_json(diagnostics_path)
    elif path.name in {"results_nusc.json", "results_nusc.json.gz"}:
        search_root = path.parent
    else:
        raise ValueError(f"unsupported method artifact: {path}")

    if path.is_file() and path.name in {"results_nusc.json", "results_nusc.json.gz"}:
        result_path = path
    else:
        raw_candidates = sorted(search_root.rglob("results_nusc.json"))
        gzip_candidates = sorted(search_root.rglob("results_nusc.json.gz"))
        # If both forms exist for the same logical result, prefer the original;
        # otherwise compressed-only experiment archives resolve normally.
        candidates_by_key = {
            str(candidate): candidate for candidate in raw_candidates
        }
        for candidate in gzip_candidates:
            logical = str(candidate)[:-3]
            candidates_by_key.setdefault(logical, candidate)
        candidates = sorted(candidates_by_key.values())
        method = diagnostic_payload.get("method")
        if method:
            exact = [candidate for candidate in candidates if candidate.parent.name == method]
            if len(exact) == 1:
                candidates = exact
        if len(candidates) != 1:
            shown = ", ".join(str(item) for item in candidates[:5]) or "none"
            raise ValueError(
                f"{path} resolves to {len(candidates)} result files ({shown}); "
                "pass the intended results_nusc.json explicitly"
            )
        result_path = candidates[0]

    result_payload = _read_json(result_path)
    if not isinstance(result_payload.get("results"), Mapping):
        raise ValueError(f"invalid nuScenes result file: {result_path}")
    reported = _diagnostic_metrics(diagnostic_payload)
    if not reported:
        reported = _summary_metrics(result_path.parent / "metrics_summary.json")
    return MethodArtifact(label, result_path, diagnostics_path, reported)


def materialize_result_for_devkit(
    source: Path,
    output_dir: Path,
    label: str,
) -> tuple[Path, Path | None]:
    """Return a plain JSON path, inflating gzip only inside our output tree."""
    if source.name == "results_nusc.json":
        return source, None
    if source.name != "results_nusc.json.gz":
        raise ValueError(f"unsupported result filename: {source}")
    temporary = output_dir / "materialized_inputs" / label / "results_nusc.json"
    temporary.parent.mkdir(parents=True, exist_ok=False)
    with gzip.open(source, "rb") as compressed, temporary.open("xb") as plain:
        shutil.copyfileobj(compressed, plain, length=1024 * 1024)
    return temporary, temporary


def cleanup_materialized_result(path: Path | None) -> None:
    if path is None:
        return
    path.unlink()
    parent = path.parent
    parent.rmdir()
    grandparent = parent.parent
    if grandparent.exists() and not any(grandparent.iterdir()):
        grandparent.rmdir()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_paired_differences(
    replicates: Sequence[Mapping[str, Mapping[str, float]]],
    left: str,
    right: str,
    confidence: float,
) -> dict[str, Any]:
    """Summarize paired LEFT-RIGHT differences from common bootstrap draws."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, Any] = {"left": left, "right": right, "difference": "left-minus-right"}
    for metric in METRIC_NAMES:
        values = [
            float(row[left][metric]) - float(row[right][metric])
            for row in replicates
        ]
        higher_is_better = metric in {"NDS", "mAP"}
        probability_better = statistics.fmean(
            float(value > 0.0 if higher_is_better else value < 0.0)
            for value in values
        )
        output[metric] = {
            "mean_difference": statistics.fmean(values),
            "median_difference": statistics.median(values),
            "ci_low": percentile(values, alpha),
            "ci_high": percentile(values, 1.0 - alpha),
            "probability_left_better": probability_better,
            "higher_is_better": higher_is_better,
        }
    return output


def _prepare_accumulators(
    gt_boxes,
    pred_boxes,
    scene_index_by_sample: Mapping[str, int],
    scene_count: int,
    cfg,
    center_distance,
    velocity_l2,
    scale_iou,
    yaw_diff,
    attr_acc,
) -> dict[tuple[str, float], MatchAccumulator]:
    """Match once, retaining enough data to reweight scenes exactly."""
    output = {}
    all_predictions = list(pred_boxes.all)
    for class_name in cfg.class_names:
        npos_by_scene = np.zeros(scene_count, dtype=np.int64)
        for sample_token in gt_boxes.sample_tokens:
            scene_index = scene_index_by_sample[sample_token]
            npos_by_scene[scene_index] += sum(
                box.detection_name == class_name for box in gt_boxes[sample_token]
            )
        class_predictions = [
            (global_index, box)
            for global_index, box in enumerate(all_predictions)
            if box.detection_name == class_name
        ]
        # Match ordering is identical to the official evaluator: confidence and
        # then original flattened prediction index, both descending.
        class_predictions.sort(
            key=lambda item: (item[1].detection_score, item[0]), reverse=True
        )
        for dist_th in cfg.dist_ths:
            taken: set[tuple[str, int]] = set()
            scene_indices, positives, confidences = [], [], []
            errors = {name: [] for name in ERROR_METRICS}
            for _global_index, pred_box in class_predictions:
                sample_token = pred_box.sample_token
                min_dist = float("inf")
                match_index = None
                for gt_index, gt_box in enumerate(gt_boxes[sample_token]):
                    if (
                        gt_box.detection_name == class_name
                        and (sample_token, gt_index) not in taken
                    ):
                        distance = cfg.dist_fcn_callable(gt_box, pred_box)
                        if distance < min_dist:
                            min_dist = distance
                            match_index = gt_index
                is_match = match_index is not None and min_dist < dist_th
                scene_indices.append(scene_index_by_sample[sample_token])
                positives.append(float(is_match))
                confidences.append(float(pred_box.detection_score))
                if is_match:
                    taken.add((sample_token, int(match_index)))
                    matched = gt_boxes[sample_token][int(match_index)]
                    period = np.pi if class_name == "barrier" else 2.0 * np.pi
                    values = {
                        "trans_err": center_distance(matched, pred_box),
                        "vel_err": velocity_l2(matched, pred_box),
                        "scale_err": 1.0 - scale_iou(matched, pred_box),
                        "orient_err": yaw_diff(matched, pred_box, period=period),
                        "attr_err": 1.0 - attr_acc(matched, pred_box),
                    }
                else:
                    values = {name: np.nan for name in ERROR_METRICS}
                for name, value in values.items():
                    errors[name].append(float(value))
            output[(class_name, float(dist_th))] = MatchAccumulator(
                npos_by_scene=npos_by_scene,
                prediction_scene=np.asarray(scene_indices, dtype=np.int16),
                true_positive=np.asarray(positives, dtype=np.bool_),
                confidence=np.asarray(confidences, dtype=np.float64),
                trans_err=np.asarray(errors["trans_err"], dtype=np.float64),
                vel_err=np.asarray(errors["vel_err"], dtype=np.float64),
                scale_err=np.asarray(errors["scale_err"], dtype=np.float64),
                orient_err=np.asarray(errors["orient_err"], dtype=np.float64),
                attr_err=np.asarray(errors["attr_err"], dtype=np.float64),
            )
    return output


def _metric_data_from_weights(acc: MatchAccumulator, weights: np.ndarray, api):
    DetectionMetricData = api["DetectionMetricData"]
    npos = int(np.dot(acc.npos_by_scene, weights))
    if npos == 0 or not len(acc.confidence):
        return DetectionMetricData.no_predictions()
    multiplicity = weights[acc.prediction_scene]
    keep = multiplicity > 0
    if not np.any(keep):
        return DetectionMetricData.no_predictions()
    repeat = multiplicity[keep]
    tp_raw = np.repeat(acc.true_positive[keep], repeat)
    confidence_raw = np.repeat(acc.confidence[keep], repeat)
    if not np.any(tp_raw):
        return DetectionMetricData.no_predictions()
    fp_raw = 1.0 - tp_raw
    tp = np.cumsum(tp_raw)
    fp = np.cumsum(fp_raw)
    precision = tp / (tp + fp)
    recall = tp / float(npos)
    recall_grid = np.linspace(0.0, 1.0, DetectionMetricData.nelem)
    precision_grid = np.interp(recall_grid, recall, precision, right=0.0)
    confidence_grid = np.interp(recall_grid, recall, confidence_raw, right=0.0)

    matched = tp_raw.astype(bool)
    matched_confidence = confidence_raw[matched]
    interpolated_errors = {}
    for name in ERROR_METRICS:
        raw = np.repeat(getattr(acc, name)[keep], repeat)[matched]
        cumulative = api["cummean"](raw)
        interpolated_errors[name] = np.interp(
            confidence_grid[::-1], matched_confidence[::-1], cumulative[::-1]
        )[::-1]
    return DetectionMetricData(
        recall=recall_grid,
        precision=precision_grid,
        confidence=confidence_grid,
        trans_err=interpolated_errors["trans_err"],
        vel_err=interpolated_errors["vel_err"],
        scale_err=interpolated_errors["scale_err"],
        orient_err=interpolated_errors["orient_err"],
        attr_err=interpolated_errors["attr_err"],
    )


def _evaluate_accumulators(accumulators, weights: np.ndarray, cfg, api) -> dict[str, float]:
    metrics = api["DetectionMetrics"](cfg)
    metric_data = {}
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = _metric_data_from_weights(
                accumulators[(class_name, float(dist_th))], weights, api
            )
            metric_data[(class_name, float(dist_th))] = md
            metrics.add_label_ap(
                class_name,
                dist_th,
                api["calc_ap"](md, cfg.min_recall, cfg.min_precision),
            )
        tp_md = metric_data[(class_name, float(cfg.dist_th_tp))]
        for raw_name in ERROR_METRICS:
            if class_name == "traffic_cone" and raw_name in {
                "attr_err", "vel_err", "orient_err"
            }:
                value = np.nan
            elif class_name == "barrier" and raw_name in {"attr_err", "vel_err"}:
                value = np.nan
            else:
                value = api["calc_tp"](tp_md, cfg.min_recall, raw_name)
            metrics.add_label_tp(class_name, raw_name, value)
    serialized = metrics.serialize()
    return {
        "NDS": float(serialized["nd_score"]),
        "mAP": float(serialized["mean_ap"]),
        **{
            friendly: float(serialized["tp_errors"][raw])
            for raw, friendly in ERROR_METRICS.items()
        },
    }


_WORKER_STATE: dict[str, Any] = {}


def _bootstrap_worker(item: tuple[int, np.ndarray]) -> tuple[int, dict[str, dict[str, float]]]:
    replicate_index, weights = item
    methods = {
        label: _evaluate_accumulators(accumulators, weights, _WORKER_STATE["cfg"], _WORKER_STATE["api"])
        for label, accumulators in _WORKER_STATE["accumulators"].items()
    }
    for group_label, members in _WORKER_STATE["groups"].items():
        methods[group_label] = {
            metric: statistics.fmean(methods[member][metric] for member in members)
            for metric in METRIC_NAMES
        }
    return replicate_index, methods


def _load_api() -> dict[str, Any]:
    from nuscenes.eval.common.utils import attr_acc, center_distance, cummean, scale_iou, velocity_l2, yaw_diff
    from nuscenes.eval.detection.algo import calc_ap, calc_tp
    from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionMetrics

    return {
        "attr_acc": attr_acc,
        "center_distance": center_distance,
        "cummean": cummean,
        "scale_iou": scale_iou,
        "velocity_l2": velocity_l2,
        "yaw_diff": yaw_diff,
        "calc_ap": calc_ap,
        "calc_tp": calc_tp,
        "DetectionMetricData": DetectionMetricData,
        "DetectionMetrics": DetectionMetrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--legacy-src", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", type=parse_named_path, action="append", required=True)
    parser.add_argument("--group", type=parse_named_group, action="append", default=[])
    parser.add_argument("--compare", type=parse_comparison, action="append", default=[])
    parser.add_argument("--eval-set", default="val")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--point-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 1 or args.workers < 1:
        raise ValueError("replicates and workers must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    method_specs = dict(args.method)
    if len(method_specs) != len(args.method):
        raise ValueError("method labels must be unique")
    groups = dict(args.group)
    if len(groups) != len(args.group):
        raise ValueError("group labels must be unique")
    for group, members in groups.items():
        if group in method_specs:
            raise ValueError(f"group conflicts with method label: {group}")
        missing = set(members).difference(method_specs)
        if missing:
            raise ValueError(f"group {group} has unknown members: {sorted(missing)}")
    available_labels = set(method_specs).union(groups)
    comparisons = list(args.compare)
    if not comparisons:
        labels = list(available_labels)
        comparisons = [(labels[0], label) for label in labels[1:]]
    for left, right in comparisons:
        missing = {left, right}.difference(available_labels)
        if missing:
            raise ValueError(f"comparison has unknown labels: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        label: resolve_method_artifact(label, path)
        for label, path in method_specs.items()
    }
    _legacy, NuScenes, config_factory, NuScenesEval, _splits = load_dependencies(
        Path(args.legacy_src)
    )
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    cfg = config_factory("detection_cvpr_2019")
    api = _load_api()
    started = time.time()

    def load_filtered_boxes(label, artifact):
        loader_dir = output_dir / "loader" / label
        devkit_path, temporary = materialize_result_for_devkit(
            artifact.result_path, output_dir, label
        )
        try:
            evaluator = NuScenesEval(
                nusc,
                config=cfg,
                result_path=str(devkit_path),
                eval_set=args.eval_set,
                output_dir=str(loader_dir),
                verbose=False,
            )
            return evaluator.gt_boxes, evaluator.pred_boxes
        finally:
            cleanup_materialized_result(temporary)

    artifact_items = list(artifacts.items())
    first_label, first_artifact = artifact_items[0]
    gt_boxes, first_pred_boxes = load_filtered_boxes(first_label, first_artifact)
    sample_tokens = set(gt_boxes.sample_tokens)
    scene_by_sample = {
        sample["token"]: sample["scene_token"]
        for sample in nusc.sample
        if sample["token"] in sample_tokens
    }
    scenes = sorted(set(scene_by_sample.values()))
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    scene_to_index = {scene: index for index, scene in enumerate(scenes)}
    selected_samples = {
        token for token, scene in scene_by_sample.items() if scene in scene_to_index
    }
    scene_index_by_sample = {
        token: scene_to_index[scene_by_sample[token]] for token in selected_samples
    }
    if not selected_samples:
        raise RuntimeError("no evaluation samples remain")

    # Limit the already-filtered EvalBoxes views without touching the source.
    class BoxSubset:
        def __init__(self, boxes, tokens):
            self._boxes = boxes
            token_set = set(tokens)
            # Preserve devkit insertion order so confidence-tie behavior is
            # identical to the already reported full-split evaluation.
            self._tokens = [
                token for token in boxes.sample_tokens if token in token_set
            ]
        def __getitem__(self, token):
            return self._boxes[token]
        @property
        def sample_tokens(self):
            return self._tokens
        @property
        def all(self):
            return [box for token in self._tokens for box in self._boxes[token]]

    gt_subset = BoxSubset(gt_boxes, selected_samples)
    accumulators = {}

    def prepare_one(label, pred_boxes):
        pred_subset = BoxSubset(pred_boxes, selected_samples)
        return _prepare_accumulators(
            gt_subset,
            pred_subset,
            scene_index_by_sample,
            len(scenes),
            cfg,
            api["center_distance"],
            api["velocity_l2"],
            api["scale_iou"],
            api["yaw_diff"],
            api["attr_acc"],
        )

    # Parse/filter one potentially 0.65 GB result at a time. Only compact
    # sufficient-statistic arrays survive to the bootstrap stage.
    accumulators[first_label] = prepare_one(first_label, first_pred_boxes)
    del first_pred_boxes
    for label, artifact in artifact_items[1:]:
        current_gt, pred_boxes = load_filtered_boxes(label, artifact)
        if set(current_gt.sample_tokens) != sample_tokens:
            raise RuntimeError(f"sample-token mismatch for {label}")
        accumulators[label] = prepare_one(label, pred_boxes)
        del current_gt, pred_boxes

    point_weights = np.ones(len(scenes), dtype=np.int64)
    point_estimates = {
        label: _evaluate_accumulators(value, point_weights, cfg, api)
        for label, value in accumulators.items()
    }
    for group_label, members in groups.items():
        point_estimates[group_label] = {
            metric: statistics.fmean(point_estimates[member][metric] for member in members)
            for metric in METRIC_NAMES
        }
    if args.max_scenes is None:
        for label, artifact in artifacts.items():
            for metric in ("NDS", "mAP"):
                if metric in artifact.reported_metrics:
                    difference = abs(point_estimates[label][metric] - artifact.reported_metrics[metric])
                    if difference > args.point_tolerance:
                        raise RuntimeError(
                            f"{label} reconstructed {metric} differs from reported value by "
                            f"{difference:.8g}; check devkit/protocol compatibility"
                        )

    rng = random.Random(args.seed)
    draws = []
    for replicate in range(args.replicates):
        weights = np.zeros(len(scenes), dtype=np.int64)
        for _ in scenes:
            weights[rng.randrange(len(scenes))] += 1
        draws.append((replicate, weights))

    _WORKER_STATE.clear()
    _WORKER_STATE.update({
        "accumulators": accumulators,
        "cfg": cfg,
        "api": api,
        "groups": groups,
    })
    if args.workers == 1:
        results = [_bootstrap_worker(item) for item in draws]
    else:
        try:
            context = mp.get_context("fork")
        except ValueError as error:
            raise RuntimeError("workers > 1 requires fork multiprocessing (run on Linux)") from error
        with context.Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_bootstrap_worker, draws, chunksize=1))
    results.sort(key=lambda item: item[0])
    replicates = [payload for _index, payload in results]
    summaries = [
        summarize_paired_differences(replicates, left, right, args.confidence)
        for left, right in comparisons
    ]

    raw_path = output_dir / "bootstrap_replicates.jsonl.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        for index, payload in enumerate(replicates):
            handle.write(json.dumps({"replicate": index, "metrics": payload}) + "\n")
    summary = {
        "protocol": "paired scene-cluster percentile bootstrap with official nuScenes aggregation",
        "eval_set": args.eval_set,
        "scene_count": len(scenes),
        "sample_count": len(selected_samples),
        "replicates": args.replicates,
        "confidence": args.confidence,
        "seed": args.seed,
        "workers": args.workers,
        "point_estimates": point_estimates,
        "comparisons": summaries,
        "methods": {
            label: {
                "results_nusc": str(artifact.result_path),
                "diagnostics": str(artifact.diagnostics_path) if artifact.diagnostics_path else None,
                "reported_metrics": dict(artifact.reported_metrics),
            }
            for label, artifact in artifacts.items()
        },
        "groups": {label: list(members) for label, members in groups.items()},
        "raw_replicates": str(raw_path),
        "elapsed_seconds": time.time() - started,
        "interpretation": {
            "sampling_unit": "scene; all frames in a sampled scene retain their pairing",
            "ci": "percentile interval of paired left-minus-right differences",
            "positive_difference_better_for": ["NDS", "mAP"],
            "negative_difference_better_for": ["mATE", "mASE", "mAOE", "mAVE", "mAAE"],
        },
    }
    (output_dir / "paired_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
