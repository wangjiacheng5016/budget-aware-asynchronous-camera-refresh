#!/usr/bin/env python3
"""Re-evaluate frozen staleness predictions on one strict common token support."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Any

try:
    from .replay_stratified_analysis import (
        _official_evaluate,
        _ordered_frames,
        build_output_manifest,
        read_json,
        sha256,
    )
except ImportError:
    from replay_stratified_analysis import (
        _official_evaluate,
        _ordered_frames,
        build_output_manifest,
        read_json,
        sha256,
    )

METHOD_SPECS = tuple(
    (f"age{str(age).replace('.', 'p')}_{mode}", age, mode)
    for age in (0.5, 1.0, 1.5, 2.0)
    for mode in ("static", "motion")
)


def discover_method_inputs(root: Path) -> dict[str, dict[str, Path]]:
    """Resolve the audited assignment, diagnostic, and compressed result inputs."""
    discovered = {}
    for name, _age, _mode in METHOD_SPECS:
        method = root / name
        results = sorted(method.glob("evaluations/**/results_nusc.json.gz"))
        raw_results = sorted(method.glob("evaluations/**/results_nusc.json"))
        assignments = method / "history_assignments.json"
        diagnostics = method / "controlled_staleness_diagnostics.json"
        if len(results) != 1 or raw_results or not assignments.is_file() or not diagnostics.is_file():
            raise RuntimeError(f"{name}: expected one compressed result, no raw result, assignments and diagnostics")
        discovered[name] = {
            "result": results[0],
            "assignments": assignments,
            "diagnostics": diagnostics,
        }
    return discovered


def support_from_assignments(
    assignments: Mapping[str, Mapping[str, Any] | None],
) -> set[str]:
    """Define eligible support by a valid history assignment, not detections."""
    return {str(token) for token, assignment in assignments.items() if assignment is not None}


def collapse_mode_supports(
    supports: Mapping[str, Set[str]],
) -> dict[float, set[str]]:
    """Verify static/motion share support, then expose the four age supports."""
    by_age = {}
    for _name, age, _mode in METHOD_SPECS:
        static = set(supports[f"age{str(age).replace('.', 'p')}_static"])
        motion = set(supports[f"age{str(age).replace('.', 'p')}_motion"])
        if static != motion:
            raise RuntimeError(f"mode support mismatch at target age {age}")
        by_age[age] = static
    return by_age


def build_common_support(supports: Mapping[str, Set[str]]) -> set[str]:
    """Return the strict intersection across every named method support."""
    if len(supports) != 8:
        raise ValueError("common-support staleness requires exactly eight methods")
    iterator = iter(supports.values())
    common = set(next(iterator))
    for support in iterator:
        common.intersection_update(support)
    if not common:
        raise RuntimeError("strict common support is empty")
    return common


def mask_predictions_to_support(
    *,
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    ordered_tokens: Sequence[str],
    common_support: Set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Keep predictions on support and emit empty lists on every other val token."""
    expected = set(ordered_tokens)
    if not common_support.issubset(expected):
        raise ValueError("common support contains tokens outside the evaluation split")
    missing = common_support.difference(predictions)
    if missing:
        raise RuntimeError(f"method is missing {len(missing)} common-support tokens")
    return {
        token: [dict(record) for record in predictions[token]]
        if token in common_support else []
        for token in ordered_tokens
    }


def token_set_sha256(tokens: Set[str]) -> str:
    encoded = "\n".join(sorted(tokens)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=tuple(spec[0] for spec in METHOD_SPECS), required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--eval-set", default="val")
    return parser.parse_args(argv)


def _seven_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "NDS": float(metrics["nd_score"]),
        "mAP": float(metrics["mean_ap"]),
        "mATE": float(metrics["tp_errors"]["trans_err"]),
        "mASE": float(metrics["tp_errors"]["scale_err"]),
        "mAOE": float(metrics["tp_errors"]["orient_err"]),
        "mAVE": float(metrics["tp_errors"]["vel_err"]),
        "mAAE": float(metrics["tp_errors"]["attr_err"]),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_root = Path(args.source_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = discover_method_inputs(source_root)

    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    ordered_frames = _ordered_frames(nusc, create_splits_scenes()[args.eval_set])
    ordered_tokens = tuple(token for token, _scene in ordered_frames)
    expected = set(ordered_tokens)
    supports = {}
    diagnostics_by_method = {}
    for name, age, mode in METHOD_SPECS:
        assignments = read_json(inputs[name]["assignments"])
        diagnostics = read_json(inputs[name]["diagnostics"])
        if set(assignments) != expected:
            raise RuntimeError(f"{name}: assignment token set is not complete val")
        if (
            diagnostics.get("protocol_name") != "controlled_staleness"
            or diagnostics.get("diagnostic_only") is not True
            or diagnostics.get("frame_count") != len(ordered_tokens)
            or diagnostics.get("mode") != mode
            or float(diagnostics.get("age", {}).get("target_age_s", -1)) != age
            or diagnostics.get("protocol", {}).get("uses_current_token_cache") is not False
            or diagnostics.get("protocol", {}).get("uses_cross_scene_history") is not False
        ):
            raise RuntimeError(f"{name}: incompatible controlled-staleness diagnostics")
        support = support_from_assignments(assignments)
        reported_matched = int(diagnostics.get("age", {}).get("matched_frames", -1))
        if len(support) != reported_matched:
            raise RuntimeError(f"{name}: assignment support disagrees with diagnostics")
        supports[name] = support
        diagnostics_by_method[name] = diagnostics
    age_supports = collapse_mode_supports(supports)
    common_support = set.intersection(*(set(value) for value in age_supports.values()))
    if not common_support:
        raise RuntimeError("strict four-age common support is empty")

    source_payload = read_json(inputs[args.method]["result"])
    source_predictions = source_payload.get("results", source_payload)
    if set(source_predictions) != expected:
        raise RuntimeError(f"{args.method}: source prediction token set is not complete val")
    predictions = mask_predictions_to_support(
        predictions=source_predictions,
        ordered_tokens=ordered_tokens,
        common_support=common_support,
    )
    metrics, result_path = _official_evaluate(
        nusc=nusc,
        predictions=predictions,
        eval_set=args.eval_set,
        output_dir=output_dir,
    )
    if result_path.name != "results_nusc.json.gz" or result_path.with_suffix("").exists():
        raise RuntimeError("official result compression audit failed")
    method_spec = next(spec for spec in METHOD_SPECS if spec[0] == args.method)
    protocol = {
        "schema_version": 1,
        "protocol_name": "controlled_staleness_strict_common_support",
        "diagnostic_only": True,
        "method": args.method,
        "target_age_s": method_spec[1],
        "mode": method_spec[2],
        "version": args.version,
        "eval_set": args.eval_set,
        "full_val_token_count": len(expected),
        "common_support_token_count": len(common_support),
        "outside_support_empty_token_count": len(expected - common_support),
        "common_support_fraction": len(common_support) / len(expected),
        "common_support_sha256": token_set_sha256(common_support),
        "support_semantics": (
            "strict intersection of non-null same-scene history assignments across all eight "
            "age/mode methods; every val token outside the intersection is an empty prediction"
        ),
        "official_metrics": _seven_metrics(metrics),
        "official_result": {"path": str(result_path), "sha256": sha256(result_path)},
        "source_support_counts": {name: len(support) for name, support in supports.items()},
        "age_support_counts": {str(age): len(support) for age, support in age_supports.items()},
        "inputs": {
            name: {
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in paths.items()
            }
            for name, paths in inputs.items()
        },
    }
    protocol_path = output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    manifest = build_output_manifest(output_dir, run_name=args.method)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
