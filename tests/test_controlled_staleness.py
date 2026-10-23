from __future__ import annotations

import unittest

from activecam_bev.activecam_core import CAMERAS
from activecam_bev.eval_controlled_staleness import (
    build_controlled_predictions,
    select_controlled_histories,
)


class _IdentityLegacy:
    @staticmethod
    def result_to_global_tuple(record, camera_index):
        return (
            float(record["translation"][0]),
            float(record["translation"][1]),
            float(record["translation"][2]),
            float(record["velocity"][0]),
            float(record["velocity"][1]),
            camera_index,
            0,
            0,
            0,
            float(record["detection_score"]),
        )

    @staticmethod
    def global_tuple_to_result(token, box):
        return {
            "sample_token": token,
            "translation": [box[0], box[1], box[2]],
            "velocity": [box[3], box[4]],
            "detection_score": box[9],
        }

    class bev:
        @staticmethod
        def nms_classwise(boxes, _merge_iou):
            return boxes


class ControlledStalenessTest(unittest.TestCase):
    def test_history_is_strictly_past_same_scene_and_within_tolerance(self) -> None:
        ordered_frames = [
            ("a0", "scene-a"),
            ("a1", "scene-a"),
            ("a2", "scene-a"),
            ("b0", "scene-b"),
            ("b1", "scene-b"),
        ]
        timestamps = {
            "a0": 0,
            "a1": 520_000,
            "a2": 1_020_000,
            "b0": 10_000_000,
            "b1": 10_500_000,
        }

        assignments, diagnostics = select_controlled_histories(
            ordered_frames=ordered_frames,
            timestamps=timestamps,
            target_age_s=0.5,
            tolerance_s=0.15,
        )

        self.assertIsNone(assignments["a0"])
        self.assertEqual(assignments["a1"]["history_token"], "a0")
        self.assertAlmostEqual(assignments["a1"]["actual_age_s"], 0.52)
        self.assertEqual(assignments["a2"]["history_token"], "a1")
        self.assertIsNone(assignments["b0"])
        self.assertEqual(assignments["b1"]["history_token"], "b0")
        self.assertEqual(diagnostics["matched_frames"], 3)
        self.assertEqual(diagnostics["skipped_frames"], 2)
        self.assertAlmostEqual(diagnostics["actual_age_mean_s"], 0.5066666667)
        self.assertEqual(diagnostics["actual_age_min_s"], 0.5)
        self.assertEqual(diagnostics["actual_age_max_s"], 0.52)

    def test_static_and_motion_differ_only_by_constant_velocity_propagation(self) -> None:
        ordered_frames = [("current", "scene-a"), ("skip", "scene-a")]
        assignments = {
            "current": {
                "history_token": "past",
                "actual_age_s": 1.5,
                "absolute_error_s": 0.0,
            },
            "skip": None,
        }
        caches = {
            camera: {
                "past": [
                    {
                        "sample_token": "past",
                        "translation": [10.0, 20.0, 1.0],
                        "velocity": [2.0, -1.0],
                        "detection_score": 0.8,
                    }
                ]
            }
            for camera in CAMERAS
        }

        static = build_controlled_predictions(
            legacy=_IdentityLegacy,
            caches=caches,
            assignments=assignments,
            ordered_frames=ordered_frames,
            mode="static",
            merge_iou=0.05,
            max_boxes=500,
        )
        motion = build_controlled_predictions(
            legacy=_IdentityLegacy,
            caches=caches,
            assignments=assignments,
            ordered_frames=ordered_frames,
            mode="motion",
            merge_iou=0.05,
            max_boxes=500,
        )

        self.assertEqual(static["current"][0]["translation"], [10.0, 20.0, 1.0])
        self.assertEqual(motion["current"][0]["translation"], [13.0, 18.5, 1.0])
        self.assertAlmostEqual(motion["current"][0]["detection_score"], 0.8)
        self.assertEqual(static["skip"], [])
        self.assertEqual(motion["skip"], [])



if __name__ == "__main__":
    unittest.main()
