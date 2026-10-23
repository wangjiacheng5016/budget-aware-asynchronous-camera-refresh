from __future__ import annotations

import math
import unittest

from activecam_bev.activecam_core import (
    CAMERAS,
    evaluation_scope,
    local_detection_utility,
    local_nds_like_utility,
    oldest_first_selections,
    propagate_detection,
    refresh_deadline_exceeded,
)


class ActiveCamCoreTest(unittest.TestCase):
    def test_motion_memory_propagates_and_decays_without_mutating_input(self) -> None:
        source = {
            "translation": [10.0, 5.0, 1.0],
            "velocity": [2.0, -1.0],
            "detection_score": 0.8,
        }
        propagated = propagate_detection(source, age_s=0.5, tau_s=1.0, use_velocity=True)
        self.assertEqual(propagated["translation"], [11.0, 4.5, 1.0])
        self.assertAlmostEqual(propagated["detection_score"], 0.8 * math.exp(-0.5))
        self.assertEqual(source["translation"], [10.0, 5.0, 1.0])

    def test_oldest_first_refreshes_disjoint_halves_at_k3(self) -> None:
        frames = [("a", "scene"), ("b", "scene"), ("c", "scene")]
        selections = oldest_first_selections(frames, camera_count=3)
        self.assertEqual(selections["a"], CAMERAS[:3])
        self.assertEqual(selections["b"], CAMERAS[3:])
        self.assertEqual(selections["c"], CAMERAS[:3])

    def test_local_utility_prefers_correct_class_and_location(self) -> None:
        target = [{"detection_name": "car", "translation": [0.0, 0.0, 0.0]}]
        good = [{
            "detection_name": "car", "translation": [0.1, 0.1, 0.0],
            "detection_score": 0.9,
        }]
        bad = [{
            "detection_name": "pedestrian", "translation": [0.1, 0.1, 0.0],
            "detection_score": 0.9,
        }]
        self.assertGreater(local_detection_utility(good, target), local_detection_utility(bad, target))

    def test_nds_like_proxy_rewards_velocity_and_geometry_quality(self) -> None:
        target = [{
            "detection_name": "car",
            "translation": [0.0, 0.0, 0.0],
            "velocity": [4.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "size": [2.0, 4.0, 1.5],
        }]
        good = [{
            **target[0],
            "translation": [0.1, 0.0, 0.0],
            "detection_score": 0.9,
        }]
        stale = [{
            **good[0],
            "translation": [2.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
        }]
        self.assertGreater(
            local_nds_like_utility(good, target),
            local_nds_like_utility(stale, target),
        )

    def test_deadline_allows_timestamp_jitter_but_rejects_infeasible_limit(self) -> None:
        values = {"age_s": 0.500435, "next_interval_s": 0.499868}
        self.assertFalse(refresh_deadline_exceeded(**values, max_age_s=1.0))
        self.assertTrue(refresh_deadline_exceeded(**values, max_age_s=0.6))

    def test_only_official_full_validation_is_formal(self) -> None:
        self.assertFalse(evaluation_scope("v1.0-trainval", "val")[0])
        self.assertTrue(evaluation_scope("v1.0-mini", "mini_val")[0])


if __name__ == "__main__":
    unittest.main()
