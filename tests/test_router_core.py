import math
import unittest

from activecam_bev.activecam_core import CAMERAS
from activecam_bev.router_core import (
    STATE_FEATURES,
    build_pre_action_state,
    detection_summary,
    enumerate_actions,
    feasible_actions,
    teacher_q_values,
)


class RouterCoreTest(unittest.TestCase):
    def test_k3_has_twenty_candidate_sets(self):
        self.assertEqual(len(enumerate_actions(3)), 20)

    def test_empty_memory_keeps_uncertainty_floor(self):
        summary = detection_summary([], age_s=0.5, max_age_s=1.5, uncertainty_tau_s=1.0)
        self.assertGreaterEqual(summary["uncertainty"], 1.0)

    def test_state_rejects_current_counterfactual_cache(self):
        memory = {camera: (2_000_000, []) for camera in CAMERAS}
        with self.assertRaises(AssertionError):
            build_pre_action_state(
                memory=memory,
                decision_timestamp=2_000_000,
                previous_action=CAMERAS[:3],
                max_age_s=1.5,
                uncertainty_tau_s=1.0,
            )

    def test_state_shape_and_past_sources(self):
        memory = {camera: (1_000_000, []) for camera in CAMERAS}
        payload = build_pre_action_state(
            memory=memory,
            decision_timestamp=1_500_000,
            previous_action=CAMERAS[:3],
            max_age_s=1.5,
            uncertainty_tau_s=1.0,
        )
        self.assertEqual(len(payload["state"]), 6)
        self.assertEqual(len(payload["state"][0]), len(STATE_FEATURES))
        self.assertTrue(all(value < 1_500_000 for value in payload["source_timestamps"]))

    def test_shield_requires_old_views(self):
        actions = enumerate_actions(3)
        ages = [1.02, 1.02, 1.02, 0.0, 0.0, 0.0]
        mask = feasible_actions(
            actions=actions,
            ages_s=ages,
            planned_interval_s=0.5,
            max_age_s=1.5,
        )
        selected = [actions[index] for index, valid in enumerate(mask) if valid]
        self.assertEqual(selected, [CAMERAS[:3]])

    def test_bootstrap_uses_exact_budget_without_free_full6(self):
        actions = enumerate_actions(3)
        first = feasible_actions(
            actions=actions,
            ages_s=[1.5] * 6,
            planned_interval_s=0.5,
            max_age_s=1.5,
            initialized=[False] * 6,
        )
        self.assertTrue(all(first))
        second = feasible_actions(
            actions=actions,
            ages_s=[0.5, 0.5, 0.5, 1.5, 1.5, 1.5],
            planned_interval_s=0.5,
            max_age_s=1.5,
            initialized=[True, True, True, False, False, False],
        )
        selected = [actions[index] for index, valid in enumerate(second) if valid]
        self.assertEqual(selected, [CAMERAS[3:]])

    def test_teacher_masks_infeasible_candidates(self):
        actions = enumerate_actions(3)
        feasible = [False] * len(actions)
        feasible[2] = True
        q, debt = teacher_q_values(
            actions=actions,
            current_utilities=[0.1] * len(actions),
            feasible=feasible,
            ages_s=[0.5] * 6,
            uncertainties=[1.0] * 6,
            previous_action=actions[0],
            planned_interval_s=0.5,
            soft_age_s=1.0,
            age_weight=0.2,
            switch_weight=0.01,
        )
        self.assertTrue(math.isfinite(q[2]))
        self.assertTrue(all(not math.isfinite(value) for i, value in enumerate(q) if i != 2))
        self.assertEqual(len(debt), len(actions))


if __name__ == "__main__":
    unittest.main()
