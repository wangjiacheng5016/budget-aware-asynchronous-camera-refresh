from __future__ import annotations

import unittest
from pathlib import Path

from activecam_bev.official_history_protocol import (
    MemoryProvenance,
    audit_protocol_a_frame,
    build_protocol_a_predictions,
    camera_capture_timestamps_us,
    official_val_stop_gate,
)


class OfficialHistoryProtocolTests(unittest.TestCase):
    def test_protocol_a_uses_actual_camera_timestamps_and_drops_over_half_second(self):
        cameras = ("A", "B", "C", "D", "E", "F")
        current = {camera: 2_000_000 for camera in cameras}
        memory = {
            "D": MemoryProvenance(1_500_000, 1_500_000),  # Exact boundary: legal.
            "E": MemoryProvenance(1_499_999, 1_500_000),  # 0.500001 s: illegal.
            "F": MemoryProvenance(1_600_000, 1_600_000),
        }

        audit = audit_protocol_a_frame(
            cameras=cameras,
            selected=("A", "B", "C"),
            current_capture_timestamps_us=current,
            memory=memory,
        )

        self.assertEqual(audit.valid_memory_cameras, ("D", "F"))
        self.assertEqual(audit.stale_memory_cameras, ("E",))
        self.assertEqual(audit.past_image_count, 2)
        self.assertEqual(audit.max_past_image_age_s, 0.5)
        self.assertEqual(audit.history_window_violations, 0)
        self.assertEqual(audit.past_image_count_violations, 0)

    def test_protocol_a_predictions_never_merge_stale_or_unselected_current_cache(self):
        caches = {
            camera: {"t1": [{"source": f"current-{camera}"}]}
            for camera in ("A", "B", "C", "D", "E", "F")
        }
        memory = {
            "D": (MemoryProvenance(1_500_000, 1_500_000), [{"source": "old-D"}]),
            "E": (MemoryProvenance(1_499_999, 1_500_000), [{"source": "old-E"}]),
        }

        predictions, audit = build_protocol_a_predictions(
            token="t1",
            current_sample_timestamp_us=2_100_000,
            cameras=("A", "B", "C", "D", "E", "F"),
            selected=("A", "B", "C"),
            current_capture_timestamps_us={camera: 2_000_000 for camera in caches},
            current_only_caches=caches,
            memory=memory,
            propagate=lambda records, age: [
                {**record, "motion_age_s": age} for record in records
            ],
            merge=lambda sources: [record for _camera, records in sources for record in records],
        )

        self.assertEqual(
            [record["source"] for record in predictions],
            ["current-A", "current-B", "current-C", "old-D"],
        )
        self.assertEqual(audit.valid_memory_cameras, ("D",))
        self.assertEqual(audit.stale_memory_cameras, ("E",))
        old_d = next(record for record in predictions if record["source"] == "old-D")
        self.assertEqual(old_d["motion_age_s"], 0.6)

    def test_stop_gate_keeps_k4_primary_and_k3_aggressive_claims_separate(self):
        decision = official_val_stop_gate(
            metrics={
                6: {"NDS": 0.300, "mAP": 0.210},
                4: {"NDS": 0.295, "mAP": 0.200},
                3: {"NDS": 0.292, "mAP": 0.185},
            },
            audits={
                k: {
                    "history_window_violations": 0,
                    "past_image_count_violations": 0,
                    "unselected_current_reads": 0,
                    "previous_branch_valid_events": 0,
                }
                for k in (3, 4, 6)
            },
        )

        self.assertTrue(decision["global_compliance_gate"])
        self.assertTrue(decision["k4_primary_gate"])
        self.assertFalse(decision["k3_aggressive_gate"])
        self.assertEqual(decision["decision"], "PROCEED_K4_ONLY")

    def test_capture_timestamps_are_read_from_each_camera_sample_data(self):
        class FakeNuScenes:
            records = {
                ("sample", "t1"): {
                    "data": {"A": "a-data", "B": "b-data"}
                },
                ("sample_data", "a-data"): {"timestamp": 1_990_000},
                ("sample_data", "b-data"): {"timestamp": 1_970_000},
            }

            def get(self, table, token):
                return self.records[(table, token)]

        self.assertEqual(
            camera_capture_timestamps_us(FakeNuScenes(), "t1", ("A", "B")),
            {"A": 1_990_000, "B": 1_970_000},
        )

    def test_current_only_exporter_does_not_request_or_forward_previous_image(self):
        source = (
            Path(__file__).parents[1] / "src/activecam_bev/export_current_only_camera_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn("need_temporal_inputs=False", source)
        self.assertIn("prev_valid = torch.zeros", source)
        self.assertIn("model.forward_outputs(*current_inputs, *masked_temporal_inputs)", source)
        self.assertNotIn("move_student_item", source)
        self.assertNotIn("item[18]", source)

    def test_current_only_exporter_registers_legacy_before_benchmark_import(self):
        source = (
            Path(__file__).parents[1] / "src/activecam_bev/export_current_only_camera_cache.py"
        ).read_text(encoding="utf-8")
        legacy_insert = source.index('sys.path.insert(0, str(legacy_src))')
        benchmark_import = source.index(
            "from benchmark_activecam_end_to_end import decode_current_camera"
        )
        self.assertLess(legacy_insert, benchmark_import)




if __name__ == "__main__":
    unittest.main()
