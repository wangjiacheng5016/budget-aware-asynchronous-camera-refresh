import unittest

from variable_budget_schedule import budget_at_step, parse_budget_sequence


class VariableBudgetPhaseTests(unittest.TestCase):
    def test_phase_offsets_are_cyclic_rotations(self):
        sequence = (2, 3, 4, 3)
        self.assertEqual(
            [budget_at_step(sequence, step, 0) for step in range(4)],
            [2, 3, 4, 3],
        )
        self.assertEqual(
            [budget_at_step(sequence, step, 1) for step in range(4)],
            [3, 4, 3, 2],
        )
        self.assertEqual(
            [budget_at_step(sequence, step, -1) for step in range(4)],
            [3, 2, 3, 4],
        )

    def test_parser_rejects_out_of_range_budget(self):
        self.assertEqual(parse_budget_sequence("2, 3,4"), (2, 3, 4))
        with self.assertRaises(Exception):
            parse_budget_sequence("2,7")

    def test_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            budget_at_step((), 0, 0)


if __name__ == "__main__":
    unittest.main()
