from __future__ import annotations

import argparse


def parse_budget_sequence(value: str, camera_count: int = 6) -> tuple[int, ...]:
    try:
        sequence = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("budget sequence must contain integers") from error
    if not sequence or any(not 1 <= budget <= camera_count for budget in sequence):
        raise argparse.ArgumentTypeError(
            f"every budget must be between 1 and {camera_count}"
        )
    return sequence


def budget_at_step(sequence: tuple[int, ...], step: int, phase_offset: int) -> int:
    if not sequence:
        raise ValueError("budget sequence must not be empty")
    return sequence[(step + phase_offset) % len(sequence)]
