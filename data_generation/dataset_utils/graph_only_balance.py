"""Helpers for balancing graph-only query answer labels."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


BINARY_LABELS = ["Yes", "No"]
MULTI_CHOICE_LABELS = ["A", "B", "C", "D"]


def compute_target_answer_counts(labels: List[str], num_examples: int) -> Dict[str, int]:
    base = num_examples // len(labels)
    remainder = num_examples % len(labels)
    return {
        label: base + (1 if idx < remainder else 0)
        for idx, label in enumerate(labels)
    }


def select_most_underfilled_label(
    target_counts: Dict[str, int],
    current_counts: Dict[str, int],
) -> Optional[str]:
    remaining = [
        label for label, target in target_counts.items()
        if current_counts.get(label, 0) < target
    ]
    if not remaining:
        return None
    return max(
        remaining,
        key=lambda label: (target_counts[label] - current_counts.get(label, 0), -list(target_counts).index(label)),
    )


def count_answer_labels(records: List[Dict], result_key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        label = record[result_key]["correct_options"]
        counts[label] = counts.get(label, 0) + 1
    return counts
