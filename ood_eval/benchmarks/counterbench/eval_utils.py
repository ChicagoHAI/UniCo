"""Grading for CounterBench: the reference must appear in the prediction."""

from __future__ import annotations


def grade_counterbench_answer(pred_str: str, ref_answer: str) -> bool:
    """Check whether the reference appears in the extracted answer, ignoring case."""
    return ref_answer.lower() in pred_str.lower()
