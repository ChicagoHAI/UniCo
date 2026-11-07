"""Grading for BBEH Causal Understanding: the reference must appear in the prediction."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def grade_bbeh_answer(pred_str: str, ref_answer: str) -> Tuple[float, Dict[str, Any]]:
    """Check whether the reference appears in the extracted answer, ignoring case."""
    correct = ref_answer.lower() in pred_str.lower()
    details: Dict[str, Any] = {"pred": pred_str, "ref": ref_answer, "correct": correct}
    return (1.0 if correct else 0.0), details
