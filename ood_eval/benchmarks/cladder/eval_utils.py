"""Grading for CLadder: the reference ("Yes" or "No") must appear in the prediction."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def grade_cladder_answer(pred_str: str, ref_answer: str) -> Tuple[float, Dict[str, Any]]:
    """Check whether the reference appears in the extracted answer, ignoring case."""
    pred_lower = pred_str.lower().strip()
    ref_lower = ref_answer.lower().strip()
    details: Dict[str, Any] = {"pred_normalized": pred_lower, "ref_normalized": ref_lower}
    return (1.0 if ref_lower in pred_lower else 0.0), details
