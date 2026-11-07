"""Grading for CaLM-Lite, dispatched by question type."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from shared_utils.grading import grade_choice_selection

PROBABILITY_TOLERANCE = 0.01


def grade_binary_classification(pred_str: str, ref_answer: str) -> Tuple[float, Dict[str, Any]]:
    """Check whether the reference ("Yes" or "No") appears in the prediction, ignoring case."""
    pred_normalized = pred_str.strip().lower()
    ref_normalized = ref_answer.strip().lower()
    details: Dict[str, Any] = {"pred_normalized": pred_normalized, "ref_answer": ref_answer}
    return (1.0 if ref_normalized in pred_normalized else 0.0), details


def grade_probability_calculation(
    pred_str: str,
    ref_answer: float,
    tolerance: float = PROBABILITY_TOLERANCE,
) -> Tuple[float, Dict[str, Any]]:
    """Score 1.0 when the prediction parses as a number within ``tolerance`` of the reference."""
    details: Dict[str, Any] = {"ref_value": ref_answer, "tolerance": tolerance}
    try:
        pred_value = float(pred_str.strip())
    except (ValueError, TypeError):
        details.update({"pred_value": None, "pred_extracted": False, "absolute_diff": None})
        return 0.0, details
    absolute_diff = abs(pred_value - ref_answer)
    details.update(
        {"pred_value": pred_value, "pred_extracted": True, "absolute_diff": absolute_diff}
    )
    return (1.0 if absolute_diff <= tolerance else 0.0), details


def grade_calm_lite_answer(
    pred_str: str,
    ref_answer: Union[str, List[str], float, int],
    question_type: str,
) -> Tuple[float, Dict[str, Any]]:
    """Grade one example with the grader for its question type."""
    details: Dict[str, Any] = {"question_type": question_type}
    if question_type == "binary classification":
        score, type_details = grade_binary_classification(pred_str, ref_answer)
    elif question_type == "choice selection":
        score, type_details = grade_choice_selection(pred_str, ref_answer)
    elif question_type == "probability calculation":
        score, type_details = grade_probability_calculation(pred_str, float(ref_answer))
    else:
        raise ValueError(f"Unknown CaLM-Lite question type: {question_type!r}")
    details.update(type_details)
    return score, details
