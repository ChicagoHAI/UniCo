"""Answer extraction and grading for the UniCo in-distribution query types."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from shared_utils.grading import (
    compute_exact_match,
    compute_f1_score,
    extract_last_boxed,
    grade_choice_selection,
)

GRAPH_ONLY_QUERY_TYPES = {
    "identification",
    "backdoor",
    "frontdoor",
    "independence_test",
    "markov_blanket",
}
NUMERICAL_TOLERANCE = 0.0001


def grade_numerical_answer(
    pred_str: str,
    ref_answer: float,
    tolerance: float = NUMERICAL_TOLERANCE,
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


def grade_binary_answer(pred_str: str, ref_answer: str) -> Tuple[bool, Dict[str, Any]]:
    """Check whether the reference ("Yes" or "No") appears in the prediction, ignoring case."""
    pred_lower = pred_str.lower()
    ref_lower = ref_answer.lower()
    details: Dict[str, Any] = {"pred_normalized": pred_lower, "ref_normalized": ref_lower}
    return ref_lower in pred_lower, details


def parse_answer_letters(answer_str: str) -> Set[str]:
    """Collect the option letters A-D from a string such as ``"A, C"``."""
    return set(re.findall(r"[A-D]", answer_str.upper()))


def grade_multi_answer(
    pred_str: str,
    ref_answer: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    """Return the F1 and exact-match scores of a multi-answer question."""
    pred_letters = parse_answer_letters(pred_str)
    ref_letters = {letter.upper() for letter in ref_answer}
    details: Dict[str, Any] = {
        "pred_letters": sorted(pred_letters),
        "ref_letters": sorted(ref_letters),
    }
    return (
        compute_f1_score(pred_letters, ref_letters),
        compute_exact_match(pred_letters, ref_letters),
        details,
    )


def extract_and_grade_prediction(
    data_item: Dict[str, Any],
    response: str,
) -> Tuple[str, bool, Any, float, Dict[str, Any]]:
    """Grade one response; return prediction, extraction flag, reference, score, and details."""
    query_type = data_item["query_type"]
    query_format = data_item.get("query_format")
    pred_str, extraction_success = extract_last_boxed(response)

    if query_type not in GRAPH_ONLY_QUERY_TYPES:
        ref_answer = float(data_item["answer"])
        score, details = grade_numerical_answer(pred_str, ref_answer)
        return pred_str, extraction_success, ref_answer, score, details

    if query_format == "binary":
        ref_answer = str(data_item["answer"]).strip()
        correct, details = grade_binary_answer(pred_str, ref_answer)
        return pred_str, extraction_success, ref_answer, 1.0 if correct else 0.0, details

    if query_format == "multi_choice":
        ref_answer = list(data_item["answer"])
        score, details = grade_choice_selection(pred_str, ref_answer)
        return pred_str, extraction_success, ref_answer, score, details

    if query_format == "multi_answer":
        ref_answer = list(data_item["answer"])
        f1_score, exact_match, details = grade_multi_answer(pred_str, ref_answer)
        details["f1_score"] = f1_score
        details["exact_match"] = exact_match
        return pred_str, extraction_success, ref_answer, exact_match, details

    raise ValueError(f"Unknown graph-only query_format: {query_format}")
