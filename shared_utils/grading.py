"""Answer extraction and scoring primitives used by both evaluators."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from shared_utils.prompt_utils import find_all_boxed


def extract_last_boxed(text: str) -> Tuple[str, bool]:
    """Return the content of the last ``\\boxed{}`` and whether one was found."""
    boxed_contents = find_all_boxed(text)
    if not boxed_contents:
        return "", False
    return boxed_contents[-1], True


def compute_f1_score(pred_set: Set[str], ref_set: Set[str]) -> float:
    """Compute F1 between two sets. Two empty sets count as a perfect match."""
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    intersection = pred_set & ref_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred_set: Set[str], ref_set: Set[str]) -> float:
    """Return 1.0 if the two sets are equal, otherwise 0.0."""
    return 1.0 if pred_set == ref_set else 0.0


def grade_choice_selection(
    pred_str: str,
    ref_answer: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """Compare the first uppercase letter A-D in the prediction with a one-element reference."""
    letters = re.findall(r"[A-D]", pred_str)
    pred_letter = letters[0] if letters else ""
    ref_letter = ref_answer[0] if ref_answer else ""
    details: Dict[str, Any] = {"pred_letter": pred_letter, "ref_letter": ref_letter}
    return (1.0 if pred_letter == ref_letter else 0.0), details
