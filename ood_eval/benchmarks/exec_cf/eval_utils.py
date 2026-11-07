"""Answer extraction and grading for Executable Counterfactuals."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from shared_utils.grading import compute_exact_match, compute_f1_score

# Matches "y = 3", "`y` is 3, 4", and "y would be 3" and captures the numbers.
_Y_VALUE_PATTERN = re.compile(
    r"(?:`)?[yY](?:`)?\s*(?:=|is|would\s+be)\s*(?:`)?"
    r"[^\d`]*?"
    r"(-?\d+(?:\s*,\s*-?\d+)*)"
    r"(?:`)?",
    re.I,
)


def _process_answer_string(answer_str: str) -> List[str]:
    """Split on commas, remove whitespace, and normalize integers (``"03"`` becomes ``"3"``)."""
    answers: List[str] = []
    for part in answer_str.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        try:
            cleaned = str(int(cleaned))
        except ValueError:
            pass  # A non-integer answer stays as written and fails the set comparison.
        answers.append(cleaned)
    return answers


def extract_exec_cf_answer(response: str) -> Tuple[List[str], bool]:
    """Extract answers from the last ``\\boxed{}`` and from any ``y = ...`` phrase inside it.

    When a response has a box, the ``y = ...`` search runs on the box content only.
    Otherwise it runs on the whole response, and the last match wins.
    """
    answers: List[str] = []
    box_hits = re.findall(r"\\boxed\{([^{}]*)\}", response, re.I)
    if box_hits:
        answers = _process_answer_string(box_hits[-1])
        response = box_hits[-1]

    y_hits = _Y_VALUE_PATTERN.findall(response)
    if y_hits:
        answers = _process_answer_string(y_hits[-1])
    return answers, len(answers) > 0


def grade_exec_cf_answer(
    pred_answers: List[str],
    ref_answer: List[int],
) -> Tuple[float, float, Dict[str, Any]]:
    """Return the F1 and exact-match scores between predicted and reference answer sets."""
    pred_set = set(pred_answers)
    ref_set = {str(value) for value in ref_answer}
    details: Dict[str, Any] = {
        "pred_answers": sorted(pred_set),
        "ref_answers": sorted(ref_set),
    }
    return compute_f1_score(pred_set, ref_set), compute_exact_match(pred_set, ref_set), details
