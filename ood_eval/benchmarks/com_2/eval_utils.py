"""Grading for Com2: one letter on counterfactual questions, F1 on decision questions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from shared_utils.grading import compute_f1_score

OPTION_LETTER_PATTERN = r"[A-F]"


def grade_com2_answer(
    pred_str: str,
    ref_answer: List[str],
    example_type: str,
) -> Tuple[float, Dict[str, Any]]:
    """Grade one example by its type.

    Both types read the set of option letters in the upper-cased prediction. Counterfactual
    questions have one answer: a letter of the set is compared with the reference. Decision
    questions have several answers: the score is the F1 between the two letter sets.
    """
    pred_letters = set(re.findall(OPTION_LETTER_PATTERN, pred_str.upper()))

    if example_type == "counterfactual":
        pred_letter = list(pred_letters)[0] if pred_letters else ""
        ref_letter = ref_answer[0].upper() if ref_answer else ""
        details: Dict[str, Any] = {"pred_letters": pred_letter, "ref_letters": ref_letter}
        return (1.0 if pred_letter == ref_letter else 0.0), details

    if example_type == "decision":
        ref_letters = {letter.upper() for letter in ref_answer}
        details = {"pred_letters": sorted(pred_letters), "ref_letters": sorted(ref_letters)}
        return compute_f1_score(pred_letters, ref_letters), details

    raise ValueError(f"Unknown Com2 example type: {example_type!r}")
