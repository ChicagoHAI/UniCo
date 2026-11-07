"""Grading for Corr2Cause: containment per example, binary F1 over the dataset."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def grade_corr2cause_answer(pred_str: str, ref_answer: str) -> Tuple[bool, Dict[str, Any]]:
    """Check whether the reference appears in the extracted answer, ignoring case."""
    pred_lower = pred_str.lower()
    ref_lower = ref_answer.lower()
    details: Dict[str, Any] = {"pred_normalized": pred_lower, "ref_normalized": ref_lower}
    return ref_lower in pred_lower, details


def compute_binary_f1(
    predictions: List[str],
    references: List[str],
    positive_label: str = "Yes",
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for the positive label."""
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        pred_is_positive = positive_label.lower() in pred.lower()
        ref_is_positive = ref.lower() == positive_label.lower()
        if pred_is_positive and ref_is_positive:
            tp += 1
        elif pred_is_positive and not ref_is_positive:
            fp += 1
        elif not pred_is_positive and ref_is_positive:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
