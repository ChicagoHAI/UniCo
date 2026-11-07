#!/usr/bin/env python3
"""Generate the 3,900-row CALM-Lite aggregated JSONL file.

This script aligns question construction with
`calm/data_processing/generate_questions.py` by first generating the canonical
`basic` prompts for each included subset, then converting those prompts into the
aggregated format used by downstream evaluation.

Copy this file to calm/data_processing/ inside a checkout of the CaLM repository and run it there,
where it imports that repository's question generator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
CALM_DIR = SCRIPT_DIR.parent
if str(CALM_DIR) not in sys.path:
    sys.path.insert(0, str(CALM_DIR))

from data_processing.generate_questions import generate_question_list

DEFAULT_DATASET_ROOT = Path("../../calm_lite_dataset")
DEFAULT_OUTPUT_PATH = Path("../../calm_lite_aggregated.jsonl")

THINK_STEP_BY_STEP = "Think step-by-step to solve this causal reasoning question."
YES_NO_INSTRUCTION = (
    "Your final answer should only include a single Yes/No formatted inside "
    "\\boxed{}, as below: \\boxed{Yes|No}."
)
NUMERIC_INSTRUCTION = (
    "Your final answer should be rounded to four decimal places and included "
    "inside \\boxed{}, as below: \\boxed{<your final answer>}."
)

ZERO_BASED_CHOICE_SUBSETS = {"PCD-C_COPA_EN", "PCD-C_E-CARE_EN"}


@dataclass(frozen=True)
class SubsetSpec:
    subset: str
    relpath: str
    question_type: str


SUBSET_SPECS: List[SubsetSpec] = [
    SubsetSpec("AR-B_CaLM-AR_EN", "causal_discovery/abstract_reasoning/AR-B_CaLM-AR_EN.json", "binary classification"),
    SubsetSpec("CA-B_FA_EN", "causal_discovery/causal_attribution/CA-B_FA_EN.json", "binary classification"),
    SubsetSpec("CA-B_FP_EN", "causal_discovery/causal_attribution/CA-B_FP_EN.json", "binary classification"),
    SubsetSpec("ECI-B_CTB_EN", "causal_discovery/event_causality_identification/ECI-B_CTB_EN.json", "binary classification"),
    SubsetSpec("ECI-B_ESC_EN", "causal_discovery/event_causality_identification/ECI-B_ESC_EN.json", "binary classification"),
    SubsetSpec("ECI-B_MAVEN-ERE_EN", "causal_discovery/event_causality_identification/ECI-B_MAVEN-ERE_EN.json", "binary classification"),
    SubsetSpec("PCD-B_COPA_EN", "causal_discovery/pairwise_causal_discovery/PCD-B_COPA_EN.json", "binary classification"),
    SubsetSpec("PCD-B_E-CARE_EN", "causal_discovery/pairwise_causal_discovery/PCD-B_E-CARE_EN.json", "binary classification"),
    SubsetSpec("PCD-C_COPA_EN", "causal_discovery/pairwise_causal_discovery/PCD-C_COPA_EN.json", "choice selection"),
    SubsetSpec("PCD-C_E-CARE_EN", "causal_discovery/pairwise_causal_discovery/PCD-C_E-CARE_EN.json", "choice selection"),
    SubsetSpec("CR-C_CRASS_EN", "counterfactual/counterfactual_reasoning/CR-C_CRASS_EN.json", "choice selection"),
    SubsetSpec("ETT-B_ETT-natural_EN", "counterfactual/effect_of_the_treatment_on_the_treated/ETT-B_ETT-natural_EN.json", "binary classification"),
    SubsetSpec("ETT-P_ETT-basic_EN", "counterfactual/effect_of_the_treatment_on_the_treated/ETT-P_ETT-basic_EN.json", "probability calculation"),
    SubsetSpec("ETT-P_ETT-hard_EN", "counterfactual/effect_of_the_treatment_on_the_treated/ETT-P_ETT-hard_EN.json", "probability calculation"),
    SubsetSpec("NDE-B_NDE-natural_EN", "counterfactual/natural_direct_effect/NDE-B_NDE-natural_EN.json", "binary classification"),
    SubsetSpec("NDE-P_NDE-basic_EN", "counterfactual/natural_direct_effect/NDE-P_NDE-basic_EN.json", "probability calculation"),
    SubsetSpec("NDE-P_NDE-hard_EN", "counterfactual/natural_direct_effect/NDE-P_NDE-hard_EN.json", "probability calculation"),
    SubsetSpec("NIE-B_NIE-natural_EN", "counterfactual/natural_indirect_effect/NIE-B_NIE-natural_EN.json", "binary classification"),
    SubsetSpec("NIE-P_NIE-basic_EN", "counterfactual/natural_indirect_effect/NIE-P_NIE-basic_EN.json", "probability calculation"),
    SubsetSpec("NIE-P_NIE-hard_EN", "counterfactual/natural_indirect_effect/NIE-P_NIE-hard_EN.json", "probability calculation"),
    SubsetSpec("PN-P_PN-basic_EN", "counterfactual/probability_of_necessity/PN-P_PN-basic_EN.json", "probability calculation"),
    SubsetSpec("PN-P_PN-hard_EN", "counterfactual/probability_of_necessity/PN-P_PN-hard_EN.json", "probability calculation"),
    SubsetSpec("PS-P_PS-basic_EN", "counterfactual/probability_of_sufficiency/PS-P_PS-basic_EN.json", "probability calculation"),
    SubsetSpec("PS-P_PS-hard_EN", "counterfactual/probability_of_sufficiency/PS-P_PS-hard_EN.json", "probability calculation"),
    SubsetSpec("ATE-B_ATE-natural_EN", "intervention/average_treatment_effect/ATE-B_ATE-natural_EN.json", "binary classification"),
    SubsetSpec("ATE-P_ATE-basic_EN", "intervention/average_treatment_effect/ATE-P_ATE-basic_EN.json", "probability calculation"),
    SubsetSpec("ATE-P_ATE-hard_EN", "intervention/average_treatment_effect/ATE-P_ATE-hard_EN.json", "probability calculation"),
    SubsetSpec("BAS-C_max-BAS_EN", "intervention/backdoor_adjustment_set/BAS-C_max-BAS_EN.json", "choice selection"),
    SubsetSpec("BAS-C_min-BAS_EN", "intervention/backdoor_adjustment_set/BAS-C_min-BAS_EN.json", "choice selection"),
    SubsetSpec("BAS-C_mix-BAS_EN", "intervention/backdoor_adjustment_set/BAS-C_mix-BAS_EN.json", "choice selection"),
    SubsetSpec("CEI-B_0.2-UC_EN", "intervention/causal_effect_identification/CEI-B_0.2-UC_EN.json", "binary classification"),
    SubsetSpec("CEI-B_0.4-UC_EN", "intervention/causal_effect_identification/CEI-B_0.4-UC_EN.json", "binary classification"),
    SubsetSpec("CEI-B_0.6-UC_EN", "intervention/causal_effect_identification/CEI-B_0.6-UC_EN.json", "binary classification"),
    SubsetSpec("CEI-B_0.8-UC_EN", "intervention/causal_effect_identification/CEI-B_0.8-UC_EN.json", "binary classification"),
    SubsetSpec("CDE-B_CDE-natural_EN", "intervention/controlled_direct_effect/CDE-B_CDE-natural_EN.json", "binary classification"),
    SubsetSpec("CDE-P_CDE-basic_EN", "intervention/controlled_direct_effect/CDE-P_CDE-basic_EN.json", "probability calculation"),
    SubsetSpec("CDE-P_CDE-hard_EN", "intervention/controlled_direct_effect/CDE-P_CDE-hard_EN.json", "probability calculation"),
    SubsetSpec("FAS-C_FAS_EN", "intervention/frontdoor_adjustment_set/FAS-C_FAS_EN.json", "choice selection"),
    SubsetSpec("IV-C_CaLM-IV_EN", "intervention/instrumental_variable/IV-C_CaLM-IV_EN.json", "choice selection"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def answer_instruction_for_letters(letter_count: int) -> str:
    if letter_count == 2:
        pattern = "A|B"
        multi = "A,B"
    elif letter_count == 3:
        pattern = "A|B|C"
        multi = "A,B,C"
    else:
        pattern = "A|B|C|D"
        multi = "A,B,C,D"
    return (
        "Your final answer should only include the option letter formatted inside "
        f"\\boxed{{}}, as below: \\boxed{{{pattern}}}. If there are more than one "
        "options in your answer, delimit them by commas inside \\boxed{}, as below: "
        f"\\boxed{{{multi}}}."
    )


def normalize_question_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def strip_answer_block(lines: List[str]) -> List[str]:
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        last = lines[-1].strip()
        if last.startswith("Answer (") or last.startswith("Provide the calculation result"):
            lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return [line.rstrip() for line in lines]


def convert_choice_prompt(base_question: str) -> str:
    lines = strip_answer_block(base_question.splitlines())
    option_lines = [line for line in lines if line.startswith("Option ")]
    prefix_lines = [line for line in lines if not line.startswith("Option ")]
    converted_options = []
    for idx, line in enumerate(option_lines):
        _, value = line.split(":", 1)
        converted_options.append(f"Option ({chr(ord('A') + idx)}):{value}")
    return normalize_question_text("\n".join(
        prefix_lines + ["", THINK_STEP_BY_STEP] + converted_options + [answer_instruction_for_letters(len(converted_options))]
    ))


def convert_binary_prompt(base_question: str) -> str:
    lines = strip_answer_block(base_question.splitlines())
    return normalize_question_text("\n".join(lines + ["", THINK_STEP_BY_STEP, YES_NO_INSTRUCTION]))


def convert_probability_prompt(base_question: str) -> str:
    lines = strip_answer_block(base_question.splitlines())
    return normalize_question_text("\n".join(lines + ["", THINK_STEP_BY_STEP, NUMERIC_INSTRUCTION]))


def normalize_binary_answer(gt_answer: Any) -> str:
    if isinstance(gt_answer, (int, float)):
        return "Yes" if int(gt_answer) == 1 else "No"
    lowered = str(gt_answer).strip().lower()
    if lowered in {"yes", "y", "true", "1"}:
        return "Yes"
    if lowered in {"no", "n", "false", "0"}:
        return "No"
    raise ValueError(f"Unsupported binary ground truth: {gt_answer!r}")


def normalize_choice_answer(subset: str, gt_answer: Any) -> List[str]:
    if subset in ZERO_BASED_CHOICE_SUBSETS:
        value = int(str(gt_answer).strip())
        return [chr(ord("A") + value)]
    value = str(gt_answer).strip()
    if value.isdigit():
        return [chr(ord("A") + int(value) - 1)]
    if value.isalpha():
        return [value.upper()]
    raise ValueError(f"Unsupported choice ground truth for {subset}: {gt_answer!r}")


def normalize_probability_answer(gt_answer: Any) -> float:
    return float(gt_answer)


def normalize_answer(subset: str, question_type: str, gt_answer: Any) -> Any:
    if question_type == "binary classification":
        return normalize_binary_answer(gt_answer)
    if question_type == "choice selection":
        return normalize_choice_answer(subset, gt_answer)
    if question_type == "probability calculation":
        return normalize_probability_answer(gt_answer)
    raise ValueError(f"Unsupported question type: {question_type}")


def convert_question(question_type: str, base_question: str) -> str:
    if question_type == "binary classification":
        return convert_binary_prompt(base_question)
    if question_type == "choice selection":
        return convert_choice_prompt(base_question)
    if question_type == "probability calculation":
        return convert_probability_prompt(base_question)
    raise ValueError(f"Unsupported question type: {question_type}")


def generate_records(dataset_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for spec in SUBSET_SPECS:
        dataset_path = dataset_root / spec.relpath
        items = generate_question_list(dataset_path, "basic")
        if len(items) != 100:
            raise ValueError(f"Expected 100 rows in {dataset_path}, found {len(items)}")
        for item in items:
            records.append(
                {
                    "idx": item["index"],
                    "question": convert_question(spec.question_type, item["question"]),
                    "answer": normalize_answer(spec.subset, spec.question_type, item["gt_answer"]),
                    "question_type": spec.question_type,
                    "subset": spec.subset,
                }
            )
    return records


def main() -> None:
    args = parse_args()
    records = generate_records(args.dataset_root)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
