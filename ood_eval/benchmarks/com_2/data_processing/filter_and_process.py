#!/usr/bin/env python3
"""
Filter and process the COM-2 dataset from main.json into cleaned JSONL format.

Actions:
- Read the upstream main.json named by --input and leave it unchanged
- Add idx field (0-based) to all examples
- Filter examples: keep only types "intervention", "counterfactual", "decision"
- Filter examples: remove questions that don't contain "?"
- Filter examples: remove examples with number-based answers (1, 2, 4.)
- Unify answer format: always List[str] (e.g., ["A"] or ["A", "C", "E"])
- Apply prompt templates based on type
- Output only fields: idx, type, question, answer

Writes ./ood_eval/benchmarks/com_2/data/main_filtered_nointerv.jsonl, the prompt file used for
evaluation. It excludes the intervention type and keeps counterfactual and decision.

Run from the repository root:
  python3 -m ood_eval.benchmarks.com_2.data_processing.filter_and_process --input <path>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Union


ALLOWED_TYPES = {"intervention", "counterfactual", "decision"}

# Unicode normalization: map smart quotes to ASCII equivalents
UNICODE_REPLACEMENTS = {
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK -> apostrophe
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK -> apostrophe
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK -> straight quote
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK -> straight quote
    "\u2013": "-",  # EN DASH -> hyphen
    "\u2014": "--", # EM DASH -> double hyphen
    "\u2026": "...",  # HORIZONTAL ELLIPSIS -> three dots
}

# Prompt template for intervention and counterfactual types (single answer)
SINGLE_ANSWER_INSTRUCTION = (
    "\n\n"
    "Think step-by-step to solve this multiple-choice causal reasoning question. "
    "Choose the most plausible option based on commonsense knowledge. "
    "Your final answer should only include a single option letter formatted inside \\boxed{}, "
    "as below: \\boxed{A|B|C|D}."
)

# Prompt template for decision type (multiple answers)
MULTI_ANSWER_INSTRUCTION = (
    "\n\n"
    "Think step-by-step to solve this multiple-choice causal reasoning question. "
    "Choose all the plausible options that can achieve the goal in the question. "
    "Your final answer should only include option letters formatted inside \\boxed{}, "
    "delimited by commas, as below: \\boxed{A, B, C, D}."
)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to ASCII equivalents.

    Replaces smart quotes, en/em dashes, and ellipsis with ASCII versions.
    Preserves accented characters (é, ñ, etc.) as they're valid UTF-8.
    """
    for unicode_char, ascii_char in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_char)
    return text


def read_json(path: str) -> List[Dict]:
    """Read a JSON file containing an array of objects."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    return data


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    """Write records to a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def extract_answer(answer_str: str) -> List[str]:
    """
    Extract answer letters from various formats, always returning a list.

    Handles:
    - "A) explanation" -> ["A"]
    - "A) ..., C) ..." -> ["A", "C"]
    - "A, C, E." -> ["A", "C", "E"]
    - "A." -> ["A"]

    Raises ValueError for number-based answers (1, 2, 4.) to filter them out.
    """
    # Reject number-based answers (1, 2, 3, etc.) - these will be filtered out
    if re.search(r"\b[1-6]\b", answer_str) and not re.search(r"[A-E]\)", answer_str):
        raise ValueError(f"Number-based answer format not supported: {answer_str}")

    # Check for X) patterns - extract ALL letters before ")"
    paren_letters = re.findall(r"([A-E])\)", answer_str)
    if paren_letters:
        return paren_letters

    # Fallback: extract all A-E letters (handles "A, C, E." and "A." formats)
    letters = re.findall(r"[A-E]", answer_str)
    if not letters:
        raise ValueError(f"Could not extract answer letters: {answer_str}")
    return letters


def format_options(options_str: str) -> str:
    """
    Format options string by replacing escaped newlines with actual newlines.

    Input format: "A) Option1  \\nB) Option2  \\nC) Option3..."
    Output format:
    A) Option1
    B) Option2
    C) Option3
    """
    # Replace "  \n" (double space + backslash + n) with actual newline
    formatted = options_str.replace("  \\n", "\n").replace("\\n", "\n")
    # Also handle literal \n sequences
    formatted = formatted.replace("  \n", "\n")
    return formatted.strip()


def apply_prompt_template(record: Dict) -> str:
    """
    Apply the appropriate prompt template based on record type.

    - intervention/counterfactual: question + options + single-answer instruction
    - decision: scenario + question + options + multi-answer instruction
    """
    record_type = record["type"]
    question = record["question"]
    options = format_options(record["options"])

    if record_type == "decision":
        # Decision type includes scenario before question
        scenario = record.get("scenario", "")
        # Clean up scenario formatting
        scenario = scenario.replace("\\n", "\n").strip()
        prompt = f"{scenario}\n\n{question}\n\n{options}{MULTI_ANSWER_INSTRUCTION}"
    else:
        # Intervention and counterfactual types
        prompt = f"{question}\n\n{options}{SINGLE_ANSWER_INSTRUCTION}"

    # Normalize smart quotes and other Unicode characters to ASCII
    prompt = normalize_unicode(prompt)

    return prompt


def should_keep_example(record: Dict) -> bool:
    """
    Check if an example should be kept based on filtering criteria.

    - Type must be in ALLOWED_TYPES
    - Question must contain "?"
    """
    record_type = record.get("type", "")
    question = record.get("question", "")

    if record_type not in ALLOWED_TYPES:
        return False
    if "?" not in question:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Filter and process COM-2 dataset: filtering, answer unification, prompt templates."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the upstream COM-2 main.json file.",
    )

    args = parser.parse_args()

    input_path = args.input

    output_path_nointerv = "./ood_eval/benchmarks/com_2/data/main_filtered_nointerv.jsonl"

    # Validate input exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    # Read original data
    print(f"Reading input: {input_path}")
    records = read_json(input_path)
    num_examples_total = len(records)
    print(f"num_examples_total: {num_examples_total}")

    # Add idx field to all records (0-based)
    for idx, rec in enumerate(records):
        rec["idx"] = idx

    # Filter and process records
    filtered_records: List[Dict] = []
    num_filtered_by_type = 0
    num_filtered_by_question = 0
    num_filtered_by_answer = 0
    type_counts: Dict[str, int] = {}

    for rec in records:
        record_type = rec.get("type", "")
        question = rec.get("question", "")

        # Check type filter
        if record_type not in ALLOWED_TYPES:
            num_filtered_by_type += 1
            continue

        # Check question filter
        if "?" not in question:
            num_filtered_by_question += 1
            continue

        # Extract and unify answer
        try:
            answer = extract_answer(rec["answer"])
        except ValueError as e:
            print(f"Warning: Skipping idx={rec['idx']} due to answer extraction error: {e}")
            num_filtered_by_answer += 1
            continue

        # Apply prompt template
        prompt = apply_prompt_template(rec)

        # Create output record with only required fields
        output_rec = {
            "idx": rec["idx"],
            "type": record_type,
            "question": prompt,
            "answer": answer,
        }

        filtered_records.append(output_rec)

        # Count by type
        type_counts[record_type] = type_counts.get(record_type, 0) + 1

    num_filtered_all = len(filtered_records)

    # Create no-intervention subset
    nointerv_records = [r for r in filtered_records if r["type"] != "intervention"]
    num_filtered_nointerv = len(nointerv_records)

    os.makedirs(os.path.dirname(output_path_nointerv), exist_ok=True)
    print(f"\nWriting {output_path_nointerv}")
    write_jsonl(output_path_nointerv, nointerv_records)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"num_examples_total (original): {num_examples_total}")
    print(f"num_filtered_by_type (not in {ALLOWED_TYPES}): {num_filtered_by_type}")
    print(f"num_filtered_by_question (no '?'): {num_filtered_by_question}")
    print(f"num_filtered_by_answer (invalid format): {num_filtered_by_answer}")
    print(f"\nExamples of all kept types: {num_filtered_all}")
    print(f"main_filtered_nointerv.jsonl: {num_filtered_nointerv} examples")
    print(f"\nBy type, before removing intervention:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
