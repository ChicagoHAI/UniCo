#!/usr/bin/env python3
"""
Process counterbench dataset: merge two JSON files and apply prompt templates.

Actions:
- Read data_balanced_alpha_V1.json and data_balanced_backdoor_V2.json
- Merge into single dataset with 0-based idx field
- Apply prompt template combining given_info and question
- Write JSONL with fields: idx, question, answer, type
- Write the prompt file used for evaluation to ./ood_eval/benchmarks/counterbench/data/

Run from the repository root:
  python3 -m ood_eval.benchmarks.counterbench.data_processing.process_counterbench --input_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterable, List


# Prompt template instruction
INSTRUCTION = (
    "\n\n"
    "Think step-by-step to solve this causal reasoning question. "
    "Your final answer should only include a single Yes/No formatted inside \\boxed{}, "
    "as below: \\boxed{Yes|No}"
)


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


def format_prompt(record: Dict) -> str:
    """
    Format a record into the evaluation prompt using the template.

    Template structure:
        {given_info} {question}

        {instruction}

    Args:
        record: Dictionary with given_info and question fields

    Returns:
        Formatted prompt string

    Raises:
        KeyError: If required fields are missing
    """
    required = ["given_info", "question"]
    for field in required:
        if field not in record:
            raise KeyError(f"Missing required field: {field}")

    given_info = record["given_info"]
    question = record["question"]

    prompt = f"{given_info} {question}{INSTRUCTION}"

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Process counterbench dataset: merge JSON files and apply prompt templates."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the two upstream JSON files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir

    # Input file paths
    alpha_path = os.path.join(input_dir, "data_balanced_alpha_V1.json")
    backdoor_path = os.path.join(input_dir, "data_balanced_backdoor_V2.json")

    # Output paths
    output_jsonl = "./ood_eval/benchmarks/counterbench/data/data_balanced_all.jsonl"

    # Validate inputs exist
    if not os.path.isfile(alpha_path):
        raise FileNotFoundError(f"Input JSON not found: {alpha_path}")
    if not os.path.isfile(backdoor_path):
        raise FileNotFoundError(f"Input JSON not found: {backdoor_path}")

    # Read input data
    print(f"Reading input: {alpha_path}")
    alpha_records = read_json(alpha_path)
    num_alpha = len(alpha_records)
    print(f"  num_examples (alpha): {num_alpha}")

    print(f"Reading input: {backdoor_path}")
    backdoor_records = read_json(backdoor_path)
    num_backdoor = len(backdoor_records)
    print(f"  num_examples (backdoor): {num_backdoor}")

    # Merge records
    all_records = alpha_records + backdoor_records
    num_examples_total = len(all_records)
    print(f"\nnum_examples_total: {num_examples_total}")

    # Process records and create JSONL records
    jsonl_records: List[Dict] = []
    type_counter: Counter = Counter()
    answer_counter: Counter = Counter()

    for idx, rec in enumerate(all_records):
        # Validate required fields
        for field in ["given_info", "question", "answer", "type"]:
            if field not in rec:
                raise KeyError(f"Missing '{field}' field in record at index {idx}")

        # Format prompt
        prompt = format_prompt(rec)

        # Capitalize answer: "yes" -> "Yes", "no" -> "No"
        answer = rec["answer"].capitalize()

        # Create JSONL record with required fields
        jsonl_rec = {
            "idx": idx,
            "question": prompt,
            "answer": answer,
            "type": rec["type"],
        }
        jsonl_records.append(jsonl_rec)

        # Track statistics
        type_counter[rec["type"]] += 1
        answer_counter[answer] += 1

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    write_jsonl(output_jsonl, jsonl_records)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"num_examples_total: {num_examples_total}")
    print(f"  num_alpha: {num_alpha}")
    print(f"  num_backdoor: {num_backdoor}")
    print("\nBy type:")
    for typ, count in sorted(type_counter.items()):
        print(f"  {typ}: {count}")
    print("\nBy answer:")
    for ans, count in sorted(answer_counter.items()):
        print(f"  {ans}: {count}")
    print(f"\nJSONL written to: {output_jsonl}")


if __name__ == "__main__":
    main()
