#!/usr/bin/env python3
"""
Process CLadder-full_v1.5_default.jsonl: rename id->idx, reindex, format prompts.

Actions:
- Read the upstream CLadder-full_v1.5_default.jsonl named by --input
- Rename "id" to "idx" and reindex from 0
- Capitalize label ("yes" -> "Yes", "no" -> "No")
- Append instruction to prompt
- Write the prompt file used for evaluation to ./ood_eval/benchmarks/cladder/data/

Run from the repository root:
  python3 -m ood_eval.benchmarks.cladder.data_processing.process_cladder --input <path>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List


# Prompt template instruction
INSTRUCTION = (
    "\n\n"
    "Think step-by-step to solve this causal reasoning question. "
    "Your final answer should only include a single Yes/No formatted inside "
    "\\boxed{}, as below: \\boxed{Yes|No}."
)


def read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    """Write records to a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def capitalize_label(label: str) -> str:
    """
    Capitalize the label (yes -> Yes, no -> No).

    Args:
        label: Original label string ("yes" or "no")

    Returns:
        Capitalized label ("Yes" or "No")

    Raises:
        ValueError: If label is not "yes" or "no"
    """
    label_lower = label.lower()
    if label_lower == "yes":
        return "Yes"
    elif label_lower == "no":
        return "No"
    else:
        raise ValueError(f"Invalid label: {label}. Expected 'yes' or 'no'.")


def format_prompt(prompt: str) -> str:
    """
    Format a prompt by appending the instruction.

    Args:
        prompt: Original prompt string

    Returns:
        Formatted prompt with instruction appended
    """
    return prompt + INSTRUCTION


def main():
    parser = argparse.ArgumentParser(
        description="Process CLadder-full_v1.5_default.jsonl: rename id->idx, reindex, format prompts."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the upstream CLadder JSONL file.",
    )
    args = parser.parse_args()

    input_path = args.input

    output_jsonl = "./ood_eval/benchmarks/cladder/data/CLadder-full_v1.5_default_processed.jsonl"

    # Validate input exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    # Read original data
    print(f"Reading input: {input_path}")
    records = read_jsonl(input_path)
    num_examples_total = len(records)
    print(f"num_examples_total: {num_examples_total}")

    # Process records
    processed_records: List[Dict] = []
    rung_counts: Dict[int, int] = defaultdict(int)
    query_type_counts: Dict[str, int] = defaultdict(int)
    answer_counts: Dict[str, int] = defaultdict(int)

    for idx, rec in enumerate(records):
        # Validate required fields
        if "id" not in rec:
            raise KeyError(f"Missing 'id' field in record at index {idx}")
        if "prompt" not in rec:
            raise KeyError(f"Missing 'prompt' field in record with id={rec.get('id')}")
        if "label" not in rec:
            raise KeyError(f"Missing 'label' field in record with id={rec.get('id')}")

        # Create processed record
        processed_rec = {}

        # Reindex idx from 0
        processed_rec["idx"] = idx

        # Format prompt -> question
        processed_rec["question"] = format_prompt(rec["prompt"])

        # Capitalize label -> answer
        answer = capitalize_label(rec["label"])
        processed_rec["answer"] = answer
        answer_counts[answer] += 1

        # Copy remaining fields (excluding id, prompt, label which are already processed)
        for key in ["reasoning", "rung", "query_type", "graph_id", "story_id", "question_property", "formal_form"]:
            if key in rec:
                processed_rec[key] = rec[key]

        # Track statistics
        if "rung" in rec:
            rung_counts[rec["rung"]] += 1
        if "query_type" in rec:
            query_type_counts[rec["query_type"]] += 1

        processed_records.append(processed_rec)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    write_jsonl(output_jsonl, processed_records)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"num_examples_total: {num_examples_total}")

    print("\nAnswer distribution:")
    for answer in sorted(answer_counts.keys()):
        print(f"  {answer}: {answer_counts[answer]}")

    print("\nRung distribution:")
    for rung in sorted(rung_counts.keys()):
        print(f"  rung {rung}: {rung_counts[rung]}")

    print("\nQuery type distribution:")
    for qt in sorted(query_type_counts.keys()):
        print(f"  {qt}: {query_type_counts[qt]}")

    print(f"\nJSONL written to: {output_jsonl}")


if __name__ == "__main__":
    main()
