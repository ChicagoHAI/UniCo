#!/usr/bin/env python3
"""
Process exec_cf counterfactual data: add idx, rename fields, transform prompt template.

Actions:
- Read the upstream code-if_else-counterfactual/test.jsonl named by --input_path
- Add 0-based idx field
- Rename prompt -> question (with template transformations)
- Rename ground_truth -> answer
- Remove ground_truth_str field
- Write the prompt file used for evaluation to ./ood_eval/benchmarks/exec_cf/data/

Run from the repository root:
  python3 -m ood_eval.benchmarks.exec_cf.data_processing.process_exec_cf --input_path <path>
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List


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


def transform_prompt(prompt: str) -> str:
    """
    Transform the prompt text according to the template requirements.

    Changes:
    - "Let's think step by step" -> "Think step by step"
    - "y = <answer>ans1, ans2, ans3</answer>, otherwise your answer will be marked as incorrect."
      -> "y = \\boxed{ans1, ans2, ans3}."
    """
    prompt = prompt.replace(
        "Let's think step by step to get the answer",
        "Think step by step to get the answer"
    )
    prompt = prompt.replace(
        "y = <answer>ans1, ans2, ans3</answer>, otherwise your answer will be marked as incorrect.",
        r"y = \boxed{ans1, ans2, ans3}."
    )
    prompt = prompt.replace(
        r"y = \boxed{ans1, ans2, ans3}, otherwise your answer will be marked as incorrect.",
        r"y = \boxed{ans1, ans2, ans3}."
    )
    return prompt


def process_record(idx: int, record: Dict) -> Dict:
    """
    Process a single record: add idx, rename/transform fields, remove unwanted fields.

    The transformed fields are written first, followed by every remaining
    source field except the original aliases and ``ground_truth_str``.
    """
    processed_record = {
        "idx": idx,
        "question": transform_prompt(record["prompt"]),
        "answer": record["ground_truth"],
    }
    excluded_fields = {
        "idx",
        "prompt",
        "ground_truth",
        "ground_truth_str",
        "causal_type",
        "dataset_name",
        "domain",
    }
    for field, value in record.items():
        if field not in excluded_fields:
            processed_record[field] = value
    return processed_record


def main():
    parser = argparse.ArgumentParser(
        description="Process exec_cf counterfactual data: add idx, rename fields, transform prompt."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the upstream JSONL file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ood_eval/benchmarks/exec_cf/data",
        help="Output directory for the processed JSONL.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="val_updated_fields_processed.jsonl",
        help="Output filename for the processed JSONL.",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = os.path.join(args.output_dir, args.output_filename)

    # Validate input exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    # Read input data
    print(f"Reading input: {input_path}")
    records = read_jsonl(input_path)
    num_examples_total = len(records)
    print(f"num_examples_total: {num_examples_total}")

    # Process records
    processed_records: List[Dict] = []
    num_single_answer = 0
    num_multi_answer = 0

    for idx, rec in enumerate(records):
        # Validate required fields
        required_fields = ["prompt", "ground_truth", "original_x", "original_y", "counterfactual_x", "meta_template"]
        for field in required_fields:
            if field not in rec:
                raise KeyError(f"Missing required field '{field}' at idx={idx}")

        # Process record
        processed_rec = process_record(idx, rec)
        processed_records.append(processed_rec)

        # Track answer cardinality
        if len(processed_rec["answer"]) == 1:
            num_single_answer += 1
        else:
            num_multi_answer += 1

    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(output_path, processed_records)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"num_examples_total: {num_examples_total}")
    print(f"num_single_answer: {num_single_answer}")
    print(f"num_multi_answer: {num_multi_answer}")
    print(f"\nJSONL written to: {output_path}")


if __name__ == "__main__":
    main()
