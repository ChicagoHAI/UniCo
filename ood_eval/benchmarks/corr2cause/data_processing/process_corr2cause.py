#!/usr/bin/env python3
"""
Process corr2cause-test.jsonl: add idx, format prompts, map labels.

Actions:
- Read the upstream corr2cause-test.jsonl named by --input
- Add 0-based idx field
- Transform input field to question using prompt template
- Map label to answer (0 -> "No", 1 -> "Yes")
- Keep num_variables and template fields
- Write the prompt file used for evaluation to ./ood_eval/benchmarks/corr2cause/data/

Run from the repository root:
  python3 -m ood_eval.benchmarks.corr2cause.data_processing.process_corr2cause --input <path>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Tuple


# Label to answer mapping
LABEL_TO_ANSWER = {0: "No", 1: "Yes"}

# Prompt template instruction
INSTRUCTION = (
    "\n\n"
    "Think step-by-step to solve this causal reasoning question. "
    "Your final answer should only include a single Yes/No formatted inside "
    "\\boxed{}, as below: \\boxed{Yes|No}"
)


def read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file and return a list of dicts."""
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
            json.dump(rec, ensure_ascii=False, fp=f)
            f.write("\n")


def parse_input_field(input_text: str) -> Tuple[str, str]:
    """
    Parse the input field to extract premise content and hypothesis content.

    The input field has format:
        "Premise: {premise_text}\nHypothesis: {hypothesis_text}"

    Returns:
        Tuple of (premise_content, hypothesis_content)

    Raises:
        ValueError: If the input format is invalid
    """
    # Match "Premise: ... \nHypothesis: ..."
    match = re.match(
        r"^Premise:\s*(.+?)\nHypothesis:\s*(.+)$",
        input_text,
        re.DOTALL
    )
    if not match:
        raise ValueError(f"Invalid input format: {input_text[:100]}...")

    premise_content = match.group(1).strip()
    hypothesis_content = match.group(2).strip()

    return premise_content, hypothesis_content


def format_prompt(premise_content: str, hypothesis_content: str) -> str:
    """
    Format a record into the evaluation prompt using the template.

    Template:
        Question: {premise_content}
        Can we deduce the following: {hypothesis_content}?

        {instruction}

    Returns:
        Formatted prompt string
    """
    prompt = (
        f"Question: {premise_content}\n"
        f"Can we deduce the following: {hypothesis_content}?"
        f"{INSTRUCTION}"
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Process corr2cause-test.jsonl: add idx, format prompts, map labels."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the upstream corr2cause-test JSONL file.",
    )
    args = parser.parse_args()

    input_path = args.input

    output_jsonl = "./ood_eval/benchmarks/corr2cause/data/corr2cause-test-processed.jsonl"

    # Validate input exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    # Read original data
    print(f"Reading input: {input_path}")
    records = read_jsonl(input_path)
    num_examples_total = len(records)
    print(f"num_examples_total: {num_examples_total}")

    # Process records
    output_records: List[Dict] = []
    num_label_yes = 0
    num_label_no = 0

    for idx, rec in enumerate(records):
        # Validate required fields
        if "input" not in rec:
            raise KeyError(f"Missing 'input' field in record at index {idx}")
        if "label" not in rec:
            raise KeyError(f"Missing 'label' field in record at index {idx}")
        if "num_variables" not in rec:
            raise KeyError(f"Missing 'num_variables' field in record at index {idx}")
        if "template" not in rec:
            raise KeyError(f"Missing 'template' field in record at index {idx}")

        # Parse input field
        premise_content, hypothesis_content = parse_input_field(rec["input"])

        # Format prompt
        question = format_prompt(premise_content, hypothesis_content)

        # Map label to answer
        label = rec["label"]
        if label not in LABEL_TO_ANSWER:
            raise ValueError(f"Invalid label value: {label}. Expected 0 or 1.")
        answer = LABEL_TO_ANSWER[label]

        # Track label distribution
        if label == 1:
            num_label_yes += 1
        else:
            num_label_no += 1

        # Create output record
        output_rec = {
            "idx": idx,
            "question": question,
            "answer": answer,
            "num_variables": rec["num_variables"],
            "template": rec["template"],
        }
        output_records.append(output_rec)

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    write_jsonl(output_jsonl, output_records)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"num_examples_total: {num_examples_total}")
    print(f"num_label_yes: {num_label_yes}")
    print(f"num_label_no: {num_label_no}")
    print(f"\nJSONL written to: {output_jsonl}")


if __name__ == "__main__":
    main()
