"""Process the BBEH causal understanding task.json into the prompt file used for evaluation.

Run from the repository root:

    python3 -m ood_eval.benchmarks.bbeh_causal_understanding.data_processing.process_task \\
        --input <path to the upstream task.json>
"""

import argparse
import json
from pathlib import Path

OUTPUT_PATH = Path("./ood_eval/benchmarks/bbeh_causal_understanding/data/task.jsonl")

# Prompt template suffixes (both keep "Question: " prefix)
COT_SUFFIX_MAJORITY = "\n\nThink step-by-step to answer this question. Include your final answer inside \\boxed{}, as below: \\boxed{Yes|No|Ambiguous}."
COT_SUFFIX_LOGICIAN = "\n\nThink step-by-step to answer this question. Include your final answer inside \\boxed{}, as below: \\boxed{Yes|No}."


def detect_pattern(input_text: str) -> str:
    """Detect whether example uses 'majority' or 'logician' pattern."""
    if "majority of people" in input_text:
        return "majority"
    elif "logician" in input_text:
        return "logician"
    else:
        raise ValueError(f"Unknown pattern in input: {input_text[:100]}...")


def apply_prompt_template(input_text: str, pattern: str) -> str:
    """Apply prompt template based on pattern type. Keep 'Question: ' prefix for both."""
    if pattern == "majority":
        return input_text + COT_SUFFIX_MAJORITY
    else:  # logician
        return input_text + COT_SUFFIX_LOGICIAN


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Upstream task.json: a JSON list of objects with input and target fields.",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples from {args.input}")

    jsonl_examples = []
    pattern_counts = {"majority": 0, "logician": 0}
    for idx, ex in enumerate(examples):
        input_text = ex["input"]
        pattern = detect_pattern(input_text)
        pattern_counts[pattern] += 1
        jsonl_examples.append(
            {
                "idx": idx,
                "question": apply_prompt_template(input_text, pattern),
                "answer": ex["target"],
            }
        )
    print(f"Pattern counts: {pattern_counts}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in jsonl_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
