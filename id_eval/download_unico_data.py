#!/usr/bin/env python3
"""Download the UniCo prompt files from Hugging Face into the layout the ID evaluation reads.

Run from the repository root:

    python3 -m id_eval.download_unico_data --splits test

Each Hugging Face config ``{domain}-{query_type}`` becomes one JSONL file per split:

    ./data/unico/{query_type}/{query_type}[-code|-nl]-{eval|train}.jsonl

The ``test`` split is written as ``eval``. On the Hub, ``answer`` and ``metadata`` are stored as
JSON-encoded strings because some configs mix value types; this script decodes them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset

REPO_ID = "ChicagoHAI/UniCo"
# The revision whose evaluation splits match the paper's prompt files record for record.
DEFAULT_REVISION = "9c1391cfaed8cecf529c7d516698615907cf959e"
DEFAULT_OUTPUT_DIR = "./data/unico"
SPLITS = ("test", "train")
SPLIT_TO_FILE_SUFFIX = {"test": "eval", "train": "train"}
DOMAIN_TO_FILE_INFIX = {"symbolic": "", "code": "-code", "nl": "-nl"}
EXPECTED_NUM_RECORDS_BY_SPLIT = {"test": 13321, "train": 66603}

# These query types are rendered in all three representation forms.
MULTI_DOMAIN_QUERY_TYPES = (
    "ate",
    "cate",
    "jate",
    "att",
    "cf",
    "conditional_prob",
    "joint_prob",
    "marginal_prob",
    "nde",
    "nie",
    "observed_difference",
    "pn",
    "ps",
)
# These depend only on the graph, so they exist only in the symbolic form.
SYMBOLIC_ONLY_QUERY_TYPES = (
    "identification",
    "backdoor",
    "frontdoor",
    "independence_test",
    "markov_blanket",
)


def list_domain_query_pairs() -> List[Tuple[str, str]]:
    """List the 44 (domain, query_type) pairs published on the Hub."""
    pairs = [("symbolic", query_type) for query_type in SYMBOLIC_ONLY_QUERY_TYPES]
    for domain in DOMAIN_TO_FILE_INFIX:
        pairs.extend((domain, query_type) for query_type in MULTI_DOMAIN_QUERY_TYPES)
    return pairs


def output_path_for(output_dir: str, domain: str, query_type: str, split: str) -> Path:
    """Return where one config's split is written."""
    infix = DOMAIN_TO_FILE_INFIX[domain]
    suffix = SPLIT_TO_FILE_SUFFIX[split]
    return Path(output_dir) / query_type / f"{query_type}{infix}-{suffix}.jsonl"


def decode_record(row: Dict[str, object]) -> Dict[str, object]:
    """Turn one Hub row into a prompt-file record, restoring the JSON-encoded fields."""
    return {
        "idx": row["idx"],
        "question": row["question"],
        "answer": json.loads(row["answer"]),
        "metadata": json.loads(row["metadata"]),
    }


def download_config(
    *,
    domain: str,
    query_type: str,
    split: str,
    output_dir: str,
    revision: str,
) -> int:
    """Write one config's split as JSONL and return its number of records."""
    dataset = load_dataset(REPO_ID, f"{domain}-{query_type}", split=split, revision=revision)
    output_path = output_path_for(output_dir, domain, query_type, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(decode_record(row), ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} records to {output_path}", flush=True)
    return len(dataset)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["test"],
        help="Splits to download. Evaluation only needs test (default: test).",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Dataset revision (default: the revision used for the paper).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for split in args.splits:
        num_records = sum(
            download_config(
                domain=domain,
                query_type=query_type,
                split=split,
                output_dir=args.output_dir,
                revision=args.revision,
            )
            for domain, query_type in list_domain_query_pairs()
        )
        print(f"Split {split}: {num_records} records in total.", flush=True)
        expected = EXPECTED_NUM_RECORDS_BY_SPLIT[split]
        if args.revision == DEFAULT_REVISION and num_records != expected:
            raise ValueError(f"Expected {expected} {split} records, downloaded {num_records}.")


if __name__ == "__main__":
    main()
