"""Filter reference datasets by text_content token length.

For each reference dataset, writes a new file containing only entries
whose ``text_content`` field has between ``--min_tokens`` and
``--max_tokens`` tokens (inclusive) as measured by tiktoken's o200k_base
encoding (GPT-5.4). Output filenames have a
``-min{min}_max{max}`` suffix before the extension.

Usage (from the repository root):
    python3 -m data_generation.data_processing.filter_reference_by_length
    python3 -m data_generation.data_processing.filter_reference_by_length --min_tokens 50 --max_tokens 300
"""

import argparse
import json
import os

import tiktoken

REFERENCE_DIR = os.path.join("data_generation", "reference_datasets")

# (subdirectory, filename)
DATASET_FILES = [
    ("bbc_news_alltime", "bbc_news_alltime.jsonl"),
    ("narrasum", "sampled_data.jsonl"),
    ("wikipedia-paragraphs", "wikipedia-paragraphs.jsonl"),
]

CONTENT_FIELD = "text_content"


def main():
    parser = argparse.ArgumentParser(
        description="Filter reference datasets by text_content token length."
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=100,
        help="Minimum number of tokens required in text_content (default: 100).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens allowed in text_content (default: 500).",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding("o200k_base")

    for subdir, filename in DATASET_FILES:
        input_path = os.path.join(REFERENCE_DIR, subdir, filename)
        if not os.path.exists(input_path):
            print(f"[SKIP] {input_path} not found")
            continue

        stem, ext = os.path.splitext(filename)
        output_filename = f"{stem}-min{args.min_tokens}_max{args.max_tokens}{ext}"
        output_path = os.path.join(REFERENCE_DIR, subdir, output_filename)

        num_total = 0
        num_kept = 0
        with open(input_path, "r") as fin, open(output_path, "w") as fout:
            for line in fin:
                num_total += 1
                entry = json.loads(line)
                num_tokens = len(enc.encode(entry[CONTENT_FIELD]))
                if args.min_tokens <= num_tokens <= args.max_tokens:
                    num_kept += 1
                    entry["length"] = num_tokens
                    fout.write(json.dumps(entry) + "\n")

        print(
            f"[OK]   {subdir}: {num_kept}/{num_total} entries kept"
            f" (min={args.min_tokens}, max={args.max_tokens}) -> {output_filename}"
        )


if __name__ == "__main__":
    main()
