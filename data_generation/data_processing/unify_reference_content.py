"""Unify the 'real content' field across reference datasets.

Renames each dataset's content field to a common ``text_content`` field
in place, removing the original field to avoid duplication.

Source-field mapping:
    bbc_news_alltime     content  -> text_content
    narrasum             summary  -> text_content
    wikipedia-paragraphs text     -> text_content

Usage (from the repository root):
    python3 -m data_generation.data_processing.unify_reference_content
"""

import json
import os

REFERENCE_DIR = os.path.join("data_generation", "reference_datasets")

# (subdirectory, input filename, original field to rename)
DATASET_CONFIGS = [
    ("bbc_news_alltime", "bbc_news_alltime.jsonl", "content"),
    ("narrasum", "sampled_data.jsonl", "summary"),
    ("wikipedia-paragraphs", "wikipedia-paragraphs.jsonl", "text"),
]

UNIFIED_FIELD = "text_content"


def unify_dataset_in_place(file_path: str, source_field: str) -> int:
    """Rename ``source_field`` to ``text_content`` in a JSONL file in place.

    Returns the number of entries processed.
    """
    records = []
    with open(file_path, "r") as fin:
        for line in fin:
            entry = json.loads(line)
            # Rename: copy value then delete old key
            entry[UNIFIED_FIELD] = entry.pop(source_field)
            # Clean up stale fields from any prior run
            entry.pop("source", None)
            records.append(entry)

    with open(file_path, "w") as fout:
        for record in records:
            fout.write(json.dumps(record) + "\n")

    return len(records)


def main():
    for subdir, filename, source_field in DATASET_CONFIGS:
        file_path = os.path.join(REFERENCE_DIR, subdir, filename)
        if not os.path.exists(file_path):
            print(f"[SKIP] {file_path} not found")
            continue

        num_entries = unify_dataset_in_place(file_path, source_field)
        print(f"[OK]   {subdir}: {num_entries} entries ('{source_field}' -> '{UNIFIED_FIELD}')")


if __name__ == "__main__":
    main()
