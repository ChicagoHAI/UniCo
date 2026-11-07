"""Aggregate filtered reference datasets into a single JSONL and create subsets.

Produces two outputs:
1. A 6K-example aggregated JSONL from the three min100_max500 files:
   - All narrasum entries
   - Wikipedia: uniformly sampled so that narrasum + wikipedia = 3K
   - BBC news: remaining 3K, uniformly sampled
   Sorted by length ascending and re-indexed from 0.
   Each entry has: idx, source, text_content, length.

2. Per-dataset 10-example subsets (suffix -subset10) for human review,
   uniformly sampled from each min100_max500 file.

Usage (from the repository root):
    python3 -m data_generation.data_processing.aggregate_reference_datasets
"""

import json
import os
import random

REFERENCE_DIR = os.path.join("data_generation", "reference_datasets")

# (subdirectory, filtered filename, source name)
DATASET_FILES = [
    ("bbc_news_alltime", "bbc_news_alltime-min100_max500.jsonl", "bbc_news_alltime"),
    ("narrasum", "sampled_data-min100_max500.jsonl", "narrasum"),
    ("wikipedia-paragraphs", "wikipedia-paragraphs-min100_max500.jsonl", "wikipedia-paragraphs"),
]

TOTAL_SIZE = 6000
NARRASUM_PLUS_WIKI_SIZE = 3000
SUBSET_SIZE = 10
SEED = 42


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main():
    random.seed(SEED)

    # --- Load all filtered datasets ---
    datasets = {}
    for subdir, filename, source in DATASET_FILES:
        path = os.path.join(REFERENCE_DIR, subdir, filename)
        datasets[source] = read_jsonl(path)
        print(f"Loaded {source}: {len(datasets[source])} entries")

    # --- Aggregate 6K examples ---
    num_narrasum = len(datasets["narrasum"])
    num_wiki = NARRASUM_PLUS_WIKI_SIZE - num_narrasum
    num_bbc = TOTAL_SIZE - NARRASUM_PLUS_WIKI_SIZE

    print(f"\nSampling: narrasum={num_narrasum}, wikipedia={num_wiki}, bbc_news={num_bbc}")

    narrasum_samples = datasets["narrasum"]
    wiki_samples = random.sample(datasets["wikipedia-paragraphs"], num_wiki)
    bbc_samples = random.sample(datasets["bbc_news_alltime"], num_bbc)

    aggregated = []
    for source_name, samples in [
        ("narrasum", narrasum_samples),
        ("wikipedia-paragraphs", wiki_samples),
        ("bbc_news_alltime", bbc_samples),
    ]:
        for entry in samples:
            aggregated.append({
                "source": source_name,
                "text_content": entry["text_content"],
                "length": entry["length"],
            })

    aggregated.sort(key=lambda e: e["length"])
    for i, entry in enumerate(aggregated):
        entry["idx"] = i

    output_path = os.path.join(REFERENCE_DIR, "reference_datasets_aggregated_6k.jsonl")
    with open(output_path, "w") as fout:
        for entry in aggregated:
            fout.write(json.dumps(entry) + "\n")
    print(f"\n[OK] Aggregated {len(aggregated)} entries -> {output_path}")

    # --- Per-dataset 10-example subsets ---
    for subdir, filename, source in DATASET_FILES:
        entries = datasets[source]
        subset = random.sample(entries, min(SUBSET_SIZE, len(entries)))

        stem, ext = os.path.splitext(filename)
        subset_path = os.path.join(REFERENCE_DIR, subdir, f"{stem}-subset10{ext}")
        with open(subset_path, "w") as fout:
            for entry in subset:
                fout.write(json.dumps(entry) + "\n")

        print(f"[OK] {source}: {len(subset)} subset entries -> {os.path.basename(subset_path)}")


if __name__ == "__main__":
    main()
