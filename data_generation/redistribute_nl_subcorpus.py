"""
redistribute_nl_subcorpus.py

Extracts NL-domain triplet sub-corpora from v1.1 triplet corpus files.

For training: selects triplets assigned to NL domain in the manifest.
For eval: selects triplets marked as selected (shared across all domains).

Output files use the same filename format as v1.1 counterparts (with updated
examples count) for full compatibility with existing NL pipeline scripts.
Sub-corpus records preserve their original idx values (NOT re-indexed).
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Set, Tuple

from data_generation.integration_pipeline import (
    DISTRIBUTION,
    DP_VALUES,
    SEEDS,
    load_jsonl,
    write_jsonl,
)
from data_generation.redistribute_manifest import _query_type_dir


# ============================================================
# Manifest Loading
# ============================================================

def load_nl_triplet_indices(
    manifest_path: str,
    split: str,
) -> Dict[Tuple[int, int, str], Set[int]]:
    """Load NL-assigned triplet indices grouped by (num_nodes, dp, variant).

    Returns {(num_nodes, decimal_places, condition_variant): set_of_triplet_idx}.

    For training: selects entries with domain_assignment == "nl".
    For eval: selects entries with selected == True.
    """
    entries = load_jsonl(manifest_path)
    result: Dict[Tuple[int, int, str], Set[int]] = {}

    for entry in entries:
        if split == "train":
            if entry["domain_assignment"] != "nl":
                continue
        else:
            if not entry["selected"]:
                continue

        key = (
            entry["num_nodes"],
            entry["decimal_places"],
            entry["condition_variant"],
        )
        if key not in result:
            result[key] = set()
        result[key].add(entry["triplet_idx"])

    return result


# ============================================================
# Triplet Corpus File Discovery
# ============================================================

def find_triplet_corpus_file(
    data_root: str,
    query_type: str,
    num_nodes: int,
    dp: int,
    seed: int,
) -> str:
    """Find the v1.1 triplet corpus JSONL file for a specific combination."""
    qt_dir = _query_type_dir(query_type)
    triplets_dir = os.path.join(
        data_root, "v1.1", qt_dir, f"nodes{num_nodes}", "triplets", "jsonls",
    )

    if query_type == "ate":
        pattern = os.path.join(
            triplets_dir,
            f"*-nodes{num_nodes}-examples*-*-dp{dp}-seed{seed}-*.jsonl",
        )
    else:
        # CATE/JATE have query type prefix
        pattern = os.path.join(
            triplets_dir,
            f"*-{query_type}-nodes{num_nodes}-examples*-*-dp{dp}-seed{seed}-*.jsonl",
        )

    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No triplet corpus file found (pattern: {pattern})"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple triplet corpus files found: {matches}"
        )
    return matches[0]


def derive_subcorpus_filename(
    source_filename: str,
    new_examples_count: int,
) -> str:
    """Derive the sub-corpus filename by replacing the examples count.

    Keeps the exact same format as v1.1, only changing examples{M}.
    """
    return re.sub(
        r"examples\d+",
        f"examples{new_examples_count}",
        source_filename,
    )


# ============================================================
# Sub-Corpus Extraction
# ============================================================

def extract_subcorpus(
    source_path: str,
    triplet_idx_set: Set[int],
    output_path: str,
) -> int:
    """Extract records from source triplet file by index set.

    Preserves original idx values so that NL pipeline outputs use the same
    per-DP offset ranges expected by integration_pipeline.find_assembly_file()
    and integrate_per_nodes().

    Returns the number of records written.
    """
    records = load_jsonl(source_path)
    filtered = [r for r in records if r["idx"] in triplet_idx_set]

    if len(filtered) != len(triplet_idx_set):
        found_idxs = {r["idx"] for r in filtered}
        missing = triplet_idx_set - found_idxs
        raise RuntimeError(
            f"Missing {len(missing)} triplet indices in {source_path}: "
            f"first 10: {sorted(missing)[:10]}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(output_path, filtered)
    return len(filtered)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract NL-domain triplet sub-corpora from v1.1 triplet corpus"
    )
    parser.add_argument(
        "--data_root",
        default="./data_generation/generated_datasets",
        help="Root directory containing version subdirectories",
    )
    parser.add_argument(
        "--manifest_dir",
        default="./data_generation/generated_datasets/v1.4/manifests",
        help="Directory containing manifest JSONL files",
    )
    parser.add_argument(
        "--manifest_timestamp",
        required=True,
        help="Timestamp used in manifest filenames",
    )
    parser.add_argument(
        "--output_base",
        default="./data_generation/generated_datasets/v1.4/nl_subcorpora-v1.1",
        help="Output base directory for sub-corpus files",
    )
    # Note: manifest_timestamp is used to namespace sub-corpus output
    # directories, so different manifests (e.g., different --base_seed)
    # produce non-overlapping output paths.
    parser.add_argument(
        "--query_types",
        nargs="+",
        default=["ate", "cate", "jate"],
        help="Query types to process",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "eval"],
        help="Splits to process (default: train eval)",
    )
    args = parser.parse_args()

    for query_type in args.query_types:
        qt_dir = _query_type_dir(query_type)
        nodes_range = (
            sorted(DISTRIBUTION[query_type]["train"].keys())
        )

        for split in args.splits:
            manifest_filename = (
                f"{args.manifest_timestamp}-{query_type}-{split}-manifest.jsonl"
            )
            manifest_path = os.path.join(args.manifest_dir, manifest_filename)

            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            print(f"\n=== {query_type} / {split} ===")

            # Load NL assignments grouped by (num_nodes, dp, variant)
            nl_indices = load_nl_triplet_indices(manifest_path, split)

            seed = SEEDS[split]

            for num_nodes in sorted(nodes_range):
                # Compute total NL count for this num_nodes (for summary only)
                total_nl = sum(
                    len(idx_set)
                    for (n, dp, variant), idx_set in nl_indices.items()
                    if n == num_nodes
                )
                if total_nl == 0:
                    continue

                print(f"  nodes={num_nodes}: {total_nl} NL triplets")

                for dp in DP_VALUES:
                    # Collect all NL indices for this (nodes, dp) across both variants
                    dp_indices = set()
                    for variant in ["full", "surgery_ancestor_prune"]:
                        key = (num_nodes, dp, variant)
                        if key in nl_indices:
                            dp_indices |= nl_indices[key]

                    if not dp_indices:
                        continue

                    # Find source triplet corpus file
                    source_path = find_triplet_corpus_file(
                        args.data_root, query_type, num_nodes, dp, seed,
                    )

                    # Use per-file record count for examples{M} (matches v1.1
                    # convention where examples{M} equals actual line count)
                    per_file_count = len(dp_indices)
                    source_filename = os.path.basename(source_path)
                    output_filename = derive_subcorpus_filename(
                        source_filename, per_file_count,
                    )
                    output_dir = os.path.join(
                        args.output_base, args.manifest_timestamp,
                        qt_dir,
                        f"nodes{num_nodes}", "triplets", "jsonls",
                    )
                    output_path = os.path.join(output_dir, output_filename)

                    num_written = extract_subcorpus(
                        source_path, dp_indices, output_path,
                    )

                    # Validate filename matches actual content
                    if num_written != per_file_count:
                        raise RuntimeError(
                            f"Filename/content mismatch: expected {per_file_count}, "
                            f"wrote {num_written} to {output_path}"
                        )
                    print(f"    dp={dp}: {num_written} records -> {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
