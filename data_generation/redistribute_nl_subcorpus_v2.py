"""
redistribute_nl_subcorpus_v2.py

Extracts NL-domain triplet sub-corpora from v2.x triplet corpus files.

For training: selects triplets assigned to NL domain in the manifest.
For eval: selects triplets marked as selected (shared across all domains).

For shared-corpus query types (pn, ps, nie, nde), triplets are sourced from
the shared triplet_dir (pn_ps/ or nie_nde/) rather than the query type's own
directory.

Sub-corpus records preserve their original idx values (NOT re-indexed).
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, Optional, Set, Tuple

from data_generation.integration_pipeline import load_jsonl, write_jsonl
from data_generation.redistribute_manifest_v2 import (
    QUERY_CONFIG,
    find_v2_cross_nodes_file,
)

SEEDS = {"train": 0, "eval": 1000000}
DP_VALUES = [1, 2, 3, 4]


# ============================================================
# Cross-Nodes to Per-Nodes Index Mapping
# ============================================================

def build_cross_to_per_nodes_map(
    cross_nodes_path: str,
) -> Dict[int, int]:
    """Build cross_nodes_idx -> per_nodes_idx mapping.

    Cross-nodes files concatenate per-nodes records in ascending num_nodes
    order with global re-indexing. Within each num_nodes group, records
    maintain their original per-nodes positions (0-based within group).
    Per-nodes idx == triplet corpus idx.
    """
    records = load_jsonl(cross_nodes_path)
    mapping = {}
    position_counter: Dict[int, int] = {}

    for record in records:
        n = record["metadata"]["num_nodes"]
        if n not in position_counter:
            position_counter[n] = 0
        mapping[record["idx"]] = position_counter[n]
        position_counter[n] += 1

    return mapping


# ============================================================
# Manifest Loading
# ============================================================

def load_nl_triplet_indices(
    manifest_path: str,
    split: str,
    cross_to_per_nodes: Dict[int, int],
) -> Dict[Tuple[int, int, str], Set[int]]:
    """Load NL-assigned triplet indices grouped by (num_nodes, dp, variant).

    Converts cross-nodes idx to per-nodes idx (= triplet corpus idx).

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

        per_nodes_idx = cross_to_per_nodes.get(entry["triplet_idx"])
        if per_nodes_idx is None:
            raise RuntimeError(
                f"Cannot map cross-nodes idx {entry['triplet_idx']} to per-nodes idx"
            )

        key = (
            entry["num_nodes"],
            entry["decimal_places"],
            entry["condition_variant"],
        )
        if key not in result:
            result[key] = set()
        result[key].add(per_nodes_idx)

    return result


# ============================================================
# Triplet Corpus File Discovery (provenance-based)
# ============================================================

def resolve_triplet_basenames_from_stats(
    stats_path: str,
    seed: int,
) -> Dict[int, str]:
    """Parse per-node stats source_files to extract triplet basenames per dp.

    Assembly output filenames contain '-from-{triplet_basename}' provenance.
    Returns {dp: triplet_basename} mapping.
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)

    source_files = stats.get("source_files", [])
    dp_to_basename: Dict[int, str] = {}

    for sf in source_files:
        basename = os.path.basename(sf)
        from_match = re.search(r"-from-(.+)$", basename)
        if not from_match:
            continue
        triplet_basename = from_match.group(1)

        dp_match = re.search(r"-dp(\d+)-", triplet_basename)
        seed_match = re.search(r"-seed(\d+)", triplet_basename)
        if not dp_match or not seed_match:
            continue

        file_dp = int(dp_match.group(1))
        file_seed = int(seed_match.group(1))
        if file_seed != seed:
            continue

        if file_dp in dp_to_basename:
            if dp_to_basename[file_dp] != triplet_basename:
                raise RuntimeError(
                    f"Conflicting triplet basenames for dp={file_dp} in {stats_path}: "
                    f"{dp_to_basename[file_dp]} vs {triplet_basename}"
                )
        else:
            dp_to_basename[file_dp] = triplet_basename

    return dp_to_basename


def find_triplet_corpus_file(
    data_root: str,
    query_type: str,
    num_nodes: int,
    dp: int,
    triplet_basenames: Optional[Dict[int, str]],
    seed: int,
) -> str:
    """Find the triplet corpus JSONL file.

    Primary: uses provenance-resolved basename from per-node stats.
    Fallback: glob-based (only when per-node stats are unavailable).
    Uses triplet_dir from QUERY_CONFIG for shared corpora (pn→pn_ps, etc.).
    """
    config = QUERY_CONFIG[query_type]
    version = config["source_version"]
    triplet_dir = config["triplet_dir"]
    triplets_dir = os.path.join(
        data_root, version, triplet_dir,
        f"nodes{num_nodes}", "triplets", "jsonls",
    )

    if triplet_basenames is not None and dp in triplet_basenames:
        triplet_path = os.path.join(triplets_dir, triplet_basenames[dp])
        if not os.path.exists(triplet_path):
            raise FileNotFoundError(
                f"Resolved triplet file does not exist: {triplet_path}"
            )
        return triplet_path

    # Fallback: glob-based (for degraded cases without per-node stats)
    pattern = os.path.join(triplets_dir, f"*-dp{dp}-seed{seed}*.jsonl")
    matches = [
        m for m in glob.glob(pattern)
        if f"-nodes{num_nodes}-" in os.path.basename(m)
    ]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No triplet corpus file found: {pattern}"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple triplet corpus files found for "
            f"{query_type}/nodes{num_nodes}/dp={dp} and no stats provenance available: "
            f"{matches}"
        )
    return matches[0]


def derive_subcorpus_filename(
    source_filename: str,
    new_examples_count: int,
) -> str:
    """Derive sub-corpus filename by replacing the examples count."""
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
    """Extract records by index set, preserving original idx values.

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
        description="Extract NL-domain triplet sub-corpora from v2.x triplet corpus"
    )
    parser.add_argument(
        "--data_root",
        default="./data_generation/generated_datasets",
        help="Root directory containing version subdirectories",
    )
    parser.add_argument(
        "--manifest_dir",
        default="./data_generation/generated_datasets/v2.2/manifests",
        help="Directory containing manifest JSONL files",
    )
    parser.add_argument(
        "--manifest_timestamp",
        required=True,
        help="Timestamp used in manifest filenames",
    )
    parser.add_argument(
        "--output_base",
        default="./data_generation/generated_datasets/v2.2/nl_subcorpora",
        help="Output base directory for sub-corpus files",
    )
    parser.add_argument(
        "--query_types",
        nargs="+",
        default=list(QUERY_CONFIG.keys()),
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
        config = QUERY_CONFIG[query_type]
        min_nodes, max_nodes = config["nodes_range"]

        for split in args.splits:
            manifest_filename = (
                f"{args.manifest_timestamp}-{query_type}-{split}-manifest.jsonl"
            )
            manifest_path = os.path.join(args.manifest_dir, manifest_filename)

            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            print(f"\n=== {query_type} / {split} ===")

            # Build cross-nodes → per-nodes idx mapping from source file
            sym_path = find_v2_cross_nodes_file(
                args.data_root, query_type, split, "symbolic", config,
            )
            if sym_path is None:
                raise FileNotFoundError(
                    f"Symbolic source not found for {query_type}/{split}"
                )
            cross_to_per_nodes = build_cross_to_per_nodes_map(sym_path)

            nl_indices = load_nl_triplet_indices(
                manifest_path, split, cross_to_per_nodes,
            )

            # Derive source cross-nodes basename for per-node stats lookup
            source_cn_basename = os.path.splitext(
                os.path.basename(sym_path)
            )[0]

            seed = SEEDS[split]

            for num_nodes in range(min_nodes, max_nodes + 1):
                total_nl = sum(
                    len(idx_set)
                    for (n, dp, variant), idx_set in nl_indices.items()
                    if n == num_nodes
                )
                if total_nl == 0:
                    continue

                print(f"  nodes={num_nodes}: {total_nl} NL triplets")

                # Load triplet basenames from per-node stats provenance
                stats_path = os.path.join(
                    os.path.dirname(sym_path),
                    f"nodes{num_nodes}",
                    f"{source_cn_basename}-nodes{num_nodes}.json",
                )
                if os.path.exists(stats_path):
                    triplet_basenames = resolve_triplet_basenames_from_stats(
                        stats_path, seed,
                    )
                else:
                    print(
                        f"    WARN: Per-node stats not found ({stats_path}), "
                        f"falling back to glob resolution"
                    )
                    triplet_basenames = None

                for dp in DP_VALUES:
                    dp_indices = set()
                    for variant in config["variants"]:
                        key = (num_nodes, dp, variant)
                        if key in nl_indices:
                            dp_indices |= nl_indices[key]

                    if not dp_indices:
                        continue

                    source_path = find_triplet_corpus_file(
                        args.data_root, query_type, num_nodes, dp,
                        triplet_basenames, seed,
                    )

                    per_file_count = len(dp_indices)
                    source_filename = os.path.basename(source_path)
                    output_filename = derive_subcorpus_filename(
                        source_filename, per_file_count,
                    )
                    output_dir = os.path.join(
                        args.output_base, args.manifest_timestamp,
                        query_type,
                        f"nodes{num_nodes}", "triplets", "jsonls",
                    )
                    output_path = os.path.join(output_dir, output_filename)

                    num_written = extract_subcorpus(
                        source_path, dp_indices, output_path,
                    )

                    if num_written != per_file_count:
                        raise RuntimeError(
                            f"Filename/content mismatch: expected {per_file_count}, "
                            f"wrote {num_written} to {output_path}"
                        )
                    print(f"    dp={dp}: {num_written} records -> {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
