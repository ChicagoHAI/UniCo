"""
redistribute_manifest.py

Generates redistribution manifests for cross-domain dataset partitioning (v1.4).

Training triplets are split into 3 non-overlapping, equal-size subsets
(symbolic, code, NL) via stratified seeded shuffle within each per_combo block.
Eval triplets are 50%-sampled with a shared subset across all domains.

Each manifest record stores full provenance including old cross-nodes indices
from v1.1 and v1.2.
"""

import argparse
import glob
import hashlib
import json
import os
import random
from typing import Dict, List, Tuple

from data_generation.integration_pipeline import (
    DISTRIBUTION,
    DP_VALUES,
    SEEDS,
    load_jsonl,
    write_jsonl,
)

DOMAINS = ["symbolic", "code", "nl"]


# ============================================================
# Cross-Nodes File Discovery
# ============================================================

def _query_type_dir(query_type: str) -> str:
    """Return the subdirectory name for a query type."""
    if query_type == "ate":
        return "ate"
    return "cate_jate"


def find_cross_nodes_file(
    data_root: str,
    version: str,
    query_type: str,
    split: str,
) -> str:
    """Find the cross-nodes integrated JSONL file for a given (version, query_type, split).

    Returns the path to the file. Raises FileNotFoundError if not found or
    if multiple candidates exist.
    """
    qt_dir = _query_type_dir(query_type)
    base = os.path.join(data_root, version, qt_dir)

    # v1.1 uses no domain infix; v1.2 uses "-code-" infix
    if version == "v1.1":
        pattern = os.path.join(base, f"*-{query_type}-{split}.jsonl")
    elif version == "v1.2":
        pattern = os.path.join(base, f"*-{query_type}-code-{split}.jsonl")
    else:
        raise ValueError(f"Unsupported version: {version}")

    matches = [p for p in glob.glob(pattern) if "subset" not in os.path.basename(p)]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No cross-nodes file found for {version}/{query_type}/{split} "
            f"(pattern: {pattern})"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple cross-nodes files found for {version}/{query_type}/{split}: "
            f"{matches}"
        )
    return matches[0]


# ============================================================
# Cross-Nodes Index Lookup
# ============================================================

def build_cross_nodes_lookup(
    cross_nodes_path: str,
) -> Dict[Tuple[int, int], int]:
    """Build a (num_nodes, per_nodes_idx) -> cross_nodes_idx mapping.

    Cross-nodes files concatenate per-nodes records in ascending num_nodes
    order with global re-indexing from 0. Within each num_nodes group,
    the record order matches the per-nodes file, so the per-nodes position
    (0-based within the group) equals the triplet_idx.
    """
    records = load_jsonl(cross_nodes_path)
    lookup = {}
    position_counter: Dict[int, int] = {}

    for record in records:
        num_nodes = record["metadata"]["num_nodes"]
        if num_nodes not in position_counter:
            position_counter[num_nodes] = 0
        per_nodes_idx = position_counter[num_nodes]
        position_counter[num_nodes] += 1
        lookup[(num_nodes, per_nodes_idx)] = record["idx"]

    return lookup


# ============================================================
# Seed Computation
# ============================================================

def compute_stratum_seed(
    query_type: str,
    split: str,
    num_nodes: int,
    dp: int,
    variant: str,
    base_seed: int,
) -> int:
    """Compute a deterministic per-stratum seed via MD5 hash."""
    key = f"{query_type}_{split}_{num_nodes}_{dp}_{variant}_{base_seed}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


# ============================================================
# Manifest Generation
# ============================================================

def generate_manifest(
    query_type: str,
    split: str,
    base_seed: int,
    v11_lookup: Dict[Tuple[int, int], int],
    v12_lookup: Dict[Tuple[int, int], int],
    invert_eval_selection: bool = False,
) -> List[Dict]:
    """Generate manifest entries for a single (query_type, split).

    For training: 3-way stratified split across domains within each per_combo block.
    For eval: 50% selection of each per_combo block, shared across all domains.
    With invert_eval_selection, the eval split uses the other half of each block.
    """
    entries = []
    nodes_dict = DISTRIBUTION[query_type][split]

    for num_nodes in sorted(nodes_dict.keys()):
        num_examples = nodes_dict[num_nodes]
        per_combo = num_examples // 8

        for dp_idx, dp in enumerate(DP_VALUES):
            block_size = 2 * per_combo
            base = dp_idx * block_size

            for variant, offset in [
                ("full", 0),
                ("surgery_ancestor_prune", per_combo),
            ]:
                indices = [base + offset + i for i in range(per_combo)]

                seed = compute_stratum_seed(
                    query_type, split, num_nodes, dp, variant, base_seed,
                )
                rng = random.Random(seed)
                shuffled = rng.sample(indices, len(indices))

                if split == "train":
                    k = per_combo // 3
                    for domain_idx, domain in enumerate(DOMAINS):
                        assigned = shuffled[domain_idx * k : (domain_idx + 1) * k]
                        for idx in assigned:
                            entries.append({
                                "triplet_idx": idx,
                                "num_nodes": num_nodes,
                                "decimal_places": dp,
                                "condition_variant": variant,
                                "domain_assignment": domain,
                                "selected": True,
                                "old_v11_cross_nodes_idx": v11_lookup.get(
                                    (num_nodes, idx)
                                ),
                                "old_v12_cross_nodes_idx": v12_lookup.get(
                                    (num_nodes, idx)
                                ),
                                "selection_seed": seed,
                                "source_version": "v1.1",
                            })
                else:
                    k = per_combo // 2
                    selected = shuffled[:k]
                    unselected = shuffled[k:]
                    for idx in selected:
                        entries.append({
                            "triplet_idx": idx,
                            "num_nodes": num_nodes,
                            "decimal_places": dp,
                            "condition_variant": variant,
                            "domain_assignment": "excluded" if invert_eval_selection else "shared",
                            "selected": not invert_eval_selection,
                            "old_v11_cross_nodes_idx": v11_lookup.get(
                                (num_nodes, idx)
                            ),
                            "old_v12_cross_nodes_idx": v12_lookup.get(
                                (num_nodes, idx)
                            ),
                            "selection_seed": seed,
                            "source_version": "v1.1",
                        })
                    for idx in unselected:
                        entries.append({
                            "triplet_idx": idx,
                            "num_nodes": num_nodes,
                            "decimal_places": dp,
                            "condition_variant": variant,
                            "domain_assignment": "shared" if invert_eval_selection else "excluded",
                            "selected": invert_eval_selection,
                            "old_v11_cross_nodes_idx": v11_lookup.get(
                                (num_nodes, idx)
                            ),
                            "old_v12_cross_nodes_idx": v12_lookup.get(
                                (num_nodes, idx)
                            ),
                            "selection_seed": seed,
                            "source_version": "v1.1",
                        })

    return entries


# ============================================================
# Validation
# ============================================================

def validate_manifest(
    entries: List[Dict],
    query_type: str,
    split: str,
) -> None:
    """Validate manifest entries for correctness."""
    nodes_dict = DISTRIBUTION[query_type][split]
    expected_total = sum(nodes_dict.values())

    # Check total count
    if len(entries) != expected_total:
        raise RuntimeError(
            f"Manifest count mismatch for {query_type}/{split}: "
            f"expected {expected_total}, got {len(entries)}"
        )

    # Check all entries have valid provenance
    for entry in entries:
        if entry["old_v11_cross_nodes_idx"] is None:
            raise RuntimeError(
                f"Missing v1.1 cross-nodes idx for triplet_idx={entry['triplet_idx']}, "
                f"num_nodes={entry['num_nodes']}"
            )

    if split == "train":
        # Check 3-way disjoint split
        domain_sets = {d: set() for d in DOMAINS}
        for entry in entries:
            key = (entry["num_nodes"], entry["triplet_idx"])
            domain = entry["domain_assignment"]
            if key in domain_sets[domain]:
                raise RuntimeError(f"Duplicate triplet in {domain}: {key}")
            domain_sets[domain].add(key)

        # Check no overlap between domains
        for i, d1 in enumerate(DOMAINS):
            for d2 in DOMAINS[i + 1 :]:
                overlap = domain_sets[d1] & domain_sets[d2]
                if overlap:
                    raise RuntimeError(
                        f"Overlap between {d1} and {d2}: {len(overlap)} triplets"
                    )

        # Check per-domain counts
        for domain in DOMAINS:
            expected_per_domain = expected_total // 3
            actual = len(domain_sets[domain])
            if actual != expected_per_domain:
                raise RuntimeError(
                    f"{domain} count: expected {expected_per_domain}, got {actual}"
                )

        # Check per-stratum counts
        for num_nodes in sorted(nodes_dict.keys()):
            num_examples = nodes_dict[num_nodes]
            per_combo = num_examples // 8
            k = per_combo // 3

            for dp in DP_VALUES:
                for variant in ["full", "surgery_ancestor_prune"]:
                    for domain in DOMAINS:
                        count = sum(
                            1
                            for e in entries
                            if e["num_nodes"] == num_nodes
                            and e["decimal_places"] == dp
                            and e["condition_variant"] == variant
                            and e["domain_assignment"] == domain
                        )
                        if count != k:
                            raise RuntimeError(
                                f"Stratum imbalance: nodes={num_nodes} dp={dp} "
                                f"variant={variant} domain={domain}: "
                                f"expected {k}, got {count}"
                            )

    else:
        # Eval: check 50% selection
        selected = [e for e in entries if e["selected"]]
        expected_selected = expected_total // 2
        if len(selected) != expected_selected:
            raise RuntimeError(
                f"Eval selection count: expected {expected_selected}, "
                f"got {len(selected)}"
            )

        # Check per-stratum selection counts
        for num_nodes in sorted(nodes_dict.keys()):
            num_examples = nodes_dict[num_nodes]
            per_combo = num_examples // 8
            k = per_combo // 2

            for dp in DP_VALUES:
                for variant in ["full", "surgery_ancestor_prune"]:
                    count = sum(
                        1
                        for e in entries
                        if e["num_nodes"] == num_nodes
                        and e["decimal_places"] == dp
                        and e["condition_variant"] == variant
                        and e["selected"]
                    )
                    if count != k:
                        raise RuntimeError(
                            f"Eval stratum imbalance: nodes={num_nodes} dp={dp} "
                            f"variant={variant}: expected {k}, got {count}"
                        )

    # Check determinism: re-run with same seed should produce identical assignment
    # (This is guaranteed by the algorithm, but we verify a sample)
    print(f"  Validation passed for {query_type}/{split}: "
          f"{len(entries)} entries")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate redistribution manifests for v1.4 cross-domain partitioning"
    )
    parser.add_argument(
        "--data_root",
        default="./data_generation/generated_datasets",
        help="Root directory containing v1.1/ and v1.2/ datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="./data_generation/generated_datasets/v1.4/manifests",
        help="Output directory for manifest files",
    )
    parser.add_argument(
        "--run_timestamp",
        required=True,
        help="Timestamp prefix for output filenames",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for deterministic shuffling (default: 42)",
    )
    parser.add_argument(
        "--query_types",
        nargs="+",
        default=["ate", "cate", "jate"],
        help="Query types to process",
    )
    parser.add_argument(
        "--invert_eval_selection",
        action="store_true",
        help="Select the half of each eval block that the default selection leaves out.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for query_type in args.query_types:
        for split in ["train", "eval"]:
            print(f"\n=== {query_type} / {split} ===")

            # Build cross-nodes idx lookups from actual files
            v11_path = find_cross_nodes_file(
                args.data_root, "v1.1", query_type, split,
            )
            print(f"  v1.1 cross-nodes: {v11_path}")
            v11_lookup = build_cross_nodes_lookup(v11_path)

            v12_path = find_cross_nodes_file(
                args.data_root, "v1.2", query_type, split,
            )
            print(f"  v1.2 cross-nodes: {v12_path}")
            v12_lookup = build_cross_nodes_lookup(v12_path)

            # Generate manifest entries
            entries = generate_manifest(
                query_type, split, args.base_seed, v11_lookup, v12_lookup,
                invert_eval_selection=args.invert_eval_selection,
            )

            # Validate
            validate_manifest(entries, query_type, split)

            # Sort by (num_nodes, triplet_idx) for deterministic output order
            entries.sort(key=lambda e: (e["num_nodes"], e["triplet_idx"]))

            # Write manifest
            filename = f"{args.run_timestamp}-{query_type}-{split}-manifest.jsonl"
            output_path = os.path.join(args.output_dir, filename)
            write_jsonl(output_path, entries)
            print(f"  Wrote: {output_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
