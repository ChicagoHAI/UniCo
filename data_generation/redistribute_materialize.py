"""
redistribute_materialize.py

Materializes redistributed datasets for symbolic and code domains under v1.4/.

Reads manifest files (from redistribute_manifest.py) to determine which
triplet_idx values belong to each domain, then filters existing v1.1 (symbolic)
and v1.2 (code) per-nodes files accordingly. Produces new per-nodes files and
cross-nodes integrated files with global re-indexing.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Set, Tuple

from data_generation.integration_pipeline import (
    DISTRIBUTION,
    compute_stats,
    load_jsonl,
    write_csv,
    write_jsonl,
)
from data_generation.redistribute_manifest import _query_type_dir


# ============================================================
# File Discovery
# ============================================================


def find_per_nodes_file(
    data_root: str,
    version: str,
    query_type: str,
    split: str,
    num_nodes: int,
) -> str:
    """Find the per-nodes integrated JSONL file."""
    qt_dir = _query_type_dir(query_type)
    node_dir = os.path.join(data_root, version, qt_dir, f"nodes{num_nodes}")

    if version == "v1.1":
        pattern = os.path.join(node_dir, f"*-{query_type}-{split}-nodes{num_nodes}.jsonl")
    elif version == "v1.2":
        pattern = os.path.join(node_dir, f"*-{query_type}-code-{split}-nodes{num_nodes}.jsonl")
    else:
        raise ValueError(f"Unsupported version: {version}")

    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No per-nodes file found for {version}/{query_type}/{split}/nodes{num_nodes} "
            f"(pattern: {pattern})"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple per-nodes files found: {matches}"
        )
    return matches[0]


# ============================================================
# Manifest Loading
# ============================================================

def load_manifest_assignments(
    manifest_path: str,
    domain: str,
    split: str,
) -> Dict[int, Set[int]]:
    """Load manifest and return {num_nodes: set_of_triplet_idx} for a domain.

    For training: selects entries with domain_assignment == domain.
    For eval: selects entries with selected == True (shared across all domains).
    """
    entries = load_jsonl(manifest_path)
    result: Dict[int, Set[int]] = {}

    for entry in entries:
        if split == "train":
            if entry["domain_assignment"] != domain:
                continue
        else:
            if not entry["selected"]:
                continue

        num_nodes = entry["num_nodes"]
        if num_nodes not in result:
            result[num_nodes] = set()
        result[num_nodes].add(entry["triplet_idx"])

    return result


# ============================================================
# Materialization
# ============================================================

def materialize_per_nodes(
    source_path: str,
    triplet_idx_set: Set[int],
    output_dir: str,
    filename_base: str,
) -> Tuple[str, List[Dict]]:
    """Filter source per-nodes file by triplet_idx set, write outputs.

    Returns (output_jsonl_path, filtered_records).
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

    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, f"{filename_base}.jsonl")
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    stats_path = os.path.join(output_dir, f"{filename_base}.json")

    write_jsonl(jsonl_path, filtered)
    write_csv(csv_path, filtered)
    stats = compute_stats(filtered, [source_path])
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return jsonl_path, filtered


def materialize_cross_nodes(
    per_nodes_records: List[Tuple[int, List[Dict]]],
    output_dir: str,
    filename_base: str,
    per_nodes_paths: List[str],
) -> str:
    """Concatenate per-nodes records in ascending num_nodes order, re-index from 0.

    per_nodes_records: list of (num_nodes, records) tuples, already sorted.
    Returns output JSONL path.
    """
    all_records = []
    for _num_nodes, records in per_nodes_records:
        all_records.extend(records)

    # Re-index globally from 0
    for new_idx, record in enumerate(all_records):
        record["idx"] = new_idx

    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, f"{filename_base}.jsonl")
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    stats_path = os.path.join(output_dir, f"{filename_base}.json")

    write_jsonl(jsonl_path, all_records)
    write_csv(csv_path, all_records)
    stats = compute_stats(
        all_records, per_nodes_paths, include_per_node_breakdown=True,
    )
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Cross-nodes: {jsonl_path} ({len(all_records)} records)")
    return jsonl_path


# ============================================================
# Domain Filename Helpers
# ============================================================

def _domain_filename_infix(domain: str) -> str:
    """Return filename infix for domain: '' for symbolic, '-code' for code."""
    if domain == "code":
        return "-code"
    return ""


def _source_version(domain: str) -> str:
    """Return source dataset version for domain."""
    if domain == "symbolic":
        return "v1.1"
    if domain == "code":
        return "v1.2"
    raise ValueError(f"Unsupported materialization domain: {domain}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Materialize redistributed datasets for symbolic and code domains"
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
        "--output_version",
        default="v1.4",
        help="Output version directory name",
    )
    parser.add_argument(
        "--run_timestamp",
        required=True,
        help="Timestamp prefix for output filenames",
    )
    parser.add_argument(
        "--manifest_timestamp",
        required=True,
        help="Timestamp used in manifest filenames",
    )
    parser.add_argument(
        "--query_types",
        nargs="+",
        default=["ate", "cate", "jate"],
        help="Query types to process",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["symbolic", "code"],
        help="Domains to materialize (default: symbolic code)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "eval"],
        help="Splits to materialize (default: train eval)",
    )
    args = parser.parse_args()

    output_base = os.path.join(args.data_root, args.output_version)

    for query_type in args.query_types:
        qt_dir = _query_type_dir(query_type)

        for split in args.splits:
            manifest_filename = (
                f"{args.manifest_timestamp}-{query_type}-{split}-manifest.jsonl"
            )
            manifest_path = os.path.join(args.manifest_dir, manifest_filename)

            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            for domain in args.domains:
                source_version = _source_version(domain)
                infix = _domain_filename_infix(domain)

                print(f"\n=== {query_type} / {split} / {domain} ({source_version}) ===")

                # Load manifest assignments
                assignments = load_manifest_assignments(
                    manifest_path, domain, split,
                )

                # Materialize per-nodes files
                per_nodes_data = []  # (num_nodes, records)
                per_nodes_paths = []
                nodes_dict = DISTRIBUTION[query_type][split]

                for num_nodes in sorted(nodes_dict.keys()):
                    if num_nodes not in assignments:
                        print(f"  WARN: No assignments for nodes={num_nodes}")
                        continue

                    triplet_idx_set = assignments[num_nodes]

                    # Find source per-nodes file
                    source_path = find_per_nodes_file(
                        args.data_root, source_version,
                        query_type, split, num_nodes,
                    )

                    # Output directory and filename
                    output_dir = os.path.join(
                        output_base, qt_dir, f"nodes{num_nodes}",
                    )
                    filename_base = (
                        f"{args.run_timestamp}-{query_type}{infix}-"
                        f"{split}-nodes{num_nodes}"
                    )

                    jsonl_path, filtered = materialize_per_nodes(
                        source_path, triplet_idx_set, output_dir, filename_base,
                    )
                    per_nodes_data.append((num_nodes, filtered))
                    per_nodes_paths.append(jsonl_path)

                    print(
                        f"  Per-nodes nodes={num_nodes}: "
                        f"{len(filtered)} records from {source_path}"
                    )

                # Cross-nodes integration
                cross_nodes_dir = os.path.join(output_base, qt_dir)
                cross_filename = (
                    f"{args.run_timestamp}-{query_type}{infix}-{split}"
                )
                materialize_cross_nodes(
                    per_nodes_data, cross_nodes_dir,
                    cross_filename, per_nodes_paths,
                )


if __name__ == "__main__":
    main()
