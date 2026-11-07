"""
redistribute_materialize_v2.py

Materializes redistributed datasets for symbolic and code domains under v2.2/.

Reads manifest files (from redistribute_manifest_v2.py) and source cross-nodes
files, filters by manifest assignments, then produces both cross-nodes files
(re-indexed from 0) and per-node disaggregated files. v2 source datasets only
have cross-nodes files (no per-node files), so materialization disaggregates
by metadata.num_nodes.
"""

import argparse
import collections
import csv
import json
import os
from typing import Dict, List, Set

from data_generation.integration_pipeline import load_jsonl, write_jsonl
from data_generation.redistribute_manifest_v2 import (
    QUERY_CONFIG,
    find_v2_cross_nodes_file,
)


# ============================================================
# Manifest Loading
# ============================================================

def load_manifest_assignments(
    manifest_path: str,
    domain: str,
    split: str,
) -> Set[int]:
    """Load manifest and return set of source cross-nodes indices for a domain.

    For training: selects entries with domain_assignment == domain,
    returns old_source_cross_nodes_idx (symbolic) or old_code_cross_nodes_idx (code).
    For eval: selects entries with selected == True.
    """
    entries = load_jsonl(manifest_path)
    idx_field = (
        "old_source_cross_nodes_idx" if domain == "symbolic"
        else "old_code_cross_nodes_idx"
    )
    result = set()

    for entry in entries:
        if split == "train":
            if entry["domain_assignment"] != domain:
                continue
        else:
            if not entry["selected"]:
                continue

        idx_val = entry[idx_field]
        if idx_val is not None:
            result.add(idx_val)

    return result


# ============================================================
# I/O Helpers
# ============================================================

def write_csv_v2(path: str, records: List[Dict]) -> None:
    """Write records to a CSV file, handling arbitrary metadata."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "question", "answer", "metadata"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "idx": r["idx"],
                "question": r["question"],
                "answer": r["answer"],
                "metadata": json.dumps(r["metadata"], ensure_ascii=False),
            })


def _aggregate_difficulty(records: List[Dict]) -> Dict:
    """Aggregate difficulty metrics from record metadata."""
    num = len(records)
    if num == 0:
        return {
            "avg_treatment_outcome_depth": 0,
            "avg_effective_prob_conds": 0,
            "naive_conditioning_rate": 0,
        }
    total_depth = sum(r["metadata"]["treatment_outcome_depth"] for r in records)
    total_eff = sum(r["metadata"]["effective_prob_conds"] for r in records)
    naive_count = 0
    for r in records:
        m = r["metadata"]
        if m.get("has_naive_conditioning") or m.get("has_naive_shortcut"):
            naive_count += 1
    return {
        "avg_treatment_outcome_depth": total_depth / num,
        "avg_effective_prob_conds": total_eff / num,
        "naive_conditioning_rate": naive_count / num,
    }


def compute_stats_v2(
    records: List[Dict],
    source_files: List[str],
    include_per_node_breakdown: bool = False,
) -> Dict:
    """Compute dataset-level statistics for v2 records.

    Conditionally includes difficulty metrics (abnormal rate, treatment-outcome
    depth, effective prob conds, naive conditioning rate) when the source
    records contain the relevant metadata fields.
    """
    num_examples = len(records)

    combo_counts = {}
    for r in records:
        variant = r["metadata"]["variant_name"]
        dp = r["metadata"].get("decimal_places", "unknown")
        key = f"{variant}_dp{dp}"
        combo_counts[key] = combo_counts.get(key, 0) + 1

    dag_edge_sets = set()
    for r in records:
        edges = r["metadata"].get("dag_edges", [])
        dag_edge_sets.add(tuple(tuple(e) for e in edges))

    sample_meta = records[0]["metadata"] if records else {}
    has_abnormal = "is_abnormal" in sample_meta
    has_verification = "verification_ok" in sample_meta
    has_difficulty = "treatment_outcome_depth" in sample_meta

    stats = {
        "num_examples": num_examples,
        "num_unique_dags": len(dag_edge_sets),
    }

    if has_abnormal:
        abnormal_count = sum(
            1 for r in records if r["metadata"]["is_abnormal"]
        )
        stats["abnormal_count"] = abnormal_count
        stats["abnormal_rate"] = abnormal_count / num_examples if num_examples else 0

    if has_verification:
        verification_false_count = sum(
            1 for r in records if r["metadata"]["verification_ok"] is False
        )
        stats["verification_false_count"] = verification_false_count
        stats["verification_false_rate"] = (
            verification_false_count / num_examples if num_examples else 0
        )

    if has_difficulty:
        global_diff = _aggregate_difficulty(records)
        if include_per_node_breakdown:
            by_nodes_list: Dict[int, List[Dict]] = collections.defaultdict(list)
            for r in records:
                by_nodes_list[r["metadata"]["num_nodes"]].append(r)
            sorted_nodes = sorted(by_nodes_list.keys())
            node_diffs = {
                n: _aggregate_difficulty(by_nodes_list[n]) for n in sorted_nodes
            }

            stats["avg_treatment_outcome_depth"] = global_diff["avg_treatment_outcome_depth"]
            for n in sorted_nodes:
                stats[f"nodes{n}_avg_treatment_outcome_depth"] = node_diffs[n]["avg_treatment_outcome_depth"]

            stats["avg_effective_prob_conds"] = global_diff["avg_effective_prob_conds"]
            for n in sorted_nodes:
                stats[f"nodes{n}_avg_effective_prob_conds"] = node_diffs[n]["avg_effective_prob_conds"]

            stats["naive_conditioning_rate"] = global_diff["naive_conditioning_rate"]
            for n in sorted_nodes:
                stats[f"nodes{n}_naive_conditioning_rate"] = node_diffs[n]["naive_conditioning_rate"]
        else:
            stats.update(global_diff)

    stats["per_combo_counts"] = combo_counts
    stats["source_files"] = source_files

    if include_per_node_breakdown:
        by_nodes_count: Dict[int, int] = collections.defaultdict(int)
        for r in records:
            by_nodes_count[r["metadata"]["num_nodes"]] += 1
        stats["per_node_counts"] = {
            f"nodes{n}": c for n, c in sorted(by_nodes_count.items())
        }

    return stats


# ============================================================
# Materialization
# ============================================================

def materialize_from_cross_nodes(
    source_path: str,
    idx_set: Set[int],
    output_base_dir: str,
    query_type: str,
    run_timestamp: str,
    split: str,
    domain: str,
) -> None:
    """Filter source cross-nodes file by idx set, produce cross-nodes and per-node files.

    Cross-nodes output is re-indexed from 0.
    Per-node files are disaggregated by metadata.num_nodes.
    """
    records = load_jsonl(source_path)
    filtered = [r for r in records if r["idx"] in idx_set]

    if len(filtered) != len(idx_set):
        found_idxs = {r["idx"] for r in filtered}
        missing = idx_set - found_idxs
        raise RuntimeError(
            f"Missing {len(missing)} indices in {source_path}: "
            f"first 10: {sorted(missing)[:10]}"
        )

    # Sort by original idx for deterministic ordering before re-indexing
    filtered.sort(key=lambda r: r["idx"])

    # Re-index globally from 0
    for new_idx, record in enumerate(filtered):
        record["idx"] = new_idx

    # Determine filename infix
    infix = "-code" if domain == "code" else ""

    # Write cross-nodes files
    qt_dir = os.path.join(output_base_dir, query_type)
    os.makedirs(qt_dir, exist_ok=True)

    cross_base = f"{run_timestamp}-{query_type}{infix}-{split}"
    cross_jsonl = os.path.join(qt_dir, f"{cross_base}.jsonl")
    cross_csv = os.path.join(qt_dir, f"{cross_base}.csv")
    cross_stats = os.path.join(qt_dir, f"{cross_base}.json")

    write_jsonl(cross_jsonl, filtered)
    write_csv_v2(cross_csv, filtered)

    per_nodes_paths = []

    # Disaggregate by num_nodes and write per-node files
    by_nodes: Dict[int, List[Dict]] = collections.defaultdict(list)
    for r in filtered:
        by_nodes[r["metadata"]["num_nodes"]].append(r)

    for num_nodes in sorted(by_nodes.keys()):
        node_records = by_nodes[num_nodes]
        node_dir = os.path.join(qt_dir, f"nodes{num_nodes}")
        os.makedirs(node_dir, exist_ok=True)

        node_base = f"{run_timestamp}-{query_type}{infix}-{split}-nodes{num_nodes}"
        node_jsonl = os.path.join(node_dir, f"{node_base}.jsonl")
        node_csv = os.path.join(node_dir, f"{node_base}.csv")
        node_stats = os.path.join(node_dir, f"{node_base}.json")

        write_jsonl(node_jsonl, node_records)
        write_csv_v2(node_csv, node_records)

        node_stats_data = compute_stats_v2(node_records, [node_jsonl])
        with open(node_stats, "w") as f:
            json.dump(node_stats_data, f, indent=2)

        per_nodes_paths.append(node_jsonl)
        print(
            f"    nodes={num_nodes}: {len(node_records)} records"
        )

    # Write cross-nodes stats
    cross_stats_data = compute_stats_v2(
        filtered, per_nodes_paths, include_per_node_breakdown=True,
    )
    with open(cross_stats, "w") as f:
        json.dump(cross_stats_data, f, indent=2)

    print(f"  Cross-nodes: {cross_jsonl} ({len(filtered)} records)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Materialize redistributed datasets for symbolic and code domains (v2.2)"
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
        "--output_version",
        default="v2.2",
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
        default=list(QUERY_CONFIG.keys()),
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
        config = QUERY_CONFIG[query_type]

        for split in args.splits:
            manifest_filename = (
                f"{args.manifest_timestamp}-{query_type}-{split}-manifest.jsonl"
            )
            manifest_path = os.path.join(args.manifest_dir, manifest_filename)

            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            for domain in args.domains:
                print(
                    f"\n=== {query_type} / {split} / {domain} "
                    f"({config['source_version']}) ==="
                )

                # Load manifest assignments
                idx_set = load_manifest_assignments(
                    manifest_path, domain, split,
                )

                if not idx_set:
                    print("  No assignments for this domain — skipping")
                    continue

                # Find source cross-nodes file
                source_path = find_v2_cross_nodes_file(
                    args.data_root, query_type, split, domain, config,
                )
                if source_path is None:
                    print(f"  Source file not found — skipping")
                    continue

                print(f"  Source: {source_path}")
                print(f"  Assigned indices: {len(idx_set)}")

                materialize_from_cross_nodes(
                    source_path, idx_set, output_base,
                    query_type, args.run_timestamp, split, domain,
                )


if __name__ == "__main__":
    main()
