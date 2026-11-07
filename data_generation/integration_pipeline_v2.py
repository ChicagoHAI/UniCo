"""v2.2 NL integration pipeline for association/counterfactual query types."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Set, Tuple

from data_generation.dataset_utils.nl_query_specs import QUERY_TYPES, get_query_spec
from data_generation.integration_pipeline import (
    DP_VALUES,
    SEEDS,
    load_jsonl,
    write_csv,
    write_jsonl,
    write_stats,
)


ManifestIndex = Dict[Tuple[int, int, str], Set[int]]


def load_manifest_indices(manifest_path: str, split: str) -> ManifestIndex:
    """Load selected per-node idxs grouped by (num_nodes, dp, variant)."""
    result: ManifestIndex = {}
    position_by_nodes: Dict[int, int] = {}
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    for entry in sorted(entries, key=lambda e: e["triplet_idx"]):
        num_nodes = int(entry["num_nodes"])
        per_nodes_idx = position_by_nodes.get(num_nodes, 0)
        position_by_nodes[num_nodes] = per_nodes_idx + 1

        if split == "train":
            if entry.get("domain_assignment") != "nl":
                continue
        elif not entry.get("selected", False):
            continue

        key = (
            num_nodes,
            int(entry["decimal_places"]),
            entry["condition_variant"],
        )
        result.setdefault(key, set()).add(per_nodes_idx)
    return result


def _node_dir_name(num_nodes: int, prompting_version: str) -> str:
    return f"nodes{num_nodes}-prompting_{prompting_version}"


def find_nl_output_file(
    base_dir: str,
    num_nodes: int,
    prompting_version: str,
    variant_name: str,
    query_type: str,
    dp: int,
    seed: int,
    variant_assignment: str,
    config_fingerprint: str | None,
) -> str:
    search_dir = os.path.join(
        base_dir,
        _node_dir_name(num_nodes, prompting_version),
        f"{variant_name}-nl",
        "jsonls",
    )
    cfg_glob = f"*-cfg{config_fingerprint}" if config_fingerprint else "*"
    pattern = (
        f"*-qt{query_type}-n{num_nodes}-dp{dp}-sd{seed}"
        f"-va_{variant_assignment}{cfg_glob}.jsonl"
    )
    matches = sorted(glob.glob(os.path.join(search_dir, pattern)))
    if not matches:
        raise FileNotFoundError(f"No NL output found: {search_dir}/{pattern}")
    return matches[-1]


def integrate_per_nodes(
    *,
    query_type: str,
    split: str,
    base_dir: str,
    num_nodes: int,
    prompting_version: str,
    run_timestamp: str,
    manifest_indices: ManifestIndex,
    seed: int,
    dp_values: List[int],
    variant_assignment: str,
    config_fingerprint: str | None,
) -> str:
    spec = get_query_spec(query_type)
    all_records: List[dict] = []
    source_files: List[str] = []
    expected_total = 0

    for dp in dp_values:
        for variant_name in spec.variants:
            expected_indices = manifest_indices.get((num_nodes, dp, variant_name), set())
            expected_total += len(expected_indices)
            if not expected_indices:
                continue

            path = find_nl_output_file(
                base_dir,
                num_nodes,
                prompting_version,
                variant_name,
                query_type,
                dp,
                seed,
                variant_assignment,
                config_fingerprint,
            )
            records = [
                r for r in load_jsonl(path)
                if r["idx"] in expected_indices
            ]
            records.sort(key=lambda r: r["idx"])
            source_files.append(path)

            if len(records) < len(expected_indices):
                print(
                    f"  WARN: {query_type} nodes={num_nodes} dp={dp} "
                    f"{variant_name} has {len(records)}/{len(expected_indices)} "
                    "manifest-selected records"
                )

            for record in records:
                record["metadata"]["decimal_places"] = dp
            all_records.extend(records)

    actual = len(all_records)
    if actual < expected_total:
        print(
            f"  WARN: Per-nodes NL record count for {query_type} nodes={num_nodes}: "
            f"{actual}/{expected_total} ({expected_total - actual} dropped or failed)"
        )

    output_dir = os.path.join(base_dir, _node_dir_name(num_nodes, prompting_version))
    filename = f"{run_timestamp}-{query_type}-nl-{split}-nodes{num_nodes}"
    jsonl_path = os.path.join(output_dir, f"{filename}.jsonl")
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    stats_path = os.path.join(output_dir, f"{filename}.json")

    write_jsonl(jsonl_path, all_records)
    write_csv(csv_path, all_records)
    write_stats(stats_path, all_records, source_files)

    print(f"  Per-nodes: {jsonl_path} ({actual} records)")
    return jsonl_path


def integrate_cross_nodes(
    *,
    query_type: str,
    split: str,
    base_dir: str,
    per_nodes_paths: List[str],
    run_timestamp: str,
) -> None:
    all_records: List[dict] = []
    for path in per_nodes_paths:
        all_records.extend(load_jsonl(path))

    for new_idx, record in enumerate(all_records):
        record["idx"] = new_idx

    filename = f"{run_timestamp}-{query_type}-nl-{split}"
    jsonl_path = os.path.join(base_dir, f"{filename}.jsonl")
    csv_path = os.path.join(base_dir, f"{filename}.csv")
    stats_path = os.path.join(base_dir, f"{filename}.json")

    write_jsonl(jsonl_path, all_records)
    write_csv(csv_path, all_records)
    write_stats(
        stats_path,
        all_records,
        per_nodes_paths,
        include_per_node_breakdown=True,
    )
    print(f"  Cross-nodes: {jsonl_path} ({len(all_records)} records)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate v2.2 NL outputs for association/counterfactual queries",
    )
    parser.add_argument("--query_type", type=str, required=True, choices=QUERY_TYPES)
    parser.add_argument("--split", type=str, required=True, choices=["train", "eval"])
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument("--manifest_timestamp", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["per_nodes", "cross_nodes", "both"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dp_values", type=int, nargs="+", default=None)
    parser.add_argument("--num_nodes_list", type=int, nargs="+", default=None)
    parser.add_argument("--prompting_version", type=str, default=None,
                        choices=["v2", "v3"])
    parser.add_argument("--prompting_version_list", type=str, nargs="+", default=None)
    parser.add_argument("--config_fingerprint", type=str, default=None)
    parser.add_argument("--config_fingerprint_list", type=str, nargs="+", default=None)
    parser.add_argument("--variant_assignment", type=str, default="manifest",
                        choices=["split", "both", "manifest"])
    args = parser.parse_args()

    if args.prompting_version and args.prompting_version_list:
        parser.error("--prompting_version and --prompting_version_list are mutually exclusive")
    if args.config_fingerprint and args.config_fingerprint_list:
        parser.error("--config_fingerprint and --config_fingerprint_list are mutually exclusive")

    manifest_path = os.path.join(
        args.manifest_dir,
        f"{args.manifest_timestamp}-{args.query_type}-{args.split}-manifest.jsonl",
    )
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest_indices = load_manifest_indices(manifest_path, args.split)

    manifest_nodes = sorted({key[0] for key in manifest_indices})
    if args.num_nodes_list is not None:
        nodes_list = args.num_nodes_list
    else:
        nodes_list = manifest_nodes

    num_nodes_count = len(nodes_list)
    if args.prompting_version_list is not None:
        if len(args.prompting_version_list) != num_nodes_count:
            raise ValueError("--prompting_version_list must match --num_nodes_list")
        prompting_versions = args.prompting_version_list
    elif args.prompting_version is not None:
        prompting_versions = [args.prompting_version] * num_nodes_count
    else:
        parser.error("--prompting_version or --prompting_version_list is required")

    if args.config_fingerprint_list is not None:
        if len(args.config_fingerprint_list) != num_nodes_count:
            raise ValueError("--config_fingerprint_list must match --num_nodes_list")
        config_fingerprints = [
            None if c in ("", "none") else c
            for c in args.config_fingerprint_list
        ]
    elif args.config_fingerprint is not None:
        config_fingerprints = [args.config_fingerprint] * num_nodes_count
    else:
        config_fingerprints = [None] * num_nodes_count

    seed = args.seed if args.seed is not None else SEEDS[args.split]
    dp_values = args.dp_values if args.dp_values is not None else DP_VALUES

    print(f"=== v2.2 NL Integration: {args.query_type} {args.split} ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  manifest_path: {manifest_path}")
    print(f"  resolved nodes: {nodes_list}")
    print(f"  resolved prompting_versions: {prompting_versions}")
    print(f"  resolved config_fingerprints: {config_fingerprints}")
    print(f"  resolved seed: {seed}")
    print(f"  resolved dp_values: {dp_values}")

    per_nodes_paths: List[str] = []

    if args.mode in ("per_nodes", "both"):
        print("--- Per-nodes integration ---")
        for idx, num_nodes in enumerate(nodes_list):
            path = integrate_per_nodes(
                query_type=args.query_type,
                split=args.split,
                base_dir=args.base_dir,
                num_nodes=num_nodes,
                prompting_version=prompting_versions[idx],
                run_timestamp=args.run_timestamp,
                manifest_indices=manifest_indices,
                seed=seed,
                dp_values=dp_values,
                variant_assignment=args.variant_assignment,
                config_fingerprint=config_fingerprints[idx],
            )
            per_nodes_paths.append(path)

    if args.mode in ("cross_nodes", "both"):
        if args.mode == "cross_nodes":
            for idx, num_nodes in enumerate(nodes_list):
                pattern = os.path.join(
                    args.base_dir,
                    _node_dir_name(num_nodes, prompting_versions[idx]),
                    f"*-{args.query_type}-nl-{args.split}-nodes{num_nodes}.jsonl",
                )
                matches = sorted(glob.glob(pattern))
                if not matches:
                    raise FileNotFoundError(f"No per-nodes file found: {pattern}")
                per_nodes_paths.append(matches[-1])
        print("--- Cross-nodes integration ---")
        integrate_cross_nodes(
            query_type=args.query_type,
            split=args.split,
            base_dir=args.base_dir,
            per_nodes_paths=per_nodes_paths,
            run_timestamp=args.run_timestamp,
        )

    print("=== Done ===")


if __name__ == "__main__":
    main()
