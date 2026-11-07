"""
redistribute_manifest_v2.py

Generates redistribution manifests for cross-domain dataset partitioning (v2.2).

Handles 10 query types from v2.0/v2.1 with two size tiers:
- Large tier (att, cf, pn, ps, nie, nde): 3-way train split, 50% eval selection
- Small tier (conditional_prob, joint_prob, marginal_prob, observed_difference):
  3-way train split, 33% eval selection

Each manifest record stores full provenance including old cross-nodes indices
from source symbolic and code datasets.
"""

import argparse
import glob
import hashlib
import os
import random
from typing import Dict, List, Optional, Set, Tuple

from data_generation.integration_pipeline import load_jsonl, write_jsonl

DOMAINS = ["symbolic", "code", "nl"]


# ============================================================
# Query Configuration Registry
# ============================================================

QUERY_CONFIG = {
    "att": {
        "source_version": "v2.0",
        "nodes_range": (3, 10),
        "variants": ["full", "surgery_ancestor_prune"],
        "triplet_dir": "att",
        "size_tier": "large",
        "eval_policy": "half",
    },
    "cf": {
        "source_version": "v2.0",
        "nodes_range": (4, 10),
        "variants": ["full", "surgery_ancestor_prune"],
        "triplet_dir": "cf",
        "size_tier": "large",
        "eval_policy": "half",
    },
    "pn": {
        "source_version": "v2.0",
        "nodes_range": (4, 10),
        "variants": ["full", "surgery_ancestor_prune"],
        "triplet_dir": "pn_ps",
        "size_tier": "large",
        "eval_policy": "half",
    },
    "ps": {
        "source_version": "v2.0",
        "nodes_range": (4, 10),
        "variants": ["full", "surgery_ancestor_prune"],
        "triplet_dir": "pn_ps",
        "size_tier": "large",
        "eval_policy": "half",
    },
    "nie": {
        "source_version": "v2.1",
        "nodes_range": (4, 10),
        "variants": ["full", "surgery_ancestor_prune"],
        "triplet_dir": "nie_nde",
        "size_tier": "large",
        "eval_policy": "half",
    },
    "nde": {
        "source_version": "v2.1",
        "nodes_range": (4, 10),
        "variants": ["full", "surgery_ancestor_prune"],
        "triplet_dir": "nie_nde",
        "size_tier": "large",
        "eval_policy": "half",
    },
    "conditional_prob": {
        "source_version": "v2.0",
        "nodes_range": (3, 10),
        "variants": ["full", "query_ancestor_prune"],
        "triplet_dir": "conditional_prob",
        "size_tier": "small",
        "eval_policy": "third",
    },
    "joint_prob": {
        "source_version": "v2.0",
        "nodes_range": (3, 10),
        "variants": ["full", "query_ancestor_prune"],
        "triplet_dir": "joint_prob",
        "size_tier": "small",
        "eval_policy": "third",
    },
    "marginal_prob": {
        "source_version": "v2.0",
        "nodes_range": (3, 10),
        "variants": ["full", "query_ancestor_prune"],
        "triplet_dir": "marginal_prob",
        "size_tier": "small",
        "eval_policy": "third",
    },
    "observed_difference": {
        "source_version": "v2.0",
        "nodes_range": (3, 10),
        "variants": ["full", "query_ancestor_prune"],
        "triplet_dir": "observed_difference",
        "size_tier": "small",
        "eval_policy": "third",
    },
}


# ============================================================
# Source File Discovery & Preflight
# ============================================================

def find_v2_cross_nodes_file(
    data_root: str,
    query_type: str,
    split: str,
    domain: str,
    config: Optional[Dict] = None,
) -> Optional[str]:
    """Find a v2.x cross-nodes JSONL file.

    domain: "symbolic" or "code".
    Returns the path, or None if graceful_missing is True and file doesn't exist.
    Raises FileNotFoundError if more than one file matches.
    """
    if config is None:
        config = QUERY_CONFIG[query_type]
    version = config["source_version"]
    base = os.path.join(data_root, version, query_type)

    if domain == "symbolic":
        pattern = os.path.join(base, f"*-{query_type}-{split}.jsonl")
    elif domain == "code":
        pattern = os.path.join(base, f"*-{query_type}-code-{split}.jsonl")
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    matches = [
        p for p in glob.glob(pattern)
        if "subset" not in os.path.basename(p)
        and not os.path.basename(p).startswith(f"{version}-")
    ]

    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Multiple cross-nodes files found for {version}/{query_type}/{split}/{domain}. "
        f"Keep only the run to redistribute.\n"
        f"  Candidates: {matches}"
    )


def preflight_validate(
    data_root: str,
    query_types: List[str],
) -> Dict[str, Dict[str, Dict[str, Optional[str]]]]:
    """Validate all source cross-nodes files exist and are unambiguous.

    Returns nested dict: {query_type: {split: {"symbolic": path, "code": path}}}.
    Raises on any fatal error (missing required file or ambiguous candidates).
    """
    resolved = {}
    errors = []

    for qt in query_types:
        config = QUERY_CONFIG[qt]
        resolved[qt] = {}

        for split in ["train", "eval"]:
            resolved[qt][split] = {}

            for domain in ["symbolic", "code"]:
                try:
                    path = find_v2_cross_nodes_file(
                        data_root, qt, split, domain, config,
                    )
                except FileNotFoundError as e:
                    errors.append(str(e))
                    continue

                if path is None:
                    errors.append(
                        f"Required source file not found: "
                        f"{config['source_version']}/{qt}/{split}/{domain}"
                    )
                else:
                    resolved[qt][split][domain] = path

    if errors:
        raise FileNotFoundError(
            "Preflight validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return resolved


# ============================================================
# Cross-Nodes Index Lookup
# ============================================================

def build_cross_nodes_lookup(
    cross_nodes_path: str,
) -> Dict[Tuple[int, int], int]:
    """Build a (num_nodes, per_nodes_position) -> cross_nodes_idx mapping."""
    records = load_jsonl(cross_nodes_path)
    lookup = {}
    position_counter: Dict[int, int] = {}

    for record in records:
        num_nodes = record["metadata"]["num_nodes"]
        if num_nodes not in position_counter:
            position_counter[num_nodes] = 0
        per_nodes_pos = position_counter[num_nodes]
        position_counter[num_nodes] += 1
        lookup[(num_nodes, per_nodes_pos)] = record["idx"]

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
# Stratified Apportionment (Hamilton's Method)
# ============================================================

def apportion_hamilton(
    strata_sizes: List[int],
    global_target: int,
) -> List[int]:
    """Apportion global_target across strata using largest-remainder method.

    Guarantees sum(result) == global_target exactly.
    """
    total = sum(strata_sizes)
    if total == 0:
        return [0] * len(strata_sizes)

    quotas = [s * global_target / total for s in strata_sizes]
    floors = [int(q) for q in quotas]
    remainders = [q - f for q, f in zip(quotas, floors)]

    deficit = global_target - sum(floors)
    indices_by_remainder = sorted(
        range(len(remainders)), key=lambda i: remainders[i], reverse=True,
    )
    for i in range(deficit):
        floors[indices_by_remainder[i]] += 1

    return floors


# ============================================================
# Manifest Generation
# ============================================================

def generate_manifest(
    query_type: str,
    split: str,
    base_seed: int,
    source_records: List[Dict],
    symbolic_lookup: Optional[Dict[Tuple[int, int], int]],
    code_lookup: Optional[Dict[Tuple[int, int], int]],
    source_symbolic_basename: Optional[str],
    source_code_basename: Optional[str],
) -> List[Dict]:
    """Generate manifest entries for a single (query_type, split).

    Groups source records by (num_nodes, decimal_places, condition_variant)
    strata, apportions the global target across strata, then assigns domains
    (train) or selects shared subset (eval).
    """
    config = QUERY_CONFIG[query_type]
    source_version = config["source_version"]
    eval_policy = config["eval_policy"]
    variants = config["variants"]

    # Group records by stratum
    strata: Dict[Tuple[int, int, str], List[int]] = {}
    for record in source_records:
        m = record["metadata"]
        key = (m["num_nodes"], m["decimal_places"], m["variant_name"])
        if key not in strata:
            strata[key] = []
        strata[key].append(record["idx"])

    # Build position lookup: (num_nodes, idx) -> per-nodes position
    position_by_nodes: Dict[int, Dict[int, int]] = {}
    for record in source_records:
        n = record["metadata"]["num_nodes"]
        if n not in position_by_nodes:
            position_by_nodes[n] = {}
        if record["idx"] not in position_by_nodes[n]:
            position_by_nodes[n][record["idx"]] = len(position_by_nodes[n])

    total = len(source_records)

    # Compute global target per domain (train)
    if split == "train":
        num_domains = 3
        domains_for_train = DOMAINS
        global_target_per_domain = total // num_domains

    # Sort strata keys for deterministic ordering
    sorted_strata_keys = sorted(strata.keys())
    strata_sizes = [len(strata[k]) for k in sorted_strata_keys]

    # Pre-shuffle all strata for deterministic ordering
    shuffled_strata = {}
    for stratum_key in sorted_strata_keys:
        num_nodes, dp, variant = stratum_key
        indices = strata[stratum_key]
        seed = compute_stratum_seed(
            query_type, split, num_nodes, dp, variant, base_seed,
        )
        rng = random.Random(seed)
        shuffled_strata[stratum_key] = (rng.sample(indices, len(indices)), seed)

    def _make_entry(idx, num_nodes, dp, variant, domain_assignment, selected, seed):
        per_nodes_pos = position_by_nodes.get(num_nodes, {}).get(idx)
        sym_cn_idx = None
        code_cn_idx = None
        if symbolic_lookup and per_nodes_pos is not None:
            sym_cn_idx = symbolic_lookup.get((num_nodes, per_nodes_pos))
        if code_lookup and per_nodes_pos is not None:
            code_cn_idx = code_lookup.get((num_nodes, per_nodes_pos))

        return {
            "triplet_idx": idx,
            "num_nodes": num_nodes,
            "decimal_places": dp,
            "condition_variant": variant,
            "domain_assignment": domain_assignment,
            "selected": selected,
            "old_source_cross_nodes_idx": sym_cn_idx,
            "old_code_cross_nodes_idx": code_cn_idx,
            "source_symbolic_dataset": source_symbolic_basename,
            "source_code_dataset": source_code_basename,
            "selection_seed": seed,
            "source_version": source_version,
        }

    entries = []

    if split == "train":
        # Two-phase training assignment:
        # Phase 1: per-stratum safe assignment (floor(stratum_size / num_domains))
        # Phase 2: distribute remainder indices to fill global target
        domain_counts = {d: 0 for d in domains_for_train}
        remainder_pool = []

        for stratum_key in sorted_strata_keys:
            num_nodes, dp, variant = stratum_key
            shuffled, seed = shuffled_strata[stratum_key]
            k = len(shuffled) // num_domains

            for domain_idx, domain in enumerate(domains_for_train):
                assigned = shuffled[domain_idx * k : (domain_idx + 1) * k]
                for idx in assigned:
                    entries.append(
                        _make_entry(idx, num_nodes, dp, variant, domain, True, seed)
                    )
                domain_counts[domain] += k

            for idx in shuffled[k * num_domains:]:
                remainder_pool.append((idx, num_nodes, dp, variant, seed))

        # Phase 2: distribute remainder to fill deficit
        rng_global = random.Random(base_seed)
        rng_global.shuffle(remainder_pool)

        rr_idx = 0
        for idx, num_nodes, dp, variant, seed in remainder_pool:
            filled = all(
                domain_counts[d] >= global_target_per_domain
                for d in domains_for_train
            )
            if filled:
                entries.append(
                    _make_entry(idx, num_nodes, dp, variant, "excluded", False, seed)
                )
            else:
                domain = domains_for_train[rr_idx % num_domains]
                while domain_counts[domain] >= global_target_per_domain:
                    rr_idx += 1
                    domain = domains_for_train[rr_idx % num_domains]
                entries.append(
                    _make_entry(idx, num_nodes, dp, variant, domain, True, seed)
                )
                domain_counts[domain] += 1
                rr_idx += 1

    else:
        # Eval: use Hamilton's method for per-stratum selection
        if eval_policy == "half":
            global_target_shared = total // 2
        elif eval_policy == "third":
            global_target_shared = total // 3
        else:
            raise ValueError(f"Unknown eval_policy: {eval_policy}")

        per_stratum_shared = apportion_hamilton(
            strata_sizes, global_target_shared,
        )

        for stratum_idx, stratum_key in enumerate(sorted_strata_keys):
            num_nodes, dp, variant = stratum_key
            shuffled, seed = shuffled_strata[stratum_key]
            k = per_stratum_shared[stratum_idx]

            for idx in shuffled[:k]:
                entries.append(
                    _make_entry(idx, num_nodes, dp, variant, "shared", True, seed)
                )
            for idx in shuffled[k:]:
                entries.append(
                    _make_entry(idx, num_nodes, dp, variant, "excluded", False, seed)
                )

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
    config = QUERY_CONFIG[query_type]
    total = len(entries)

    if split == "train":
        num_domains = 3
        expected_per_domain = total // num_domains

        domain_sets: Dict[str, Set[Tuple[int, int]]] = {d: set() for d in DOMAINS}
        for entry in entries:
            if entry["domain_assignment"] in DOMAINS:
                key = (entry["num_nodes"], entry["triplet_idx"])
                domain = entry["domain_assignment"]
                if key in domain_sets[domain]:
                    raise RuntimeError(f"Duplicate triplet in {domain}: {key}")
                domain_sets[domain].add(key)

        for domain in DOMAINS:
            actual = len(domain_sets[domain])
            if actual != expected_per_domain:
                raise RuntimeError(
                    f"{query_type}/{split}: {domain} count: "
                    f"expected {expected_per_domain}, got {actual}"
                )

        for i in range(len(DOMAINS)):
            for j in range(i + 1, len(DOMAINS)):
                overlap = domain_sets[DOMAINS[i]] & domain_sets[DOMAINS[j]]
                if overlap:
                    raise RuntimeError(
                        f"{query_type}/{split}: overlap between "
                        f"{DOMAINS[i]} and {DOMAINS[j]}: {len(overlap)} triplets"
                    )

        for entry in entries:
            if entry["old_source_cross_nodes_idx"] is None:
                raise RuntimeError(
                    f"{query_type}/{split}: null old_source_cross_nodes_idx "
                    f"(triplet_idx={entry['triplet_idx']})"
                )

        print(
            f"  Validation passed: {query_type}/{split} — "
            f"{expected_per_domain}/domain × {num_domains} domains "
            f"({total} total)"
        )

    else:
        eval_policy = config["eval_policy"]
        if eval_policy == "half":
            expected_selected = total // 2
        else:
            expected_selected = total // 3

        selected = [e for e in entries if e["selected"]]
        if len(selected) != expected_selected:
            raise RuntimeError(
                f"{query_type}/{split}: eval selection count: "
                f"expected {expected_selected}, got {len(selected)}"
            )

        # Check uniqueness among selected
        selected_keys = set()
        for e in selected:
            key = (e["num_nodes"], e["triplet_idx"])
            if key in selected_keys:
                raise RuntimeError(f"Duplicate selected triplet: {key}")
            selected_keys.add(key)

        print(
            f"  Validation passed: {query_type}/{split} — "
            f"{expected_selected} selected / {total} total "
            f"(policy: {eval_policy})"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate redistribution manifests for v2.2 cross-domain partitioning"
    )
    parser.add_argument(
        "--data_root",
        default="./data_generation/generated_datasets",
        help="Root directory containing v2.0/ and v2.1/ datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="./data_generation/generated_datasets/v2.2/manifests",
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
        default=list(QUERY_CONFIG.keys()),
        help="Query types to process",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Preflight: resolve all source files
    print("=== Preflight Validation ===")
    resolved = preflight_validate(args.data_root, args.query_types)
    print("Preflight passed.\n")

    for query_type in args.query_types:
        for split in ["train", "eval"]:
            print(f"\n=== {query_type} / {split} ===")

            sym_path = resolved[query_type][split]["symbolic"]
            code_path = resolved[query_type][split]["code"]

            source_records = load_jsonl(sym_path)
            sym_basename = os.path.splitext(os.path.basename(sym_path))[0]
            code_basename = os.path.splitext(os.path.basename(code_path))[0]
            print(f"  Source symbolic: {sym_path} ({len(source_records)} records)")

            symbolic_lookup = build_cross_nodes_lookup(sym_path)
            code_lookup = build_cross_nodes_lookup(code_path)

            entries = generate_manifest(
                query_type, split, args.base_seed,
                source_records, symbolic_lookup, code_lookup,
                sym_basename, code_basename,
            )

            validate_manifest(entries, query_type, split)

            # Sort by (num_nodes, triplet_idx) for deterministic output
            entries.sort(key=lambda e: (e["num_nodes"], e["triplet_idx"]))

            # Write manifest
            filename = f"{args.run_timestamp}-{query_type}-{split}-manifest.jsonl"
            output_path = os.path.join(args.output_dir, filename)
            write_jsonl(output_path, entries)
            print(f"  Wrote: {output_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
