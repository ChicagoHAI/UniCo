"""
integration_pipeline.py

Integrates assembly outputs into unified dataset files. Two integration levels:

  - per_nodes: for a single (query, split, #nodes), collects records across
    (variant, dp) combos in dp-major order. No re-indexing.
  - cross_nodes: concatenates per-nodes files in ascending #nodes order,
    then re-indexes globally from 0.

Most numerical/causal query families use 2 condition variants
(`full`, `surgery_ancestor_prune`) x 4 decimal_places values = 8 combos.
The standalone association numerical families use
(`full`, `query_ancestor_prune`) x 4 decimal_places values = 8 combos.
The identification, backdoor, frontdoor, independence_test, and markov_blanket
families are single-variant special cases using `full`. Identification/backdoor/frontdoor
and markov_blanket use dp=0; `independence_test` uses dp=2 only because it remains
CPT-based. All five integrate across the three query_format variants
{binary, multi_choice, multi_answer}.
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
from typing import Dict, List


# ============================================================
# Distribution Table
# ============================================================
# Values are num_examples (total triplets for that split).
# Train = 3V, Eval = V from the spec.
# For the standard case, per_combo = num_examples // 8  (4 dp values x 2 variants).
# Identification, backdoor, and frontdoor are the special cases: one variant and dp=0.

DISTRIBUTION = {
    "ate": {
        "train": {3: 120, 4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {3: 16,  4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "cate": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "jate": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "cf": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "att": {
        "train": {3: 120, 4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {3: 16,  4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "nie": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "nde": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "pn": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "ps": {
        "train": {4: 600, 5: 960, 6: 1200, 7: 1200, 8: 960, 9: 600, 10: 360},
        "eval":  {4: 80,  5: 128, 6: 160,  7: 160,  8: 128, 9: 80,  10: 48},
    },
    "identification": {
        "train": {4: 153, 5: 246, 6: 306, 7: 306, 8: 246, 9: 150, 10: 93},
        "eval":  {4: 30, 5: 48, 6: 63, 7: 63, 8: 48, 9: 30, 10: 18},
    },
    "backdoor": {
        "train": {4: 153, 5: 246, 6: 306, 7: 306, 8: 246, 9: 150, 10: 93},
        "eval":  {4: 30, 5: 48, 6: 63, 7: 63, 8: 48, 9: 30, 10: 18},
    },
    "frontdoor": {
        "train": {4: 153, 5: 246, 6: 306, 7: 306, 8: 246, 9: 150, 10: 93},
        "eval":  {4: 30, 5: 48, 6: 63, 7: 63, 8: 48, 9: 30, 10: 18},
    },
    "marginal_prob": {
        "train": {3: 30, 4: 150, 5: 240, 6: 300, 7: 300, 8: 240, 9: 150, 10: 90},
        "eval":  {3: 6,  4: 30,  5: 48,  6: 60,  7: 60,  8: 48,  9: 30,  10: 18},
    },
    "conditional_prob": {
        "train": {3: 30, 4: 150, 5: 240, 6: 300, 7: 300, 8: 240, 9: 150, 10: 90},
        "eval":  {3: 6,  4: 30,  5: 48,  6: 60,  7: 60,  8: 48,  9: 30,  10: 18},
    },
    "joint_prob": {
        "train": {3: 30, 4: 150, 5: 240, 6: 300, 7: 300, 8: 240, 9: 150, 10: 90},
        "eval":  {3: 6,  4: 30,  5: 48,  6: 60,  7: 60,  8: 48,  9: 30,  10: 18},
    },
    "observed_difference": {
        "train": {3: 30, 4: 150, 5: 240, 6: 300, 7: 300, 8: 240, 9: 150, 10: 90},
        "eval":  {3: 6,  4: 30,  5: 48,  6: 60,  7: 60,  8: 48,  9: 30,  10: 18},
    },
    "independence_test": {
        "train": {4: 153, 5: 246, 6: 306, 7: 306, 8: 246, 9: 150, 10: 93},
        "eval":  {4: 30, 5: 48, 6: 63, 7: 63, 8: 48, 9: 30, 10: 18},
    },
    "markov_blanket": {
        "train": {4: 153, 5: 246, 6: 306, 7: 306, 8: 246, 9: 150, 10: 93},
        "eval":  {4: 30, 5: 48, 6: 63, 7: 63, 8: 48, 9: 30, 10: 18},
    },
}

SEEDS = {"train": 0, "eval": 1000000}
DP_VALUES = [1, 2, 3, 4]
GRAPH_ONLY_QUERY_FORMATS = ["binary", "multi_choice", "multi_answer"]
SINGLE_VARIANT_MULTI_FORMAT_QUERY_TYPES = {
    "identification", "backdoor", "frontdoor", "independence_test", "markov_blanket"
}
ASSOCIATION_NUMERICAL_QUERY_TYPES = {
    "marginal_prob", "conditional_prob", "joint_prob", "observed_difference"
}
MULTI_FORMAT_QUERY_TYPES = SINGLE_VARIANT_MULTI_FORMAT_QUERY_TYPES
NO_DP_QUERY_TYPES = {"identification", "backdoor", "frontdoor", "markov_blanket"}
# CATE and JATE share one folder per node count, so file discovery also filters on the query
# type in the filename.
SHARED_FOLDER_QUERY_TYPES = {"cate", "jate"}


# ============================================================
# Path / Filename Helpers
# ============================================================

def _domain_filename_infix(domain: str) -> str:
    if domain == "symbolic":
        return ""
    return f"-{domain}"


def _node_dir_name(num_nodes: int, domain: str, prompting_version: str | None) -> str:
    if domain == "nl":
        if prompting_version is None:
            raise ValueError("prompting_version is required for domain='nl'")
        return f"nodes{num_nodes}-prompting_{prompting_version}"
    return f"nodes{num_nodes}"


def _variant_dir_candidates(variant_name: str, domain: str) -> List[str]:
    if domain == "symbolic":
        return [f"{variant_name}-symbolic", variant_name]
    return [f"{variant_name}-{domain}"]


# ============================================================
# Combo Sizing
# ============================================================

def _num_records_for_combo(
    query_type: str,
    num_examples: int,
    *,
    num_dp_values: int = 0,
    dp_idx: int = 0,
    query_format_idx: int = 0,
) -> int:
    if query_type in ASSOCIATION_NUMERICAL_QUERY_TYPES or query_type in {
        "ate", "cate", "jate", "cf", "att", "nie", "nde", "pn", "ps"
    }:
        if num_dp_values <= 0:
            raise ValueError("num_dp_values must be positive for two-variant queries")
        half_total = num_examples // 2
        base_per_combo = half_total // num_dp_values
        remainder = half_total % num_dp_values
        return base_per_combo + (1 if dp_idx < remainder else 0)
    if query_type == "independence_test":
        if num_dp_values <= 0:
            raise ValueError("num_dp_values must be positive for independence_test")
        num_combos = num_dp_values * len(GRAPH_ONLY_QUERY_FORMATS)
        base = num_examples // num_combos
        remainder = num_examples % num_combos
        combo_idx = dp_idx * len(GRAPH_ONLY_QUERY_FORMATS) + query_format_idx
        return base + (1 if combo_idx < remainder else 0)
    if query_type in {"identification", "backdoor", "frontdoor", "markov_blanket"}:
        num_combos = len(GRAPH_ONLY_QUERY_FORMATS)
        base = num_examples // num_combos
        remainder = num_examples % num_combos
        return base + (1 if query_format_idx < remainder else 0)
    raise ValueError(f"Unsupported combo sizing for query_type={query_type}")


def _start_end_for_single_variant_combo(
    query_type: str,
    num_examples: int,
    *,
    num_dp_values: int = 0,
    dp_idx: int = 0,
    query_format_idx: int = 0,
) -> tuple[int, int]:
    if query_type in {"identification", "backdoor", "frontdoor", "markov_blanket"}:
        start_idx = 0
        for prev_query_idx in range(query_format_idx):
            start_idx += _num_records_for_combo(
                query_type,
                num_examples,
                query_format_idx=prev_query_idx,
            )
        end_idx = start_idx + _num_records_for_combo(
            query_type,
            num_examples,
            query_format_idx=query_format_idx,
        )
        return start_idx, end_idx

    if query_type == "independence_test":
        start_idx = 0
        for prev_dp_idx in range(dp_idx):
            for prev_query_idx in range(len(GRAPH_ONLY_QUERY_FORMATS)):
                start_idx += _num_records_for_combo(
                    query_type,
                    num_examples,
                    num_dp_values=num_dp_values,
                    dp_idx=prev_dp_idx,
                    query_format_idx=prev_query_idx,
                )
        for prev_query_idx in range(query_format_idx):
            start_idx += _num_records_for_combo(
                query_type,
                num_examples,
                num_dp_values=num_dp_values,
                dp_idx=dp_idx,
                query_format_idx=prev_query_idx,
            )
        end_idx = start_idx + _num_records_for_combo(
            query_type,
            num_examples,
            num_dp_values=num_dp_values,
            dp_idx=dp_idx,
            query_format_idx=query_format_idx,
        )
        return start_idx, end_idx

    raise ValueError(f"Unsupported single-variant combo indexing for query_type={query_type}")


def _start_end_for_two_variant_dp_combo(
    num_examples: int,
    *,
    num_dp_values: int,
    dp_idx: int,
) -> tuple[int, int, int]:
    if num_dp_values <= 0:
        raise ValueError("num_dp_values must be positive for two-variant queries")
    half_total = num_examples // 2
    base_per_combo = half_total // num_dp_values
    remainder = half_total % num_dp_values
    start_idx = 0
    for prev_dp_idx in range(dp_idx):
        prev_per_combo = base_per_combo + (1 if prev_dp_idx < remainder else 0)
        start_idx += prev_per_combo * 2
    per_combo = base_per_combo + (1 if dp_idx < remainder else 0)
    end_idx = start_idx + per_combo * 2
    return start_idx, end_idx, per_combo


def _variant_names_for_query_type(query_type: str) -> List[str]:
    if query_type in SINGLE_VARIANT_MULTI_FORMAT_QUERY_TYPES:
        return ["full"]
    if query_type in ASSOCIATION_NUMERICAL_QUERY_TYPES:
        return ["full", "query_ancestor_prune"]
    return ["full", "surgery_ancestor_prune"]


# ============================================================
# I/O Helpers
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, records: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "question", "answer", "metadata"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "idx": r["idx"],
                "question": r["question"],
                "answer": r["answer"],
                "metadata": json.dumps(r["metadata"], ensure_ascii=False),
            })


def _aggregate_difficulty_metrics(records: List[Dict]) -> Dict:
    num = len(records)
    if num == 0:
        return {
            "avg_treatment_outcome_depth": 0,
            "avg_effective_prob_conds": 0,
            "naive_conditioning_rate": 0,
        }
    first_meta = records[0].get("metadata", {})
    if "treatment_outcome_depth" not in first_meta or "effective_prob_conds" not in first_meta:
        return {}
    total_depth = sum(r["metadata"]["treatment_outcome_depth"] for r in records)
    total_eff = sum(r["metadata"]["effective_prob_conds"] for r in records)
    naive_count = 0
    for r in records:
        m = r["metadata"]
        if "has_naive_conditioning" in m:
            val = m["has_naive_conditioning"]
        elif "has_naive_shortcut" in m:
            val = m["has_naive_shortcut"]
        else:
            raise KeyError(
                f"Record idx={r.get('idx')} missing both "
                "'has_naive_conditioning' and 'has_naive_shortcut' in metadata"
            )
        if val:
            naive_count += 1
    return {
        "avg_treatment_outcome_depth": total_depth / num,
        "avg_effective_prob_conds": total_eff / num,
        "naive_conditioning_rate": naive_count / num,
    }


def compute_stats(
    records: List[Dict],
    source_files: List[str],
    *,
    include_per_node_breakdown: bool = False,
) -> Dict:
    num_examples = len(records)

    abnormal_count = sum(1 for r in records if r["metadata"]["is_abnormal"])
    verification_false_count = sum(
        1
        for r in records
        if "verification_ok" in r["metadata"] and r["metadata"]["verification_ok"] is False
    )

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
    num_unique_dags = len(dag_edge_sets)

    difficulty = _aggregate_difficulty_metrics(records)

    stats = {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples if num_examples else 0,
        "verification_false_count": verification_false_count,
        "verification_false_rate": (
            verification_false_count / num_examples if num_examples else 0
        ),
    }

    if include_per_node_breakdown and difficulty:
        by_nodes = collections.defaultdict(list)
        for r in records:
            n = r["metadata"]["num_nodes"]
            by_nodes[n].append(r)
        sorted_nodes = sorted(by_nodes.keys())

        node_diffs = {
            n: _aggregate_difficulty_metrics(by_nodes[n])
            for n in sorted_nodes
        }

        stats["avg_treatment_outcome_depth"] = difficulty["avg_treatment_outcome_depth"]
        for n in sorted_nodes:
            stats[f"nodes{n}_avg_treatment_outcome_depth"] = node_diffs[n]["avg_treatment_outcome_depth"]

        stats["avg_effective_prob_conds"] = difficulty["avg_effective_prob_conds"]
        for n in sorted_nodes:
            stats[f"nodes{n}_avg_effective_prob_conds"] = node_diffs[n]["avg_effective_prob_conds"]

        stats["naive_conditioning_rate"] = difficulty["naive_conditioning_rate"]
        for n in sorted_nodes:
            stats[f"nodes{n}_naive_conditioning_rate"] = node_diffs[n]["naive_conditioning_rate"]
    elif difficulty:
        stats.update(difficulty)

    stats["per_combo_counts"] = combo_counts
    if records and records[0]["metadata"].get("query_type") in MULTI_FORMAT_QUERY_TYPES:
        query_format_counts = {}
        for r in records:
            fmt = r["metadata"].get("query_format", "unknown")
            query_format_counts[fmt] = query_format_counts.get(fmt, 0) + 1
        stats["query_format_counts"] = query_format_counts
    stats["source_files"] = source_files
    return stats


def write_stats(
    path: str,
    records: List[Dict],
    source_files: List[str],
    *,
    include_per_node_breakdown: bool = False,
) -> None:
    stats = compute_stats(
        records,
        source_files,
        include_per_node_breakdown=include_per_node_breakdown,
    )
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ============================================================
# File Discovery
# ============================================================

def find_assembly_file(
    base_dir: str,
    num_nodes: int,
    variant_name: str,
    start_idx: int,
    end_idx: int,
    dp: int,
    seed: int,
    query_type: str,
    query_format: str | None = None,
    domain: str = "symbolic",
    prompting_version: str | None = None,
    allow_missing: bool = False,
    config_fingerprint: str | None = None,
    variant_assignment: str | None = None,
) -> str | None:
    """Find the assembly output JSONL for a specific combo."""
    pattern = None
    if query_type in SHARED_FOLDER_QUERY_TYPES:
        pattern = (
            f"*-start{start_idx}-end{end_idx}"
            f"-*-{query_type}-*-dp{dp}-seed{seed}*.jsonl"
        )
    elif query_type in ["ate", "cf", "att", "nie", "nde", "pn", "ps",
                        "marginal_prob", "conditional_prob", "joint_prob", "observed_difference"]:
        pattern = (
            f"*-start{start_idx}-end{end_idx}"
            f"-*-dp{dp}-seed{seed}*.jsonl"
        )
    elif query_type == "independence_test":
        pattern = (
            f"*-start{start_idx}-end{end_idx}"
            f"-*-{query_type}-*-fmt{query_format}-*-dp{dp}-seed{seed}*.jsonl"
        )
    elif query_type == "markov_blanket":
        pattern = (
            f"*-start{start_idx}-end{end_idx}"
            f"-*-{query_type}-*-fmt{query_format}-seed{seed}*.jsonl"
        )
    elif query_type in SINGLE_VARIANT_MULTI_FORMAT_QUERY_TYPES:
        pattern = (
            f"*-start{start_idx}-end{end_idx}"
            f"-*-fmt{query_format}-seed{seed}*.jsonl"
        )
    else:
        raise ValueError(f"Unsupported query_type for file discovery: {query_type}")

    all_matches: List[str] = []
    node_dir = _node_dir_name(num_nodes, domain, prompting_version)
    for variant_dir in _variant_dir_candidates(variant_name, domain):
        search_dir = os.path.join(base_dir, node_dir, variant_dir, "jsonls")
        if not os.path.isdir(search_dir):
            continue
        matches = sorted(glob.glob(os.path.join(search_dir, pattern)))
        all_matches.extend(matches)

    if variant_assignment is not None:
        all_matches = [m for m in all_matches if f"-va_{variant_assignment}" in os.path.basename(m)]
    if config_fingerprint is not None:
        all_matches = [m for m in all_matches if f"-cfg{config_fingerprint}" in os.path.basename(m)]

    if not all_matches:
        search_hint = os.path.join(base_dir, node_dir, "<variant>", "jsonls", pattern)
        if allow_missing:
            return None
        raise FileNotFoundError(f"No assembly output found: {search_hint}")

    matches = sorted(all_matches)
    if len(matches) > 1 and config_fingerprint is None and domain == "nl":
        cfg_files = []
        cfg_hashes = set()
        for m in matches:
            cfg_match = re.search(r"-cfg([a-f0-9]+)", os.path.basename(m))
            if cfg_match:
                cfg_files.append(m)
                cfg_hashes.add(cfg_match.group(1))
        if cfg_files:
            if len(cfg_hashes) > 1:
                basenames = [os.path.basename(m) for m in matches]
                raise RuntimeError(
                    f"Ambiguous NL assembly match: {len(cfg_hashes)} different config presets found. "
                    f"Use --config_fingerprint to disambiguate. Matches: {basenames}"
                )
            matches = sorted(cfg_files)

    return matches[-1]


# ============================================================
# Per-Nodes Integration
# ============================================================

def _slice_single_variant_records(records: List[Dict], start_idx: int, end_idx: int) -> List[Dict]:
    by_idx = [r for r in records if start_idx <= r.get("idx", -1) < end_idx]
    if by_idx:
        return by_idx
    if len(records) == (end_idx - start_idx):
        return records
    return records[: max(0, end_idx - start_idx)]


def _check_record_query_types(records: List[Dict], query_type: str, num_nodes: int) -> None:
    """Raise if a CATE or JATE record belongs to the other query type: discovery picked a wrong file."""
    if query_type not in SHARED_FOLDER_QUERY_TYPES:
        return
    found_query_types = {r["metadata"]["query_type"] for r in records}
    if found_query_types - {query_type}:
        raise RuntimeError(
            f"Integrating {query_type} for nodes={num_nodes} found records of "
            f"query types {sorted(found_query_types)}"
        )


def integrate_per_nodes(
    query_type: str,
    split: str,
    base_dir: str,
    num_nodes: int,
    run_timestamp: str,
    num_examples: int,
    seed: int,
    dp_values: List[int],
    domain: str = "symbolic",
    prompting_version: str | None = None,
    config_fingerprint: str | None = None,
    variant_assignment: str | None = None,
) -> str:
    variant_names = _variant_names_for_query_type(query_type)
    query_formats = GRAPH_ONLY_QUERY_FORMATS if query_type in MULTI_FORMAT_QUERY_TYPES else [None]

    all_records: List[Dict] = []
    source_files: List[str] = []
    num_dp_values = len(dp_values)

    if query_type in SINGLE_VARIANT_MULTI_FORMAT_QUERY_TYPES:
        for dp_idx, dp in enumerate(dp_values):
            for query_format_idx, query_format in enumerate(query_formats):
                start_idx, end_idx = _start_end_for_single_variant_combo(
                    query_type,
                    num_examples,
                    num_dp_values=num_dp_values,
                    dp_idx=dp_idx,
                    query_format_idx=query_format_idx,
                )
                per_combo = end_idx - start_idx
                full_path = find_assembly_file(
                    base_dir,
                    num_nodes,
                    "full",
                    start_idx,
                    end_idx,
                    dp,
                    seed,
                    query_type,
                    query_format=query_format,
                    domain=domain,
                    prompting_version=prompting_version,
                    config_fingerprint=config_fingerprint,
                    variant_assignment=variant_assignment,
                )
                full_records = load_jsonl(full_path)
                source_files.append(full_path)
                full_slice = _slice_single_variant_records(full_records, start_idx, end_idx)
                if len(full_slice) > per_combo:
                    full_slice = full_slice[:per_combo]
                for r in full_slice:
                    r["metadata"]["decimal_places"] = dp
                all_records.extend(full_slice)
    else:
        full_variant, second_variant = variant_names
        for dp_idx, dp in enumerate(dp_values):
            start_idx, end_idx, per_combo = _start_end_for_two_variant_dp_combo(
                num_examples,
                num_dp_values=num_dp_values,
                dp_idx=dp_idx,
            )

            full_path = find_assembly_file(
                base_dir,
                num_nodes,
                full_variant,
                start_idx,
                end_idx,
                dp,
                seed,
                query_type,
                domain=domain,
                prompting_version=prompting_version,
                config_fingerprint=config_fingerprint,
                variant_assignment=variant_assignment,
            )
            second_path = find_assembly_file(
                base_dir,
                num_nodes,
                second_variant,
                start_idx,
                end_idx,
                dp,
                seed,
                query_type,
                domain=domain,
                prompting_version=prompting_version,
                config_fingerprint=config_fingerprint,
                variant_assignment=variant_assignment,
            )
            full_records = load_jsonl(full_path)
            second_records = load_jsonl(second_path)
            source_files.extend([full_path, second_path])

            if domain == "nl":
                mid_idx = (start_idx + end_idx) // 2
                full_slice = [r for r in full_records if start_idx <= r["idx"] < mid_idx]
                second_slice = [r for r in second_records if mid_idx <= r["idx"] < end_idx]
            else:
                full_slice = full_records[:per_combo]
                second_slice = second_records[-per_combo:] if per_combo > 0 else []

            if len(full_slice) < per_combo:
                print(
                    f"  WARN: nodes={num_nodes} dp={dp} {full_variant} variant has "
                    f"{len(full_slice)} records (expected {per_combo})"
                )
            if len(second_slice) < per_combo:
                print(
                    f"  WARN: nodes={num_nodes} dp={dp} {second_variant} variant has "
                    f"{len(second_slice)} records (expected {per_combo})"
                )

            for r in full_slice:
                r["metadata"]["decimal_places"] = dp
            for r in second_slice:
                r["metadata"]["decimal_places"] = dp

            all_records.extend(full_slice)
            all_records.extend(second_slice)

    _check_record_query_types(all_records, query_type, num_nodes)

    expected = num_examples
    actual = len(all_records)
    if actual != expected:
        if domain == "nl" and actual < expected:
            print(
                f"  WARN: Per-nodes NL record count for nodes={num_nodes}: "
                f"{actual}/{expected} ({expected - actual} records dropped by generation)"
            )
        else:
            raise RuntimeError(
                f"Per-nodes record count mismatch for nodes={num_nodes}: expected {expected}, got {actual}"
            )

    output_dir = os.path.join(base_dir, f"nodes{num_nodes}")
    os.makedirs(output_dir, exist_ok=True)
    infix = _domain_filename_infix(domain)
    filename = f"{run_timestamp}-{query_type}{infix}-{split}-nodes{num_nodes}"

    jsonl_path = os.path.join(output_dir, f"{filename}.jsonl")
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    stats_path = os.path.join(output_dir, f"{filename}.json")

    write_jsonl(jsonl_path, all_records)
    write_csv(csv_path, all_records)
    write_stats(stats_path, all_records, source_files)

    print(f"  Per-nodes: {jsonl_path} ({actual} records)")
    return jsonl_path


# ============================================================
# Cross-Nodes Integration
# ============================================================

def integrate_cross_nodes(
    query_type: str,
    split: str,
    base_dir: str,
    per_nodes_paths: List[str],
    run_timestamp: str,
    domain: str = "symbolic",
) -> None:
    all_records = []
    for path in per_nodes_paths:
        records = load_jsonl(path)
        all_records.extend(records)

    for new_idx, r in enumerate(all_records):
        r["idx"] = new_idx

    infix = _domain_filename_infix(domain)
    filename = f"{run_timestamp}-{query_type}{infix}-{split}"
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integrate assembly outputs into unified dataset files"
    )
    parser.add_argument("--query_type", type=str, required=True)
    parser.add_argument(
        "--split", type=str, required=True,
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--base_dir", type=str, required=True,
        help="Base directory for this query type",
    )
    parser.add_argument("--run_timestamp", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["per_nodes", "cross_nodes", "both"],
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed for assembly file discovery. Defaults to SEEDS[split].",
    )
    parser.add_argument(
        "--dp_values", type=int, nargs="+", default=None,
        help="Decimal places values.",
    )
    parser.add_argument(
        "--num_nodes_list", type=int, nargs="+", default=None,
        help="Node counts to integrate (overrides DISTRIBUTION).",
    )
    parser.add_argument(
        "--num_examples_list", type=int, nargs="+", default=None,
        help="Num examples per node count, parallel with --num_nodes_list (overrides DISTRIBUTION).",
    )
    parser.add_argument(
        "--domain", type=str, default="symbolic",
        choices=["symbolic", "code", "nl"],
        help="Domain for variant folder resolution and filename suffix (default: symbolic).",
    )
    parser.add_argument(
        "--prompting_version", type=str, default=None,
        choices=["v2", "v3"],
        help="Prompting pipeline version for NL domain.",
    )
    parser.add_argument(
        "--prompting_version_list", type=str, nargs="+", default=None,
        choices=["v2", "v3"],
        help="Per-node prompting versions for NL domain.",
    )
    parser.add_argument(
        "--config_fingerprint", type=str, default=None,
        help="Optional config fingerprint for NL file disambiguation.",
    )
    parser.add_argument(
        "--config_fingerprint_list", type=str, nargs="+", default=None,
        help="Per-node config fingerprints for NL file disambiguation.",
    )
    parser.add_argument(
        "--variant_assignment", type=str, default=None,
        choices=["split", "both"],
        help="Optional NL variant assignment setting used to filter files.",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else SEEDS[args.split]
    if args.dp_values is not None:
        dp_values = args.dp_values
    elif args.query_type in NO_DP_QUERY_TYPES:
        dp_values = [0]
    elif args.query_type == "independence_test":
        dp_values = [2]
    else:
        dp_values = DP_VALUES

    if args.num_nodes_list is not None and args.num_examples_list is not None:
        if len(args.num_nodes_list) != len(args.num_examples_list):
            raise ValueError(
                f"--num_nodes_list ({len(args.num_nodes_list)}) and "
                f"--num_examples_list ({len(args.num_examples_list)}) must have the same length"
            )
        distribution = dict(zip(args.num_nodes_list, args.num_examples_list))
    elif args.num_nodes_list is not None or args.num_examples_list is not None:
        raise ValueError("--num_nodes_list and --num_examples_list must be specified together")
    else:
        distribution = DISTRIBUTION[args.query_type][args.split]

    nodes_list = sorted(distribution.keys())
    num_nodes_count = len(nodes_list)

    if args.prompting_version_list is not None:
        if len(args.prompting_version_list) != num_nodes_count:
            raise ValueError(
                f"--prompting_version_list length ({len(args.prompting_version_list)}) must match node count ({num_nodes_count})"
            )
        pv_list = args.prompting_version_list
    elif args.prompting_version is not None:
        pv_list = [args.prompting_version] * num_nodes_count
    else:
        if args.domain == "nl":
            parser.error("--prompting_version or --prompting_version_list is required when --domain nl")
        pv_list = [None] * num_nodes_count

    if args.config_fingerprint_list is not None:
        if len(args.config_fingerprint_list) != num_nodes_count:
            raise ValueError(
                f"--config_fingerprint_list length ({len(args.config_fingerprint_list)}) must match node count ({num_nodes_count})"
            )
        cfg_list = [None if c in ("", "none", None) else c for c in args.config_fingerprint_list]
    elif args.config_fingerprint is not None:
        cfg_list = [args.config_fingerprint] * num_nodes_count
    else:
        cfg_list = [None] * num_nodes_count

    print(f"=== Integration Pipeline: {args.query_type} {args.split} ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  resolved seed: {seed}")
    print(f"  resolved dp_values: {dp_values}")
    print(f"  resolved distribution: {distribution}")
    print(f"  resolved pv_list: {pv_list}")
    print(f"  resolved cfg_list: {cfg_list}")
    print()

    per_nodes_paths: List[str] = []

    if args.mode in ("per_nodes", "both"):
        print("--- Per-nodes integration ---")
        for i, num_nodes in enumerate(nodes_list):
            path = integrate_per_nodes(
                args.query_type,
                args.split,
                args.base_dir,
                num_nodes,
                args.run_timestamp,
                num_examples=distribution[num_nodes],
                seed=seed,
                dp_values=dp_values,
                domain=args.domain,
                prompting_version=pv_list[i],
                config_fingerprint=cfg_list[i],
                variant_assignment=args.variant_assignment,
            )
            per_nodes_paths.append(path)
        print()

    if args.mode in ("cross_nodes", "both"):
        if args.mode == "cross_nodes":
            infix = _domain_filename_infix(args.domain)
            for num_nodes in nodes_list:
                pattern = os.path.join(
                    args.base_dir,
                    f"nodes{num_nodes}",
                    f"*-{args.query_type}{infix}-{args.split}-nodes{num_nodes}.jsonl",
                )
                matches = sorted(glob.glob(pattern))
                if not matches:
                    raise FileNotFoundError(f"No per-nodes file found: {pattern}")
                per_nodes_paths.append(matches[-1])

        print("--- Cross-nodes integration ---")
        integrate_cross_nodes(
            args.query_type,
            args.split,
            args.base_dir,
            per_nodes_paths,
            args.run_timestamp,
            domain=args.domain,
        )
        print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
