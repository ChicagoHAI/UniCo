"""
frontdoor_query_sampling_pipeline.py

Pipeline: enumerate DAGs, construct frontdoor queries, and save graph-only
triplet corpora.
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List

from tqdm import tqdm

from data_generation.dag_cache_utils import (
    load_complete_unique_dags,
    save_complete_unique_dags,
)
from data_generation.ate.ate_query_sampling_pipeline import (
    assign_dags_to_examples,
    ensure_output_dirs,
)
from data_generation.rlcausal_dag_sampling import CausalGraphSampler
from data_generation.dataset_utils.difficulty_metrics import (
    compute_effective_prob_conds,
    compute_treatment_outcome_depth,
    has_naive_conditioning,
)
from data_generation.frontdoor.frontdoor_query_sampling import (
    QUERY_FORMATS,
    build_frontdoor_query,
    is_abnormal_pair,
    validate_frontdoor_set,
)
from data_generation.dataset_utils.graph_only_balance import (
    BINARY_LABELS,
    MULTI_CHOICE_LABELS,
    compute_target_answer_counts,
    count_answer_labels,
    select_most_underfilled_label,
)


def enumerate_unique_dags(
    num_nodes: int,
    seed: int,
    max_unique_dags: int,
):
    cached_dags = load_complete_unique_dags(num_nodes)
    if cached_dags is not None:
        print(
            f"  Loaded {len(cached_dags)} cached complete DAGs for num_nodes={num_nodes}"
        )
        return cached_dags[:max_unique_dags]

    sampler = CausalGraphSampler(num_nodes=num_nodes, max_attempts=10000, seed=seed)
    unique_dags = []
    exhausted = False
    with tqdm(total=max_unique_dags, desc="Sampling unique DAGs") as pbar:
        while len(unique_dags) < max_unique_dags:
            try:
                unique_dags.append(sampler.sample())
                pbar.update(1)
            except RuntimeError:
                exhausted = True
                break

    if exhausted and len(unique_dags) < max_unique_dags:
        cache_path = save_complete_unique_dags(num_nodes, unique_dags)
        print(
            f"  Saved complete DAG cache with {len(unique_dags)} DAGs to {cache_path}"
        )

    return unique_dags


MAX_PAIR_RETRIES = 1000
MAX_FAILED_TARGET_PASSES = 2


def _compute_verification_summary(
    dag,
    treatment: int,
    outcome: int,
    candidate_sets: List[List[int]],
) -> Dict:
    validated_results = [
        validate_frontdoor_set(dag, treatment, outcome, tuple(sorted(candidate_set)))
        for candidate_set in candidate_sets
    ]
    num_valid_sets = sum(result.valid_frontdoor for result in validated_results)
    return {
        "verification_ok": True,
        "verification_mismatch_frontdoor_sets": [],
        "verification_mismatch_details": [],
        "verification_num_mismatches": 0,
        "num_valid_frontdoor_sets": num_valid_sets,
    }


def compute_stats(
    records: List[Dict],
    num_unique_dags: int,
    args,
    *,
    balance_mode_effective: str = "exact",
    relaxed_answer_labels: List[str] | None = None,
    num_fallback_examples: int = 0,
) -> Dict:
    num_examples = len(records)
    total_edges = sum(record["num_edges"] for record in records)
    abnormal_count = sum(record["frontdoor_result"]["is_abnormal"] for record in records)
    total_depth = sum(record["treatment_outcome_depth"] for record in records)
    total_eff = sum(record["effective_prob_conds"] for record in records)
    naive_count = sum(1 for record in records if record["has_naive_conditioning"])
    total_num_correct = sum(
        sum(option["valid_frontdoor"] for option in record["frontdoor_result"]["option_results"])
        for record in records
    )
    verification_false_count = sum(
        1 for record in records if record["frontdoor_result"]["verification_ok"] is False
    )
    stats = {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_edges": total_edges / num_examples,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,
        "avg_num_correct_options": total_num_correct / num_examples,
        "verification_false_count": verification_false_count,
        "verification_false_rate": verification_false_count / num_examples,
        "num_nodes": args.num_nodes,
        "seed": args.seed,
        "max_unique_dags": args.max_unique_dags,
        "max_abnormal_ratio": args.max_abnormal_ratio,
        "max_naive_conditioning_ratio": args.max_naive_conditioning_ratio,
        "query_format": args.query_format,
    }
    if args.query_format == "multi_choice":
        stats["balance_mode_requested"] = "exact"
        stats["balance_mode_effective"] = balance_mode_effective
        stats["relaxed_answer_labels"] = sorted(relaxed_answer_labels or [])
        stats["num_fallback_examples"] = num_fallback_examples
    if args.query_format in {"binary", "multi_choice"}:
        answer_label_counts = count_answer_labels(records, "frontdoor_result")
        stats["answer_label_counts"] = answer_label_counts
        stats["answer_label_rates"] = {
            label: count / num_examples for label, count in answer_label_counts.items()
        }
    return stats


def build_filename(args) -> str:
    parts = (
        f"{args.run_timestamp}"
        f"-nodes{args.num_nodes}"
        f"-examples{args.num_examples}"
        f"-maxdags{args.max_unique_dags}"
        f"-max_abnormal{args.max_abnormal_ratio}"
        f"-fmt{args.query_format}"
        f"-seed{args.seed}"
    )
    if args.max_naive_conditioning_ratio < 1.0:
        parts += f"-max_naive_cond{args.max_naive_conditioning_ratio}"
    return parts


def write_outputs(
    args,
    filename: str,
    records: List[Dict],
    num_unique_dags: int,
    *,
    balance_mode_effective: str = "exact",
    relaxed_answer_labels: List[str] | None = None,
    num_fallback_examples: int = 0,
) -> None:
    jsonl_path = os.path.join(args.output_dir, "triplets", "jsonls", f"{filename}.jsonl")
    stats_path = os.path.join(args.output_dir, "triplets", "stats", f"{filename}.json")

    with open(jsonl_path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            compute_stats(
                records,
                num_unique_dags,
                args,
                balance_mode_effective=balance_mode_effective,
                relaxed_answer_labels=relaxed_answer_labels,
                num_fallback_examples=num_fallback_examples,
            ),
            file_obj,
            indent=2,
            ensure_ascii=False,
        )
        file_obj.write("\n")

    print(f"  JSONL: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Frontdoor query sampling pipeline")
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--query_format", type=str, choices=QUERY_FORMATS, default="multi_answer")
    parser.add_argument("--max_abnormal_ratio", type=float, default=1.0)
    parser.add_argument(
        "--max_naive_conditioning_ratio",
        type=float,
        default=1.0,
        help="Max ratio of naive conditioning examples (default: 1.0 = no control)",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    if args.query_format == "multi_choice" and args.num_nodes < 3:
        parser.error(
            "--num_nodes must be at least 3 for multi_choice frontdoor queries because they require 2 distinct candidate frontdoor sets"
        )
    if args.query_format == "multi_answer" and args.num_nodes < 4:
        parser.error(
            "--num_nodes must be at least 4 for multi_answer frontdoor queries because they require 4 distinct candidate frontdoor-set options"
        )
    if args.query_format == "binary" and args.num_nodes < 2:
        parser.error("--num_nodes must be at least 2 for binary frontdoor queries")
    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error("--max_naive_conditioning_ratio must be in [0.0, 1.0]")

    print("=== Frontdoor Query Sampling Pipeline Config ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    ensure_output_dirs(args.output_dir)

    print("Phase 1: Enumerating unique DAGs...")
    unique_dags = enumerate_unique_dags(args.num_nodes, args.seed, args.max_unique_dags)
    num_unique = len(unique_dags)
    print(f"  Sampled {num_unique} unique DAGs for num_nodes={args.num_nodes}")
    if num_unique == 0:
        raise RuntimeError(
            f"No valid DAGs found for num_nodes={args.num_nodes}. "
            "Check CausalGraphSampler constraints."
        )
    print()

    print("Phase 2: Preparing DAG traversal order...")
    dag_traversal = assign_dags_to_examples(unique_dags, max(args.num_examples, num_unique), args.seed)
    print(f"  Prepared {len(dag_traversal)} DAG slots for traversal")
    print()

    print("Phase 3: Generating frontdoor triplets...")
    max_abnormal_count = math.floor(args.max_abnormal_ratio * args.num_examples)
    max_naive_cond_count = math.floor(args.max_naive_conditioning_ratio * args.num_examples)
    num_abnormal_so_far = 0
    num_naive_cond_so_far = 0
    max_retries = MAX_PAIR_RETRIES
    if args.max_naive_conditioning_ratio < 1.0:
        max_retries = MAX_PAIR_RETRIES * 2
    print(f"  Max abnormal count: {max_abnormal_count} (ratio={args.max_abnormal_ratio})")
    print(
        f"  Max naive conditioning count: {max_naive_cond_count} "
        f"(ratio={args.max_naive_conditioning_ratio})"
    )
    print(f"  Max retries per example: {max_retries}")

    target_answer_counts = None
    current_answer_counts = None
    balance_mode_effective = "exact"
    relaxed_answer_labels = set()
    num_fallback_examples = 0
    failed_target_passes: Dict[str, int] = {}
    fallback_allow_any_label = False
    if args.query_format == "binary":
        target_answer_counts = compute_target_answer_counts(BINARY_LABELS, args.num_examples)
        current_answer_counts = {label: 0 for label in BINARY_LABELS}
        print(f"  Target binary answer counts: {target_answer_counts}")
    elif args.query_format == "multi_choice":
        target_answer_counts = compute_target_answer_counts(MULTI_CHOICE_LABELS, args.num_examples)
        current_answer_counts = {label: 0 for label in MULTI_CHOICE_LABELS}
        failed_target_passes = {label: 0 for label in MULTI_CHOICE_LABELS}
        print(f"  Target multi-choice answer counts: {target_answer_counts}")
        print(f"  Max failed passes before relaxing a target label: {MAX_FAILED_TARGET_PASSES}")

    records: List[Dict] = []
    traversal_pass = 0
    with tqdm(total=args.num_examples, desc="Generating triplets") as progress_bar:
        while len(records) < args.num_examples:
            progress_this_pass = False
            requested_label = None
            primary_label = None
            pass_uses_fallback = False
            if current_answer_counts is not None:
                primary_label = select_most_underfilled_label(target_answer_counts, current_answer_counts)
                if primary_label is None:
                    break
                if args.query_format == "multi_choice":
                    underfilled_unrelaxed = [
                        label
                        for label in MULTI_CHOICE_LABELS
                        if current_answer_counts[label] < target_answer_counts[label]
                        and label not in relaxed_answer_labels
                    ]
                    if fallback_allow_any_label:
                        requested_label = None
                        pass_uses_fallback = True
                    elif primary_label in relaxed_answer_labels:
                        if underfilled_unrelaxed:
                            requested_label = underfilled_unrelaxed[0]
                            pass_uses_fallback = requested_label != primary_label
                        else:
                            requested_label = None
                            fallback_allow_any_label = True
                            pass_uses_fallback = True
                    else:
                        requested_label = primary_label
                else:
                    requested_label = primary_label

            for dag_offset in range(len(dag_traversal)):
                if len(records) >= args.num_examples:
                    break
                if (
                    requested_label is not None
                    and current_answer_counts is not None
                    and current_answer_counts[requested_label] >= target_answer_counts[requested_label]
                ):
                    break

                idx = len(records)
                dag = dag_traversal[(traversal_pass + dag_offset) % len(dag_traversal)]
                example_seed = args.seed + traversal_pass * len(dag_traversal) + dag_offset
                pair_rng = random.Random(example_seed)
                dag_nodes = list(dag.nodes())

                abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
                naive_cond_cap_reached = num_naive_cond_so_far >= max_naive_cond_count

                frontdoor_result = None
                verification_summary = None
                treatment = None
                outcome = None
                for _ in range(max_retries):
                    candidate_treatment, candidate_outcome = pair_rng.sample(dag_nodes, 2)
                    candidate_naive = has_naive_conditioning(
                        dag,
                        [candidate_treatment],
                        candidate_outcome,
                        evidence_nodes=[],
                    )
                    candidate_abnormal = is_abnormal_pair(dag, candidate_treatment, candidate_outcome)
                    if abnormal_cap_reached and candidate_abnormal:
                        continue
                    if naive_cond_cap_reached and candidate_naive:
                        continue

                    option_order_rng = random.Random(
                        example_seed + 1000 + candidate_treatment * 100 + candidate_outcome
                    )
                    candidate_result = build_frontdoor_query(
                        dag,
                        candidate_treatment,
                        candidate_outcome,
                        query_format=args.query_format,
                        option_order_rng=option_order_rng,
                        target_answer_label=requested_label,
                    )
                    if candidate_result is None:
                        continue

                    treatment = candidate_treatment
                    outcome = candidate_outcome
                    frontdoor_result = candidate_result
                    verification_summary = _compute_verification_summary(
                        dag,
                        treatment,
                        outcome,
                        [option.frontdoor_variables for option in frontdoor_result.option_results],
                    )
                    break

                if frontdoor_result is None or verification_summary is None or treatment is None or outcome is None:
                    continue

                if frontdoor_result.is_abnormal:
                    num_abnormal_so_far += 1
                if current_answer_counts is not None:
                    current_answer_counts[frontdoor_result.correct_options] += 1
                record_has_naive = has_naive_conditioning(
                    dag,
                    [treatment],
                    outcome,
                    evidence_nodes=[],
                )
                if record_has_naive:
                    num_naive_cond_so_far += 1

                if args.query_format == "multi_choice":
                    if pass_uses_fallback or requested_label is None:
                        num_fallback_examples += 1
                        balance_mode_effective = "soft_fallback"
                    if requested_label is not None:
                        failed_target_passes[requested_label] = 0
                    failed_target_passes[frontdoor_result.correct_options] = 0

                frontdoor_result_payload = frontdoor_result.to_json_dict()
                frontdoor_result_payload.update(verification_summary)
                records.append(
                    {
                        "idx": idx,
                        "num_nodes": args.num_nodes,
                        "num_edges": dag.number_of_edges(),
                        "dag_edges": sorted(list(dag.edges())),
                        "query_type": "frontdoor",
                        "query_format": args.query_format,
                        "frontdoor_result": frontdoor_result_payload,
                        "treatment_outcome_depth": compute_treatment_outcome_depth(dag, [treatment], outcome),
                        "effective_prob_conds": compute_effective_prob_conds(dag, [treatment], outcome, {}),
                        "has_naive_conditioning": record_has_naive,
                        "example_seed": example_seed,
                    }
                )
                progress_this_pass = True
                progress_bar.update(1)
                break

            if len(records) >= args.num_examples:
                break
            if not progress_this_pass:
                if args.query_format == "multi_choice":
                    should_retry = False
                    if primary_label is not None:
                        failed_target_passes[primary_label] += 1
                        if failed_target_passes[primary_label] >= MAX_FAILED_TARGET_PASSES:
                            relaxed_answer_labels.add(primary_label)
                            balance_mode_effective = "soft_fallback"
                        should_retry = True
                    if requested_label is not None and requested_label != primary_label:
                        should_retry = True
                    if not fallback_allow_any_label:
                        underfilled_unrelaxed = [
                            label
                            for label in MULTI_CHOICE_LABELS
                            if current_answer_counts[label] < target_answer_counts[label]
                            and label not in relaxed_answer_labels
                        ]
                        if not underfilled_unrelaxed:
                            fallback_allow_any_label = True
                            balance_mode_effective = "soft_fallback"
                            should_retry = True
                    if should_retry:
                        traversal_pass += 1
                        continue
                raise RuntimeError(
                    "Could not construct any additional frontdoor queries to reach num_examples "
                    f"under query_format={args.query_format}. Generated {len(records)} / {args.num_examples}. "
                    "This indicates total query infeasibility under the current node count and quota constraints, "
                    "not just exact-balance infeasibility."
                )
            traversal_pass += 1

    filename = build_filename(args)
    write_outputs(
        args,
        filename,
        records,
        num_unique,
        balance_mode_effective=balance_mode_effective,
        relaxed_answer_labels=sorted(relaxed_answer_labels),
        num_fallback_examples=num_fallback_examples,
    )
    print("=== Done ===")


if __name__ == "__main__":
    main()
