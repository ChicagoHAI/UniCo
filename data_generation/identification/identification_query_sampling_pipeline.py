"""
identification_query_sampling_pipeline.py

Pipeline: enumerate DAGs, construct identification queries, and save graph-only
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
from data_generation.dataset_utils.graph_only_balance import (
    BINARY_LABELS,
    MULTI_CHOICE_LABELS,
    compute_target_answer_counts,
    count_answer_labels,
    select_most_underfilled_label,
)
from data_generation.identification.identification_query_sampling import (
    QUERY_FORMATS,
    build_identification_query,
    identify_observed_set_y0,
    identify_observed_set_pgmpy,
    is_abnormal_pair,
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


def _compute_verification_summary(
    dag,
    treatment: int,
    outcome: int,
    observed_sets: List[List[int]],
) -> Dict:
    mismatch_details = []
    for observed_variables in observed_sets:
        y0_result = identify_observed_set_y0(
            dag,
            treatment,
            outcome,
            observed_variables,
        )
        pgmpy_result = identify_observed_set_pgmpy(
            dag,
            treatment,
            outcome,
            observed_variables,
        )
        if y0_result.identifiable != pgmpy_result.identifiable:
            mismatch_details.append(
                {
                    "observed_variables": list(observed_variables),
                    "y0_identifiable": "Yes" if y0_result.identifiable else "No",
                    "pgmpy_identifiable": "Yes" if pgmpy_result.identifiable else "No",
                }
            )

    mismatch_details = sorted(
        mismatch_details,
        key=lambda item: item["observed_variables"],
    )
    return {
        "verification_ok": len(mismatch_details) == 0,
        "verification_mismatch_observed_sets": [
            item["observed_variables"] for item in mismatch_details
        ],
        "verification_mismatch_details": mismatch_details,
        "verification_num_mismatches": len(mismatch_details),
    }


def compute_stats(records: List[Dict], num_unique_dags: int, args) -> Dict:
    num_examples = len(records)
    total_edges = sum(record["num_edges"] for record in records)
    abnormal_count = sum(record["identification_result"]["is_abnormal"] for record in records)
    total_depth = sum(record["treatment_outcome_depth"] for record in records)
    total_eff = sum(record["effective_prob_conds"] for record in records)
    naive_count = sum(1 for record in records if record["has_naive_conditioning"])
    total_num_correct = sum(
        sum(option["identifiable"] for option in record["identification_result"]["option_results"])
        for record in records
    )
    verification_false_count = sum(
        1 for record in records if record["identification_result"]["verification_ok"] is False
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
    if args.query_format in {"binary", "multi_choice"}:
        answer_label_counts = count_answer_labels(records, "identification_result")
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


def write_outputs(args, filename: str, records: List[Dict], num_unique_dags: int) -> None:
    jsonl_path = os.path.join(args.output_dir, "triplets", "jsonls", f"{filename}.jsonl")
    stats_path = os.path.join(args.output_dir, "triplets", "stats", f"{filename}.json")

    with open(jsonl_path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as file_obj:
        json.dump(compute_stats(records, num_unique_dags, args), file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    print(f"  JSONL: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Identification query sampling pipeline")
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
            "--num_nodes must be at least 3 for multi_choice identification queries because they require 2 distinct observed sets"
        )
    if args.query_format == "multi_answer" and args.num_nodes < 4:
        parser.error(
            "--num_nodes must be at least 4 for multi_answer identification queries because they require 4 distinct observed-set options"
        )
    if args.query_format == "binary" and args.num_nodes < 2:
        parser.error("--num_nodes must be at least 2 for binary identification queries")
    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error("--max_naive_conditioning_ratio must be in [0.0, 1.0]")

    print("=== Identification Query Sampling Pipeline Config ===")
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

    print("Phase 3: Generating identification triplets...")
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
    if args.query_format == "binary":
        target_answer_counts = compute_target_answer_counts(BINARY_LABELS, args.num_examples)
        current_answer_counts = {label: 0 for label in BINARY_LABELS}
        print(f"  Target binary answer counts: {target_answer_counts}")
    elif args.query_format == "multi_choice":
        target_answer_counts = compute_target_answer_counts(MULTI_CHOICE_LABELS, args.num_examples)
        current_answer_counts = {label: 0 for label in MULTI_CHOICE_LABELS}
        print(f"  Target multi-choice answer counts: {target_answer_counts}")

    records: List[Dict] = []
    traversal_pass = 0
    with tqdm(total=args.num_examples, desc="Generating triplets") as progress_bar:
        while len(records) < args.num_examples:
            progress_this_pass = False
            target_answer_label = None
            if current_answer_counts is not None:
                target_answer_label = select_most_underfilled_label(target_answer_counts, current_answer_counts)
                if target_answer_label is None:
                    break

            for dag_offset in range(len(dag_traversal)):
                if len(records) >= args.num_examples:
                    break
                if current_answer_counts is not None and current_answer_counts[target_answer_label] >= target_answer_counts[target_answer_label]:
                    break

                idx = len(records)
                dag = dag_traversal[(traversal_pass + dag_offset) % len(dag_traversal)]
                example_seed = args.seed + traversal_pass * len(dag_traversal) + dag_offset
                pair_rng = random.Random(example_seed)
                dag_nodes = list(dag.nodes())

                abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
                naive_cond_cap_reached = num_naive_cond_so_far >= max_naive_cond_count

                identification_result = None
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

                    candidate_result = build_identification_query(
                        dag,
                        candidate_treatment,
                        candidate_outcome,
                        query_format=args.query_format,
                        option_order_rng=pair_rng,
                        target_answer_label=target_answer_label,
                        identify_observed_set_fn=identify_observed_set_y0,
                    )
                    if candidate_result is None:
                        continue

                    treatment = candidate_treatment
                    outcome = candidate_outcome
                    identification_result = candidate_result
                    verification_summary = _compute_verification_summary(
                        dag,
                        treatment,
                        outcome,
                        [
                            option.observed_variables
                            for option in identification_result.option_results
                        ],
                    )
                    break

                if identification_result is None or verification_summary is None or treatment is None or outcome is None:
                    continue

                treatment_outcome_depth = compute_treatment_outcome_depth(dag, [treatment], outcome)
                effective_prob_conds = compute_effective_prob_conds(dag, [treatment], outcome, {})
                naive_conditioning = has_naive_conditioning(dag, [treatment], outcome, [])

                if identification_result.is_abnormal:
                    num_abnormal_so_far += 1
                if naive_conditioning:
                    num_naive_cond_so_far += 1
                if current_answer_counts is not None:
                    current_answer_counts[identification_result.correct_options] += 1

                identification_result_payload = identification_result.to_json_dict()
                identification_result_payload.update(verification_summary)

                record = {
                    "idx": idx,
                    "num_nodes": dag.number_of_nodes(),
                    "num_edges": dag.number_of_edges(),
                    "dag_edges": sorted([list(edge) for edge in dag.edges()]),
                    "treatment": treatment,
                    "outcome": outcome,
                    "query_format": args.query_format,
                    "identification_result": identification_result_payload,
                    "treatment_outcome_depth": treatment_outcome_depth,
                    "effective_prob_conds": effective_prob_conds,
                    "has_naive_conditioning": naive_conditioning,
                    "example_seed": example_seed,
                }
                records.append(record)
                progress_this_pass = True
                progress_bar.update(1)

            if len(records) >= args.num_examples:
                break
            if not progress_this_pass:
                raise RuntimeError(
                    "Could not construct enough identification queries to reach num_examples "
                    f"under query_format={args.query_format}. Generated {len(records)} / {args.num_examples}."
                )
            traversal_pass += 1

    print()
    print("Writing outputs...")
    filename = build_filename(args)
    write_outputs(args, filename, records, num_unique)
    print("=== Done ===")


if __name__ == "__main__":
    main()
