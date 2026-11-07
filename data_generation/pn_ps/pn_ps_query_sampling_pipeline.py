"""
pn_ps_query_sampling_pipeline.py

Pipeline: enumerate DAGs, sample SCMs, compute PN/PS, and save triplet corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

from tqdm import tqdm

from data_generation.ate.ate_query_sampling_pipeline import (
    _serialize_cpts,
    assign_dags_to_examples,
    ensure_output_dirs,
    enumerate_unique_dags,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_counterfactual_effective_prob_conds,
    compute_treatment_outcome_depth,
)
from data_generation.cf.cf_query_sampling import (
    has_naive_conditioning_counterfactual,
    is_abnormal_counterfactual,
)
from data_generation.pn_ps.pn_ps_query_sampling import compute_pn_ps
from data_generation.scm_mechanism_sampling import RLCausalSCM


MAX_PAIR_RETRIES = 10000


def compute_stats(
    records: List[Dict],
    num_unique_dags: int,
    args,
) -> Dict:
    num_examples = len(records)
    abnormal_count = sum(r["pn_ps_result"]["is_abnormal"] for r in records)
    total_edges = sum(r["num_edges"] for r in records)
    total_depth = sum(r["treatment_outcome_depth"] for r in records)
    total_eff_conds = sum(r["effective_prob_conds"] for r in records)
    normalization_fail_count_pn = sum(
        1
        for r in records
        if r["pn_ps_result"]["cf_result_pn"]["normalization_ok"] is False
    )
    normalization_fail_count_ps = sum(
        1
        for r in records
        if r["pn_ps_result"]["cf_result_ps"]["normalization_ok"] is False
    )
    naive_count = sum(1 for r in records if r["has_naive_conditioning"])

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_edges": total_edges / num_examples,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "normalization_fail_count_pn": normalization_fail_count_pn,
        "normalization_fail_rate_pn": normalization_fail_count_pn / num_examples,
        "normalization_fail_count_ps": normalization_fail_count_ps,
        "normalization_fail_rate_ps": normalization_fail_count_ps / num_examples,
        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,
        "num_nodes": args.num_nodes,
        "seed": args.seed,
        "max_unique_dags": args.max_unique_dags,
        "max_abnormal_ratio": args.max_abnormal_ratio,
        "max_naive_conditioning_ratio": args.max_naive_conditioning_ratio,
        "atol": args.atol,
        "decimal_places": args.decimal_places,
    }


def build_filename(args) -> str:
    parts = (
        f"{args.run_timestamp}"
        f"-nodes{args.num_nodes}"
        f"-examples{args.num_examples}"
        f"-maxdags{args.max_unique_dags}"
        f"-max_abnormal{args.max_abnormal_ratio}"
        f"-atol{args.atol}"
        f"-dp{args.decimal_places}"
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

    stats = compute_stats(records, num_unique_dags, args)
    with open(stats_path, "w", encoding="utf-8") as file_obj:
        json.dump(stats, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    print(f"  JSONL: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="PN/PS query sampling pipeline")
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--decimal_places", type=int, required=True)
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

    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error("--max_naive_conditioning_ratio must be in [0.0, 1.0]")

    print("=== PN/PS Query Sampling Pipeline Config ===")
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
    if num_unique < args.num_examples:
        print(
            f"  Note: {num_unique} unique DAGs < {args.num_examples} examples. "
            "Examples will be distributed across DAGs with different SCM seeds."
        )
    print()

    print("Phase 2: Assigning DAGs to examples...")
    dag_assignment = assign_dags_to_examples(unique_dags, args.num_examples, args.seed)
    print(f"  Assigned {len(dag_assignment)} DAGs to examples")
    print()

    print("Phase 3: Generating PN/PS triplets...")
    max_abnormal_count = math.floor(args.max_abnormal_ratio * args.num_examples)
    max_naive_count = math.floor(args.max_naive_conditioning_ratio * args.num_examples)
    num_abnormal_so_far = 0
    num_naive_so_far = 0
    print(f"  Max abnormal count: {max_abnormal_count} (ratio={args.max_abnormal_ratio})")
    print(f"  Max naive conditioning count: {max_naive_count} (ratio={args.max_naive_conditioning_ratio})")

    records: List[Dict] = []
    for idx in tqdm(range(args.num_examples), desc="Generating triplets"):
        dag = dag_assignment[idx]
        example_seed = args.seed + idx
        scm = RLCausalSCM(dag, seed=example_seed, decimal_places=args.decimal_places)
        pair_rng = random.Random(example_seed)

        treatment, outcome = pair_rng.sample(scm.nodes, 2)
        abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
        naive_cap_reached = num_naive_so_far >= max_naive_count
        attempt = 0
        while attempt < MAX_PAIR_RETRIES:
            candidate_is_abnormal = is_abnormal_counterfactual(
                dag,
                {treatment: 1},
                outcome,
            )
            candidate_is_naive = has_naive_conditioning_counterfactual(
                dag,
                {treatment: 1},
                {treatment: 1, outcome: 1},
            )
            if abnormal_cap_reached and candidate_is_abnormal:
                treatment, outcome = pair_rng.sample(scm.nodes, 2)
                attempt += 1
                continue
            if naive_cap_reached and candidate_is_naive:
                treatment, outcome = pair_rng.sample(scm.nodes, 2)
                attempt += 1
                continue
            break
        if attempt >= MAX_PAIR_RETRIES:
            print(
                f"  Warning: idx={idx}, could not find acceptable treatment/outcome "
                f"pair after {MAX_PAIR_RETRIES} retries. Accepting final sampled pair."
            )

        pn_ps_result = compute_pn_ps(
            scm,
            treatment,
            outcome,
            atol=args.atol,
        )
        if pn_ps_result.is_abnormal:
            num_abnormal_so_far += 1

        treatment_outcome_depth = compute_treatment_outcome_depth(
            dag,
            [treatment],
            outcome,
        )
        effective_prob_conds = compute_counterfactual_effective_prob_conds(
            dag,
            [treatment],
            outcome,
            evidence={treatment: 1, outcome: 1},
        )
        has_naive = has_naive_conditioning_counterfactual(
            dag,
            {treatment: 1},
            {treatment: 1, outcome: 1},
        )
        if has_naive:
            num_naive_so_far += 1

        record = {
            "idx": idx,
            "num_nodes": len(scm.nodes),
            "num_edges": dag.number_of_edges(),
            "dag_edges": sorted([list(e) for e in dag.edges()]),
            "cpts": _serialize_cpts(scm),
            "treatment": treatment,
            "outcome": outcome,
            "pn_ps_result": pn_ps_result.to_json_dict(),
            "treatment_outcome_depth": treatment_outcome_depth,
            "effective_prob_conds": effective_prob_conds,
            "has_naive_conditioning": has_naive,
            "example_seed": example_seed,
            "decimal_places": args.decimal_places,
        }
        records.append(record)

    print()
    print("Writing outputs...")
    filename = build_filename(args)
    write_outputs(args, filename, records, num_unique)
    print()

    abnormal_count = sum(1 for record in records if record["pn_ps_result"]["is_abnormal"])
    print("=== Summary ===")
    print(f"  Total examples: {args.num_examples}")
    print(f"  Unique DAGs: {num_unique}")
    print(f"  Abnormal count: {abnormal_count}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
