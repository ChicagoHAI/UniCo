"""
ate_query_sampling_pipeline.py

Pipeline 1: end-to-end ATE query sampling — DAG enumeration, SCM sampling,
ATE computation, and triplet corpus generation.

Given hyperparameters (num_nodes, num_examples, seed, atol, decimal_places),
generates a corpus of (DAG, SCM, ATEResult) triplets saved as JSONL.

Pipeline phases:
1. Exhaustive DAG enumeration — sample all structurally unique DAGs for
   the given num_nodes.
2. Example distribution — assign DAGs to examples (1:1 if enough unique
   DAGs, otherwise round-robin assignment with different SCM seeds).
3. Example generation — for each example, create SCM, pick a random
   (treatment, outcome) pair, compute ATE, serialize triplet.

Outputs: triplet JSONL + stats JSON.
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List

from tqdm import tqdm

from data_generation.rlcausal_dag_sampling import CausalGraphSampler
from data_generation.scm_mechanism_sampling import RLCausalSCM
from data_generation.ate.ate_query_sampling import (
    compute_ate,
    is_abnormal_pair,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_treatment_outcome_depth,
    compute_effective_prob_conds,
    has_naive_conditioning,
)


MAX_PAIR_RETRIES = 10000


def ensure_output_dirs(output_dir: str) -> None:
    """Create output directory structure for triplet corpus.

    Creates jsonls/ and stats/ subdirectories under triplets/.
    The .logs/ directory is created by the shell runner (needed by tee
    before Python starts).
    """
    for subdir in ("jsonls", "stats"):
        os.makedirs(
            os.path.join(output_dir, "triplets", subdir), exist_ok=True
        )


def enumerate_unique_dags(
    num_nodes: int, seed: int, max_unique_dags: int = 1000
) -> list:
    """Sample structurally unique DAGs for the given num_nodes.

    Uses a generous max_attempts (10000) to maximize coverage. Stops
    when max_unique_dags is reached or RuntimeError is caught (no more
    unique DAGs found within max_attempts consecutive tries).
    """
    sampler = CausalGraphSampler(
        num_nodes=num_nodes, max_attempts=10000, seed=seed
    )
    unique_dags = []
    with tqdm(total=max_unique_dags, desc="Sampling unique DAGs") as pbar:
        while len(unique_dags) < max_unique_dags:
            try:
                unique_dags.append(sampler.sample())
                pbar.update(1)
            except RuntimeError:
                break
    return unique_dags


def assign_dags_to_examples(
    unique_dags: list, num_examples: int, seed: int
) -> list:
    """Assign DAGs to examples.

    If num_unique >= num_examples: uniformly select num_examples DAGs
    without replacement.
    If num_unique < num_examples: round-robin assignment so any
    contiguous slice of examples gets maximum DAG diversity.
    """
    num_unique = len(unique_dags)

    if num_unique >= num_examples:
        dag_selector = random.Random(seed)
        return dag_selector.sample(unique_dags, num_examples)

    # Round-robin: cycle through unique DAGs
    return [unique_dags[idx % num_unique] for idx in range(num_examples)]


def _serialize_cpts(scm: RLCausalSCM) -> Dict:
    """Serialize SCM CPTs to a JSON-compatible dict.

    Mirrors the in-memory scm.cpts format. Tuple keys are converted to
    their string representation for JSON compatibility (e.g., (0, 1) →
    "(0, 1)"). Node IDs become string keys (JSON requirement).
    """
    result = {}
    for node in scm.nodes:
        parents = scm.cpts[node]['parents']
        table = scm.cpts[node]['table']
        str_table = {str(k): v for k, v in table.items()}
        result[str(node)] = {"parents": parents, "table": str_table}
    return result


def compute_triplet_stats(
    triplet_records: List[Dict],
    num_unique_dags: int,
    args,
) -> Dict:
    """Compute corpus-level statistics for the triplet dataset."""
    num_examples = len(triplet_records)

    abnormal_count = sum(
        1 for r in triplet_records if r["ate_result"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in triplet_records if r["ate_result"]["verification_ok"] is False
    )

    total_edges = sum(r["num_edges"] for r in triplet_records)

    # Difficulty metrics
    total_depth = sum(r["treatment_outcome_depth"] for r in triplet_records)
    total_eff_conds = sum(r["effective_prob_conds"] for r in triplet_records)
    naive_count = sum(1 for r in triplet_records if r["has_naive_conditioning"])

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_edges": total_edges / num_examples,

        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "verification_false_count": verification_false_count,
        "verification_false_rate": verification_false_count / num_examples,

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
    """Build the base filename (without extension) from args."""
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
    # Append naive conditioning ratio only when non-default
    if args.max_naive_conditioning_ratio < 1.0:
        parts += f"-max_naive_cond{args.max_naive_conditioning_ratio}"
    return parts


def write_outputs(
    args,
    filename: str,
    triplet_records: List[Dict],
    num_unique_dags: int,
) -> None:
    """Write triplet JSONL and stats JSON."""
    # Paths
    jsonl_path = os.path.join(
        args.output_dir, "triplets", "jsonls", f"{filename}.jsonl"
    )
    stats_path = os.path.join(
        args.output_dir, "triplets", "stats", f"{filename}.json"
    )

    # JSONL
    with open(jsonl_path, "w") as f:
        for record in triplet_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Stats JSON
    stats = compute_triplet_stats(triplet_records, num_unique_dags, args)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  JSONL: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ATE query sampling pipeline — produces triplet corpus"
    )
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--decimal_places", type=int, required=True)
    parser.add_argument("--max_abnormal_ratio", type=float, default=1.0)
    parser.add_argument("--max_naive_conditioning_ratio", type=float, default=1.0,
                        help="Max ratio of naive conditioning examples (default: 1.0 = no control)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error("--max_naive_conditioning_ratio must be in [0.0, 1.0]")

    # Print config
    print("=== ATE Query Sampling Pipeline Config ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    # Create output directories
    ensure_output_dirs(args.output_dir)

    # Phase 1: Exhaustive DAG enumeration
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
            f"Examples will be distributed across DAGs with different SCM seeds."
        )
    print()

    # Phase 2: Assign DAGs to examples
    print("Phase 2: Assigning DAGs to examples...")
    dag_assignment = assign_dags_to_examples(
        unique_dags, args.num_examples, args.seed
    )
    print(f"  Assigned {len(dag_assignment)} DAGs to examples")
    print()

    # Phase 3: Generate examples
    print("Phase 3: Generating triplet corpus...")
    max_abnormal_count = math.floor(args.max_abnormal_ratio * args.num_examples)
    num_abnormal_so_far = 0
    max_naive_cond_count = math.floor(
        args.max_naive_conditioning_ratio * args.num_examples
    )
    num_naive_cond_so_far = 0
    # Scale up retries when naive conditioning control is active
    max_retries = MAX_PAIR_RETRIES
    if args.max_naive_conditioning_ratio < 1.0:
        max_retries = MAX_PAIR_RETRIES * 2
    print(f"  Max abnormal count: {max_abnormal_count} "
          f"(ratio={args.max_abnormal_ratio})")
    print(f"  Max naive conditioning count: {max_naive_cond_count} "
          f"(ratio={args.max_naive_conditioning_ratio})")
    print(f"  Max retries per example: {max_retries}")

    triplet_records: List[Dict] = []

    for idx in tqdm(range(args.num_examples), desc="Generating triplets"):
        dag = dag_assignment[idx]
        example_seed = args.seed + idx

        scm = RLCausalSCM(dag, seed=example_seed, decimal_places=args.decimal_places)

        pair_rng = random.Random(example_seed)
        treatment, outcome = pair_rng.sample(scm.nodes, 2)

        # Rejection sampling: reject if query fails any check whose cap is reached
        abnormal_cap_reached = (num_abnormal_so_far >= max_abnormal_count)
        naive_cond_cap_reached = (num_naive_cond_so_far >= max_naive_cond_count)

        if abnormal_cap_reached or naive_cond_cap_reached:
            def _reject_reason(dag, t, y):
                """Return rejection reason: 'abnormal', 'naive_cond', or None."""
                if abnormal_cap_reached and is_abnormal_pair(dag, t, y):
                    return "abnormal"
                if naive_cond_cap_reached and has_naive_conditioning(
                    dag, [t], y, evidence_nodes=[],
                ):
                    return "naive_cond"
                return None

            attempt = 0
            num_rejected_abnormal = 0
            num_rejected_naive_cond = 0
            # Track fallback: earliest non-abnormal pair (aligns with
            # behavior when naive conditioning control is inactive)
            fallback_treatment, fallback_outcome = None, None
            reason = _reject_reason(dag, treatment, outcome)
            if reason == "naive_cond" and fallback_treatment is None:
                fallback_treatment, fallback_outcome = treatment, outcome
            while reason is not None and attempt < max_retries:
                if reason == "abnormal":
                    num_rejected_abnormal += 1
                else:
                    num_rejected_naive_cond += 1
                treatment, outcome = pair_rng.sample(scm.nodes, 2)
                attempt += 1
                reason = _reject_reason(dag, treatment, outcome)
                if reason == "naive_cond" and fallback_treatment is None:
                    fallback_treatment, fallback_outcome = treatment, outcome
            if attempt >= max_retries and reason is not None:
                # Prefer non-abnormal fallback over the last random sample
                if reason == "abnormal" and fallback_treatment is not None:
                    treatment, outcome = fallback_treatment, fallback_outcome
                print(
                    f"  Warning: idx={idx}, could not find acceptable pair after "
                    f"{max_retries} retries (rejected: {num_rejected_abnormal} abnormal, "
                    f"{num_rejected_naive_cond} naive_cond). Accepting current pair."
                )

        ate_result = compute_ate(
            scm, treatment, outcome,
            atol=args.atol,
        )

        if ate_result.is_abnormal:
            num_abnormal_so_far += 1

        # Difficulty metrics
        treatment_outcome_depth = compute_treatment_outcome_depth(
            dag, [treatment], outcome,
        )
        effective_prob_conds = compute_effective_prob_conds(
            dag, [treatment], outcome, evidence={},
        )
        naive_conditioning = has_naive_conditioning(
            dag, [treatment], outcome, evidence_nodes=[],
        )

        if naive_conditioning:
            num_naive_cond_so_far += 1

        record = {
            "idx": idx,
            "num_nodes": len(scm.nodes),
            "num_edges": dag.number_of_edges(),
            "dag_edges": sorted([list(e) for e in dag.edges()]),
            "cpts": _serialize_cpts(scm),
            "treatment": treatment,
            "outcome": outcome,
            "ate_result": {
                "ate_surgery": ate_result.ate_surgery,
                "ate_causal_query": ate_result.ate_causal_query,
                "is_abnormal": ate_result.is_abnormal,
                "verification_ok": ate_result.verification_ok,
            },
            "treatment_outcome_depth": treatment_outcome_depth,
            "effective_prob_conds": effective_prob_conds,
            "has_naive_conditioning": naive_conditioning,
            "example_seed": example_seed,
            "decimal_places": args.decimal_places,
        }
        triplet_records.append(record)

    print()

    # Write outputs
    print("Writing outputs...")
    filename = build_filename(args)
    write_outputs(args, filename, triplet_records, num_unique)
    print()

    # Summary
    abnormal_count = sum(
        1 for r in triplet_records if r["ate_result"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in triplet_records if r["ate_result"]["verification_ok"] is False
    )
    print("=== Summary ===")
    print(f"  Total examples: {args.num_examples}")
    print(f"  Unique DAGs: {num_unique}")
    print(f"  Max abnormal ratio: {args.max_abnormal_ratio}")
    print(f"  Max abnormal count: {max_abnormal_count}")
    print(f"  Abnormal count: {abnormal_count}")
    print(f"  Verification false count: {verification_false_count}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
