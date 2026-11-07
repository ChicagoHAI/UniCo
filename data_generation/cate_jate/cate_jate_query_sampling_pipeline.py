"""
cate_jate_query_sampling_pipeline.py

Pipeline 1: CATE/JATE query sampling — DAG enumeration, SCM sampling,
effect computation, and triplet corpus generation.

Given hyperparameters (query_type, num_nodes, num_examples, seed, atol,
decimal_places, max_treatments, max_evidence), generates a corpus of
(DAG, SCM, CATEJATEResult) triplets saved as JSONL.

Pipeline phases:
1. Exhaustive DAG enumeration — sample all structurally unique DAGs for
   the given num_nodes.
2. Example distribution — assign DAGs to examples (1:1 if enough unique
   DAGs, otherwise round-robin assignment with different SCM seeds).
3. Example generation — for each example, create SCM, sample a random
   (treatments, outcome, evidence) triplet, compute effect, serialize.

Outputs: triplet JSONL + stats JSON.
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

from tqdm import tqdm

from data_generation.rlcausal_dag_sampling import CausalGraphSampler
from data_generation.scm_mechanism_sampling import RLCausalSCM
from data_generation.cate_jate.cate_jate_query_sampling import (
    compute_cate_jate,
    is_abnormal_triplets,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_treatment_outcome_depth,
    compute_effective_prob_conds,
    has_naive_conditioning,
)


MAX_TRIPLET_RETRIES = 10000


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


def sample_triplet(
    triplet_rng: random.Random,
    nodes: list,
    query_type: str,
    max_treatments: int,
    max_evidence: int,
) -> Tuple[List[int], int, Dict[int, int]]:
    """Sample a (treatments, outcome, evidence) triplet for one example.

    For JATE: num_treatments in [2, max_treatments], no evidence.
    For CATE: 1 treatment, num_evidence in [1, max_evidence].

    Returns:
        (treatments, outcome, evidence) where evidence is a dict
        mapping evidence node IDs to their values (0 or 1).
    """
    if query_type == "jate":
        num_treatments = triplet_rng.randint(2, max_treatments)
        num_evidence = 0
    else:  # cate
        num_treatments = 1
        num_evidence = triplet_rng.randint(1, max_evidence)

    total_needed = num_treatments + 1 + num_evidence
    selected = triplet_rng.sample(nodes, total_needed)

    treatments = sorted(selected[:num_treatments])
    outcome = selected[num_treatments]
    evidence_nodes = selected[num_treatments + 1:]
    evidence = {e: triplet_rng.randint(0, 1) for e in evidence_nodes}

    return treatments, outcome, evidence


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


def _serialize_evidence(evidence: Dict[int, int]) -> Dict[str, int]:
    """Serialize evidence dict to JSON-compatible format.

    Converts integer node IDs to string keys.
    """
    return {str(k): v for k, v in evidence.items()}


def compute_triplet_stats(
    triplet_records: List[Dict],
    num_unique_dags: int,
    args,
) -> Dict:
    """Compute corpus-level statistics for the triplet dataset."""
    num_examples = len(triplet_records)

    abnormal_count = sum(
        1 for r in triplet_records if r["cate_jate_result"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in triplet_records if r["cate_jate_result"]["verification_ok"] is False
    )

    total_edges = sum(r["num_edges"] for r in triplet_records)
    total_treatments = sum(len(r["treatments"]) for r in triplet_records)
    total_evidence = sum(len(r["evidence"]) for r in triplet_records)

    # Difficulty metrics
    total_depth = sum(r["treatment_outcome_depth"] for r in triplet_records)
    total_eff_conds = sum(r["effective_prob_conds"] for r in triplet_records)
    naive_count = sum(1 for r in triplet_records if r["has_naive_conditioning"])

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_edges": total_edges / num_examples,
        "avg_num_treatments": total_treatments / num_examples,
        "avg_num_evidence": total_evidence / num_examples,

        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "verification_false_count": verification_false_count,
        "verification_false_rate": verification_false_count / num_examples,

        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,

        "query_type": args.query_type,
        "num_nodes": args.num_nodes,
        "seed": args.seed,
        "max_unique_dags": args.max_unique_dags,
        "max_abnormal_ratio": args.max_abnormal_ratio,
        "max_treatments": args.max_treatments,
        "max_evidence": args.max_evidence,
        "atol": args.atol,
        "decimal_places": args.decimal_places,
        "cate_relaxed_mode": args.cate_relaxed_mode,
        "jate_relaxed": args.jate_relaxed,
        "max_naive_conditioning_ratio": args.max_naive_conditioning_ratio,
    }


def build_filename(args) -> str:
    """Build the base filename (without extension) from args."""
    parts = (
        f"{args.run_timestamp}"
        f"-{args.query_type}"
        f"-nodes{args.num_nodes}"
        f"-examples{args.num_examples}"
        f"-maxdags{args.max_unique_dags}"
        f"-max_abnormal{args.max_abnormal_ratio}"
        f"-max_treatments{args.max_treatments}"
        f"-max_evidence{args.max_evidence}"
        f"-atol{args.atol}"
        f"-dp{args.decimal_places}"
        f"-seed{args.seed}"
    )
    # Append mode flags only when non-default
    if args.cate_relaxed_mode is not None:
        short = {"ancestor_evidence": "ancestor", "non_dsep_evidence": "nondsep"}[
            args.cate_relaxed_mode
        ]
        parts += f"-crelax_{short}"
    if args.jate_relaxed:
        parts += "-jrelax"
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
        description="CATE/JATE query sampling pipeline — produces triplet corpus"
    )
    parser.add_argument("--query_type", type=str, required=True,
                        choices=["cate", "jate"])
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--decimal_places", type=int, required=True)
    parser.add_argument("--max_abnormal_ratio", type=float, default=1.0)
    parser.add_argument("--max_treatments", type=int, default=2)
    parser.add_argument("--max_evidence", type=int, default=2)
    parser.add_argument("--cate_relaxed_mode", type=str, default=None,
                        choices=["ancestor_evidence", "non_dsep_evidence"],
                        help="CATE relaxed abnormality mode (default: None = strict)")
    parser.add_argument("--jate_relaxed", action="store_true", default=False,
                        help="Use relaxed JATE abnormality checks")
    parser.add_argument("--max_naive_conditioning_ratio", type=float, default=1.0,
                        help="Max ratio of naive conditioning examples (default: 1.0 = no control)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    # Convert argparse bool to Optional[bool] for is_abnormal_triplets
    jate_relaxed_val = True if args.jate_relaxed else None

    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error("--max_naive_conditioning_ratio must be in [0.0, 1.0]")
    if args.max_treatments < 2:
        parser.error("--max_treatments must be >= 2")
    if args.max_evidence < 1:
        parser.error("--max_evidence must be >= 1")

    # Validate num_nodes is sufficient for the query type
    if args.query_type == "jate":
        min_nodes = args.max_treatments + 1
        if args.num_nodes < min_nodes:
            parser.error(
                f"JATE requires num_nodes >= max_treatments + 1 = {min_nodes}, "
                f"got num_nodes={args.num_nodes}"
            )
    else:  # cate
        min_nodes = 1 + 1 + args.max_evidence
        if args.num_nodes < min_nodes:
            parser.error(
                f"CATE requires num_nodes >= 1 + 1 + max_evidence = {min_nodes}, "
                f"got num_nodes={args.num_nodes}"
            )

    # Print config
    print(f"=== CATE/JATE Query Sampling Pipeline Config ({args.query_type.upper()}) ===")
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
    max_retries = MAX_TRIPLET_RETRIES
    if args.max_naive_conditioning_ratio < 1.0:
        max_retries = MAX_TRIPLET_RETRIES * 2
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

        triplet_rng = random.Random(example_seed)
        treatments, outcome, evidence = sample_triplet(
            triplet_rng, scm.nodes, args.query_type,
            args.max_treatments, args.max_evidence,
        )

        # Rejection sampling: reject if query fails any check whose cap is reached
        abnormal_cap_reached = (num_abnormal_so_far >= max_abnormal_count)
        naive_cond_cap_reached = (num_naive_cond_so_far >= max_naive_cond_count)

        if abnormal_cap_reached or naive_cond_cap_reached:
            def _reject_reason(dag, treatments, outcome, evidence):
                """Return rejection reason: 'abnormal', 'naive_cond', or None."""
                if abnormal_cap_reached:
                    ev_nodes = sorted(evidence.keys()) if evidence else None
                    if is_abnormal_triplets(
                        dag, treatments, outcome, ev_nodes,
                        max_treatments=args.max_treatments,
                        max_evidence=args.max_evidence,
                        cate_relaxed_mode=args.cate_relaxed_mode,
                        jate_relaxed=jate_relaxed_val,
                    ):
                        return "abnormal"
                if naive_cond_cap_reached:
                    ev_nodes_list = sorted(evidence.keys()) if evidence else []
                    if has_naive_conditioning(
                        dag, treatments, outcome, ev_nodes_list,
                    ):
                        return "naive_cond"
                return None

            attempt = 0
            num_rejected_abnormal = 0
            num_rejected_naive_cond = 0
            # Track fallback: earliest non-abnormal triplet (aligns with
            # behavior when naive conditioning control is inactive)
            fallback_triplet = None
            reason = _reject_reason(dag, treatments, outcome, evidence)
            if reason == "naive_cond" and fallback_triplet is None:
                fallback_triplet = (treatments, outcome, evidence)
            while reason is not None and attempt < max_retries:
                if reason == "abnormal":
                    num_rejected_abnormal += 1
                else:
                    num_rejected_naive_cond += 1
                treatments, outcome, evidence = sample_triplet(
                    triplet_rng, scm.nodes, args.query_type,
                    args.max_treatments, args.max_evidence,
                )
                attempt += 1
                reason = _reject_reason(dag, treatments, outcome, evidence)
                if reason == "naive_cond" and fallback_triplet is None:
                    fallback_triplet = (treatments, outcome, evidence)
            if attempt >= max_retries and reason is not None:
                # Prefer non-abnormal fallback over the last random sample
                if reason == "abnormal" and fallback_triplet is not None:
                    treatments, outcome, evidence = fallback_triplet
                print(
                    f"  Warning: idx={idx}, could not find acceptable triplet after "
                    f"{max_retries} retries (rejected: {num_rejected_abnormal} abnormal, "
                    f"{num_rejected_naive_cond} naive_cond). Accepting current triplet."
                )

        cate_jate_result = compute_cate_jate(
            scm, treatments, outcome, evidence,
            atol=args.atol,
            cate_relaxed_mode=args.cate_relaxed_mode,
            jate_relaxed=jate_relaxed_val,
        )

        if cate_jate_result.is_abnormal:
            num_abnormal_so_far += 1

        # Difficulty metrics
        treatment_outcome_depth = compute_treatment_outcome_depth(
            dag, treatments, outcome,
        )
        effective_prob_conds = compute_effective_prob_conds(
            dag, treatments, outcome, evidence,
        )
        evidence_nodes = sorted(evidence.keys())
        naive_conditioning = has_naive_conditioning(
            dag, treatments, outcome, evidence_nodes,
        )

        if naive_conditioning:
            num_naive_cond_so_far += 1

        record = {
            "idx": idx,
            "num_nodes": len(scm.nodes),
            "num_edges": dag.number_of_edges(),
            "dag_edges": sorted([list(e) for e in dag.edges()]),
            "cpts": _serialize_cpts(scm),
            "treatments": treatments,
            "outcome": outcome,
            "evidence": _serialize_evidence(evidence),
            "cate_jate_result": {
                "query_type": cate_jate_result.query_type,
                "effect_surgery": cate_jate_result.effect_surgery,
                "effect_causal_query": cate_jate_result.effect_causal_query,
                "is_abnormal": cate_jate_result.is_abnormal,
                "verification_ok": cate_jate_result.verification_ok,
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
        1 for r in triplet_records if r["cate_jate_result"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in triplet_records if r["cate_jate_result"]["verification_ok"] is False
    )
    avg_num_treatments = sum(
        len(r["treatments"]) for r in triplet_records
    ) / len(triplet_records)
    avg_num_evidence = sum(
        len(r["evidence"]) for r in triplet_records
    ) / len(triplet_records)
    print(f"=== Summary ({args.query_type.upper()}) ===")
    print(f"  Total examples: {args.num_examples}")
    print(f"  Unique DAGs: {num_unique}")
    print(f"  Query type: {args.query_type}")
    print(f"  Max abnormal ratio: {args.max_abnormal_ratio}")
    print(f"  Max abnormal count: {max_abnormal_count}")
    print(f"  Abnormal count: {abnormal_count}")
    print(f"  Verification false count: {verification_false_count}")
    print(f"  Avg num treatments: {avg_num_treatments:.2f}")
    print(f"  Avg num evidence: {avg_num_evidence:.2f}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
