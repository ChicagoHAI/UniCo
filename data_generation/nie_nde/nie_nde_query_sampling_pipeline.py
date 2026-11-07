"""
nie_nde_query_sampling_pipeline.py

Triplet corpus generation for Natural Indirect Effect (NIE) and
Natural Direct Effect (NDE) queries.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import random
from typing import Dict, List, Tuple

import networkx as nx
from tqdm import tqdm

from data_generation.ate.ate_query_sampling_pipeline import (
    _serialize_cpts,
    ensure_output_dirs,
)
from data_generation.dag_cache_utils import (
    load_complete_unique_dags,
    save_complete_unique_dags,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_effective_prob_conds,
    compute_treatment_outcome_depth,
)
from data_generation.nie_nde.nie_nde_query_sampling import (
    compute_nie_nde,
    is_abnormal,
    is_invalid_condition,
)
from data_generation.rlcausal_dag_sampling import CausalGraphSampler
from data_generation.scm_mechanism_sampling import RLCausalSCM


def has_naive_conditioning(
    dag: nx.DiGraph,
    treatment: int,
    mediator: int,
    outcome: int,
) -> bool:
    """Return True when there is no mediator-outcome confounding given treatment.

    We operationalize mediator-outcome confounding via backdoor paths only: remove
    outgoing edges from the mediator so the direct causal path M->Y does not count,
    then test whether treatment d-separates mediator and outcome.
    """
    graph = dag.copy()
    graph.remove_edges_from(list(graph.out_edges(mediator)))
    return nx.is_d_separator(graph, {mediator}, {outcome}, {treatment})


def _has_valid_triplet(dag: nx.DiGraph) -> bool:
    nodes = list(dag.nodes())
    return any(
        not is_invalid_condition(dag, treatment, mediator, outcome)
        for treatment in nodes
        for mediator in nodes
        for outcome in nodes
    )


def enumerate_unique_dags(
    num_nodes: int,
    seed: int,
    max_unique_dags: int,
) -> List[nx.DiGraph]:
    cached_dags = load_complete_unique_dags(num_nodes)
    if cached_dags is not None and len(cached_dags) < max_unique_dags:
        print(
            f"  Loaded {len(cached_dags)} cached complete DAGs for num_nodes={num_nodes}"
        )
        unique_valid_dags = [dag for dag in cached_dags if _has_valid_triplet(dag)]
        return unique_valid_dags

    sampler = CausalGraphSampler(num_nodes=num_nodes, max_attempts=10000, seed=seed)
    unique_dags = []
    unique_valid_dags = []
    exhausted = False
    with tqdm(total=max_unique_dags, desc="Sampling unique DAGs") as pbar:
        while len(unique_dags) < max_unique_dags:
            try:
                dag = sampler.sample()
            except RuntimeError:
                exhausted = True
                break
            unique_dags.append(dag)
            if _has_valid_triplet(dag):
                unique_valid_dags.append(dag)
                pbar.update(1)

    if exhausted and len(unique_dags) < max_unique_dags:
        cache_path = save_complete_unique_dags(num_nodes, unique_dags)
        print(
            f"  Saved complete DAG cache with {len(unique_dags)} DAGs to {cache_path}"
        )

    return unique_valid_dags


def _is_abnormal_only_dag(
    dag: nx.DiGraph,
    valid_triplets: List[Tuple[int, int, int]],
) -> bool:
    return all(is_abnormal(dag, treatment, mediator, outcome) for treatment, mediator, outcome in valid_triplets)


def assign_dags_to_examples(
    unique_dags: List[nx.DiGraph],
    dag_to_valid_triplets: Dict[int, List[Tuple[int, int, int]]],
    num_examples: int,
    seed: int,
    max_abnormal_ratio: float,
) -> List[nx.DiGraph]:
    rng = random.Random(seed)

    abnormal_only_dags = [
        dag for dag in unique_dags
        if _is_abnormal_only_dag(dag, dag_to_valid_triplets[id(dag)])
    ]
    rng.shuffle(abnormal_only_dags)
    mixed_dags = [
        dag for dag in unique_dags
        if not dag in abnormal_only_dags
    ]
    rng.shuffle(mixed_dags)

    max_abnormal_only_examples = max(0, math.floor(max_abnormal_ratio * num_examples) - 1)
    if len(mixed_dags) == 0 and num_examples > 0:
        raise RuntimeError(
            "All candidate DAGs are abnormal-only; cannot keep the assigned abnormal-only DAG ratio below max_abnormal_ratio."
        )

    num_mixed_assigned = min(max(len(mixed_dags), num_examples-max_abnormal_only_examples), num_examples)
    num_abnormal_only_assigned = num_examples - num_mixed_assigned
    
    assigned_dags: List[nx.DiGraph] = []
    
    if num_abnormal_only_assigned > 0:
        if len(abnormal_only_dags) >= num_abnormal_only_assigned:
            assigned_dags.extend(rng.sample(abnormal_only_dags, num_abnormal_only_assigned))
        else:
            assigned_dags.extend(
                abnormal_only_dags[idx % len(abnormal_only_dags)]
                for idx in range(num_abnormal_only_assigned)
            )

    if mixed_dags:
        if len(mixed_dags) >= num_mixed_assigned:
            assigned_dags.extend(rng.sample(mixed_dags, num_mixed_assigned))
        else:
            assigned_dags.extend(mixed_dags[idx % len(mixed_dags)] for idx in range(num_mixed_assigned))

    return assigned_dags


def _enumerate_valid_triplets(
    dag: nx.DiGraph,
) -> List[Tuple[int, int, int]]:
    nodes = list(dag.nodes())
    return [
        (treatment, mediator, outcome)
        for treatment in nodes
        for mediator in nodes
        for outcome in nodes
        if not is_invalid_condition(dag, treatment, mediator, outcome)
    ]


def _sample_triplet(
    valid_triplets: List[Tuple[int, int, int]],
    rng: random.Random,
) -> Tuple[int, int, int]:
    if not valid_triplets:
        raise RuntimeError("No valid (treatment, mediator, outcome) triplets in DAG")
    return rng.choice(valid_triplets)


MAX_TRIPLET_RETRIES = 1000
ZERO_ANSWER_ATOL = 0.00005


def _is_zero_answer(value: float) -> bool:
    return abs(value) < ZERO_ANSWER_ATOL


def compute_stats(
    records: List[Dict],
    num_unique_dags: int,
    args,
) -> Dict:
    num_examples = len(records)
    abnormal_count = sum(
        1 for record in records if record["nie_nde_result"]["is_abnormal"]
    )
    total_edges = sum(record["num_edges"] for record in records)
    total_depth = sum(record["treatment_outcome_depth"] for record in records)
    total_eff_conds = sum(record["effective_prob_conds"] for record in records)
    naive_count = sum(1 for record in records if record["has_naive_conditioning"])
    nie_zero_count = sum(
        1
        for record in records
        if abs(record["nie_nde_result"]["nie_value"]) < ZERO_ANSWER_ATOL
    )
    nde_zero_count = sum(
        1
        for record in records
        if abs(record["nie_nde_result"]["nde_value"]) < ZERO_ANSWER_ATOL
    )

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_edges": total_edges / num_examples,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_count": naive_count,
        "naive_conditioning_rate": naive_count / num_examples,
        "nie_zero_answer_count": nie_zero_count,
        "nie_zero_answer_rate": nie_zero_count / num_examples,
        "nde_zero_answer_count": nde_zero_count,
        "nde_zero_answer_rate": nde_zero_count / num_examples,
        "zero_answer_atol": ZERO_ANSWER_ATOL,
        "num_nodes": args.num_nodes,
        "seed": args.seed,
        "max_unique_dags": args.max_unique_dags,
        "max_abnormal_ratio": args.max_abnormal_ratio,
        "max_naive_conditioning_ratio": args.max_naive_conditioning_ratio,
        "atol": args.atol,
        "decimal_places": args.decimal_places,
    }


def build_filename(args) -> str:
    return (
        f"{args.run_timestamp}"
        f"-nodes{args.num_nodes}"
        f"-examples{args.num_examples}"
        f"-maxdags{args.max_unique_dags}"
        f"-atol{args.atol}"
        f"-dp{args.decimal_places}"
        f"-seed{args.seed}"
    )


def write_outputs(
    args,
    filename: str,
    records: List[Dict],
    num_unique_dags: int,
) -> None:
    jsonl_path = os.path.join(
        args.output_dir, "triplets", "jsonls", f"{filename}.jsonl"
    )
    stats_path = os.path.join(
        args.output_dir, "triplets", "stats", f"{filename}.json"
    )

    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = compute_stats(records, num_unique_dags, args)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"  JSONL: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NIE/NDE query sampling pipeline"
    )
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--decimal_places", type=int, required=True)
    parser.add_argument("--max_abnormal_ratio", type=float, default=1.0)
    parser.add_argument("--max_naive_conditioning_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error("--max_naive_conditioning_ratio must be in [0.0, 1.0]")

    print("=== NIE/NDE Query Sampling Pipeline Config ===")
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
            "Examples will reuse DAGs with different SCM seeds."
        )
    print()

    print("Phase 1.5: Enumerating valid triplets for sampled DAGs...")
    dag_to_valid_triplets = {}
    for dag in tqdm(unique_dags, desc="Enumerating valid triplets"):
        valid_triplets = _enumerate_valid_triplets(dag)
        if not valid_triplets:
            raise RuntimeError("Sampled DAG unexpectedly has no valid NIE/NDE triplets")
        dag_to_valid_triplets[id(dag)] = valid_triplets
    num_abnormal_only_dags = sum(
        1 for dag in unique_dags if _is_abnormal_only_dag(dag, dag_to_valid_triplets[id(dag)])
    )
    print(f"  Cached valid triplets for {len(dag_to_valid_triplets)} sampled DAGs")
    print(f"  Abnormal-only DAGs: {num_abnormal_only_dags} / {len(unique_dags)}")
    print()

    print("Phase 2: Assigning DAGs to examples...")
    dag_assignment = assign_dags_to_examples(
        unique_dags,
        dag_to_valid_triplets,
        args.num_examples,
        args.seed,
        args.max_abnormal_ratio,
    )
    num_abnormal_only_assigned = sum(
        1 for dag in dag_assignment if _is_abnormal_only_dag(dag, dag_to_valid_triplets[id(dag)])
    )
    print(f"  Assigned {len(dag_assignment)} DAGs to examples")
    print(
        f"  Assigned abnormal-only DAGs: {num_abnormal_only_assigned} / {len(dag_assignment)}"
    )
    print()

    print("Phase 3: Generating triplet corpus...")
    max_abnormal_count = math.floor(
        args.max_abnormal_ratio * args.num_examples
    )
    max_naive_cond_count = math.floor(
        args.max_naive_conditioning_ratio * args.num_examples
    )
    num_abnormal_so_far = 0
    num_naive_cond_so_far = 0
    num_nie_zero_so_far = 0
    num_nde_zero_so_far = 0
    print(
        f"  Max abnormal count: {max_abnormal_count} "
        f"(ratio={args.max_abnormal_ratio})"
    )
    print(
        f"  Max naive conditioning count: {max_naive_cond_count} "
        f"(ratio={args.max_naive_conditioning_ratio})"
    )

    records: List[Dict] = []

    for idx in tqdm(range(args.num_examples), desc="Generating triplets"):
        dag = dag_assignment[idx]
        example_seed = args.seed + idx
        rng = random.Random(example_seed)
        scm = RLCausalSCM(dag, seed=example_seed, decimal_places=args.decimal_places)

        valid_triplets = dag_to_valid_triplets[id(dag)]

        try:
            treatment, mediator, outcome = _sample_triplet(valid_triplets, rng)
        except RuntimeError as exc:
            raise RuntimeError(f"idx={idx}: {exc}") from exc

        abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
        naive_cap_reached = num_naive_cond_so_far >= max_naive_cond_count

        if abnormal_cap_reached or naive_cap_reached:
            def _reject_reason(
                candidate_abnormal: bool,
                candidate_naive: bool,
            ) -> str | None:
                if abnormal_cap_reached and candidate_abnormal:
                    return "abnormal"
                if naive_cap_reached and candidate_naive:
                    return "naive_cond"
                return None

            attempt = 0
            num_rejected_abnormal = 0
            num_rejected_naive_cond = 0
            # Track fallback: earliest non-abnormal triplet (aligns with
            # behavior when naive conditioning control is inactive)
            fallback_triplet = None
            current_result = compute_nie_nde(scm, treatment, mediator, outcome)
            current_naive = has_naive_conditioning(dag, treatment, mediator, outcome)
            reason = _reject_reason(
                current_result.is_abnormal,
                current_naive,
            )
            if reason == "naive_cond" and fallback_triplet is None:
                fallback_triplet = (treatment, mediator, outcome)
            while reason is not None and attempt < MAX_TRIPLET_RETRIES:
                if reason == "abnormal":
                    num_rejected_abnormal += 1
                else:
                    num_rejected_naive_cond += 1
                candidate_treatment, candidate_mediator, candidate_outcome = _sample_triplet(valid_triplets, rng)
                attempt += 1
                candidate_result = compute_nie_nde(
                    scm, candidate_treatment, candidate_mediator, candidate_outcome
                )
                candidate_naive = has_naive_conditioning(
                    dag, candidate_treatment, candidate_mediator, candidate_outcome
                )
                reason = _reject_reason(
                    candidate_result.is_abnormal,
                    candidate_naive,
                )
                if reason == "naive_cond" and fallback_triplet is None:
                    fallback_triplet = (
                        candidate_treatment,
                        candidate_mediator,
                        candidate_outcome,
                    )
                treatment = candidate_treatment
                mediator = candidate_mediator
                outcome = candidate_outcome
            if attempt >= MAX_TRIPLET_RETRIES and reason is not None:
                # Prefer non-abnormal fallback over the last random sample
                if reason == "abnormal" and fallback_triplet is not None:
                    treatment, mediator, outcome = fallback_triplet
                print(
                    f"  Warning: idx={idx}, could not find acceptable triplet after "
                    f"{MAX_TRIPLET_RETRIES} retries (rejected: {num_rejected_abnormal} abnormal, "
                    f"{num_rejected_naive_cond} naive_cond). Accepting current triplet."
                )

        nie_nde_result = compute_nie_nde(scm, treatment, mediator, outcome)
        if is_invalid_condition(dag, treatment, mediator, outcome):
            raise RuntimeError(
                f"idx={idx}: sampled invalid NIE/NDE triplet, which should be impossible"
            )

        treatment_outcome_depth = compute_treatment_outcome_depth(
            dag, [treatment], outcome
        )
        effective_prob_conds = compute_effective_prob_conds(
            dag, [treatment], outcome, {mediator: 1}
        )
        naive_conditioning = has_naive_conditioning(
            dag, treatment, mediator, outcome
        )
        if nie_nde_result.is_abnormal:
            num_abnormal_so_far += 1
        if naive_conditioning:
            num_naive_cond_so_far += 1
        if _is_zero_answer(nie_nde_result.nie_value):
            num_nie_zero_so_far += 1
        if _is_zero_answer(nie_nde_result.nde_value):
            num_nde_zero_so_far += 1

        records.append(
            {
                "idx": idx,
                "num_nodes": len(scm.nodes),
                "num_edges": dag.number_of_edges(),
                "dag_edges": sorted([list(edge) for edge in dag.edges()]),
                "cpts": _serialize_cpts(scm),
                "treatment": treatment,
                "mediator": mediator,
                "outcome": outcome,
                "nie_nde_result": nie_nde_result.to_json_dict(),
                "treatment_outcome_depth": treatment_outcome_depth,
                "effective_prob_conds": effective_prob_conds,
                "has_naive_conditioning": naive_conditioning,
                "example_seed": example_seed,
                "decimal_places": args.decimal_places,
            }
        )

    print()
    print("Writing outputs...")
    filename = build_filename(args)
    write_outputs(args, filename, records, num_unique)
    print()

    print("=== Summary ===")
    print(f"  Total examples: {args.num_examples}")
    print(f"  Unique DAGs: {num_unique}")
    print("  Source query type: NIE/NDE")
    print("=== Done ===")


if __name__ == "__main__":
    main()
