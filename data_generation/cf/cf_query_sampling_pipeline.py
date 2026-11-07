"""
cf_query_sampling_pipeline.py

Pipeline 1: counterfactual query sampling — DAG enumeration, SCM sampling,
counterfactual computation, and triplet corpus generation.

Given hyperparameters (num_nodes, num_examples, seed, atol, decimal_places,
max_treatments, max_evidence), generates a corpus of
(DAG, SCM, CounterfactualQueryResult) triplets saved as JSONL.

Pipeline phases:
1. Exhaustive DAG enumeration — sample all structurally unique DAGs for
   the given num_nodes.
2. Example distribution — assign DAGs to examples (1:1 if enough unique
   DAGs, otherwise round-robin assignment with different SCM seeds).
3. Example generation — for each example, create SCM, sample a random
   (treatments, outcome, evidence) triplet, compute the counterfactual,
   and serialize the triplet.

Outputs: triplet JSONL + stats JSON.
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

from tqdm import tqdm

from data_generation.ate.ate_query_sampling_pipeline import (
    _serialize_cpts,
    assign_dags_to_examples,
    ensure_output_dirs,
    enumerate_unique_dags,
)
from data_generation.cf.cf_query_sampling import (
    compute_counterfactual_result,
    has_naive_conditioning_counterfactual,
    is_abnormal_counterfactual,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_counterfactual_effective_prob_conds,
    compute_treatment_outcome_depth,
)
from data_generation.scm_mechanism_sampling import RLCausalSCM


MAX_SPEC_RETRIES = 100
MAX_TRIPLET_RETRIES = 10000


def _sample_cf_spec(
    rng: random.Random,
    nodes: List[int],
    max_treatments: int,
    max_evidence: int,
) -> Tuple[Dict[int, int], int, Dict[int, int]]:
    if len(nodes) < 2:
        raise ValueError(
            'Counterfactual queries require at least 2 nodes (1 outcome + 1 non-outcome).'
        )
    if max_evidence < 1:
        raise ValueError('max_evidence must be >= 1 when evidence is mandatory')

    max_treatments_allowed = min(max_treatments, len(nodes) - 1)
    if max_treatments_allowed < 1:
        raise ValueError('max_treatments leaves no room for an outcome variable')

    for _ in range(MAX_SPEC_RETRIES):
        num_treatments = rng.randint(1, max_treatments_allowed)
        do_nodes = rng.sample(nodes, num_treatments)
        available_for_outcome = [node for node in nodes if node not in do_nodes]
        if not available_for_outcome:
            continue

        outcome = rng.choice(available_for_outcome)
        evidence_candidates = [
            node for node in nodes if node not in do_nodes and node != outcome
        ]
        max_evidence_allowed = min(max_evidence, len(evidence_candidates))
        if max_evidence_allowed < 1:
            continue

        num_evidence = rng.randint(1, max_evidence_allowed)
        evidence_nodes = rng.sample(evidence_candidates, num_evidence)

        treatments = {node: rng.choice([0, 1]) for node in do_nodes}
        evidence = {node: rng.choice([0, 1]) for node in evidence_nodes}
        return treatments, outcome, evidence

    raise RuntimeError(
        'Could not sample a valid counterfactual specification with mandatory evidence within retries'
    )


def compute_triplet_stats(
    triplet_records: List[Dict],
    num_unique_dags: int,
    args,
) -> Dict:
    """Compute corpus-level statistics for the triplet dataset."""
    num_examples = len(triplet_records)

    abnormal_count = sum(
        1 for record in triplet_records if record['cf_result']['is_abnormal']
    )
    normalization_false_count = sum(
        1 for record in triplet_records if record['cf_result']['normalization_ok'] is False
    )
    total_edges = sum(record['num_edges'] for record in triplet_records)
    total_do = sum(len(record['treatments']) for record in triplet_records)
    total_evidence = sum(len(record['evidence']) for record in triplet_records)
    total_depth = sum(record['treatment_outcome_depth'] for record in triplet_records)
    total_eff_conds = sum(record['effective_prob_conds'] for record in triplet_records)
    naive_count = sum(1 for record in triplet_records if record['has_naive_conditioning'])

    return {
        'num_examples': num_examples,
        'num_unique_dags': num_unique_dags,
        'avg_num_edges': total_edges / num_examples,
        'avg_num_treatments': total_do / num_examples,
        'avg_evidence_size': total_evidence / num_examples,
        'abnormal_count': abnormal_count,
        'abnormal_rate': abnormal_count / num_examples,
        'normalization_false_count': normalization_false_count,
        'normalization_false_rate': normalization_false_count / num_examples,
        'avg_treatment_outcome_depth': total_depth / num_examples,
        'avg_effective_prob_conds': total_eff_conds / num_examples,
        'naive_conditioning_rate': naive_count / num_examples,
        'num_nodes': args.num_nodes,
        'seed': args.seed,
        'max_unique_dags': args.max_unique_dags,
        'max_abnormal_ratio': args.max_abnormal_ratio,
        'max_naive_conditioning_ratio': args.max_naive_conditioning_ratio,
        'decimal_places': args.decimal_places,
        'max_treatments': args.max_treatments,
        'max_evidence': args.max_evidence,
        'atol': args.atol,
    }


def build_filename(args) -> str:
    """Build the base filename (without extension) from args."""
    parts = (
        f'{args.run_timestamp}'
        f'-nodes{args.num_nodes}'
        f'-examples{args.num_examples}'
        f'-maxdags{args.max_unique_dags}'
        f'-max_abnormal{args.max_abnormal_ratio}'
        f'-max_treatments{args.max_treatments}'
        f'-max_evidence{args.max_evidence}'
        f'-atol{args.atol}'
        f'-dp{args.decimal_places}'
        f'-seed{args.seed}'
    )
    if args.max_naive_conditioning_ratio < 1.0:
        parts += f'-max_naive_cond{args.max_naive_conditioning_ratio}'
    return parts


def write_outputs(
    args,
    filename: str,
    triplet_records: List[Dict],
    num_unique_dags: int,
) -> None:
    """Write triplet JSONL and stats JSON."""
    jsonl_path = os.path.join(args.output_dir, 'triplets', 'jsonls', f'{filename}.jsonl')
    stats_path = os.path.join(args.output_dir, 'triplets', 'stats', f'{filename}.json')

    with open(jsonl_path, 'w', encoding='utf-8') as file_obj:
        for record in triplet_records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + '\n')

    stats = compute_triplet_stats(triplet_records, num_unique_dags, args)
    with open(stats_path, 'w', encoding='utf-8') as file_obj:
        json.dump(stats, file_obj, indent=2, ensure_ascii=False)
        file_obj.write('\n')

    print(f'  JSONL: {jsonl_path}')
    print(f'  Stats: {stats_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Counterfactual query sampling pipeline — produces triplet corpus'
    )
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--num_examples', type=int, required=True)
    parser.add_argument('--max_unique_dags', type=int, default=1000)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--atol', type=float, required=True)
    parser.add_argument('--decimal_places', type=int, required=True)
    parser.add_argument('--max_abnormal_ratio', type=float, default=1.0)
    parser.add_argument('--max_naive_conditioning_ratio', type=float, default=1.0)
    parser.add_argument('--max_treatments', type=int, default=1)
    parser.add_argument('--max_evidence', type=int, default=2)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--run_timestamp', type=str, required=True)
    args = parser.parse_args()

    if args.num_nodes < 2:
        parser.error('--num_nodes must be >= 2 for counterfactual queries')
    if args.max_treatments < 1:
        parser.error('--max_treatments must be >= 1')
    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error('--max_abnormal_ratio must be in [0.0, 1.0]')
    if not (0.0 <= args.max_naive_conditioning_ratio <= 1.0):
        parser.error('--max_naive_conditioning_ratio must be in [0.0, 1.0]')

    print('=== CF Query Sampling Pipeline Config ===')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()

    ensure_output_dirs(args.output_dir)

    print('Phase 1: Enumerating unique DAGs...')
    unique_dags = enumerate_unique_dags(args.num_nodes, args.seed, args.max_unique_dags)
    num_unique = len(unique_dags)
    print(f'  Sampled {num_unique} unique DAGs for num_nodes={args.num_nodes}')

    if num_unique == 0:
        raise RuntimeError(
            f'No valid DAGs found for num_nodes={args.num_nodes}. '
            'Check CausalGraphSampler constraints.'
        )

    if num_unique < args.num_examples:
        print(
            f'  Note: {num_unique} unique DAGs < {args.num_examples} examples. '
            'Examples will be distributed across DAGs with different SCM seeds.'
        )
    print()

    print('Phase 2: Assigning DAGs to examples...')
    dag_assignment = assign_dags_to_examples(unique_dags, args.num_examples, args.seed)
    print(f'  Assigned {len(dag_assignment)} DAGs to examples')
    print()

    print('Phase 3: Generating triplet corpus...')
    max_abnormal_count = math.floor(args.max_abnormal_ratio * args.num_examples)
    num_abnormal_so_far = 0
    max_naive_cond_count = math.floor(
        args.max_naive_conditioning_ratio * args.num_examples
    )
    num_naive_cond_so_far = 0
    max_retries = MAX_TRIPLET_RETRIES
    if args.max_naive_conditioning_ratio < 1.0:
        max_retries = MAX_TRIPLET_RETRIES * 2
    print(f'  Max abnormal count: {max_abnormal_count} (ratio={args.max_abnormal_ratio})')
    print(
        f'  Max naive conditioning count: {max_naive_cond_count} '
        f'(ratio={args.max_naive_conditioning_ratio})'
    )
    print(f'  Max retries per example: {max_retries}')

    triplet_records: List[Dict] = []

    for idx in tqdm(range(args.num_examples), desc='Generating triplets'):
        dag = dag_assignment[idx]
        example_seed = args.seed + idx
        scm = RLCausalSCM(dag, seed=example_seed, decimal_places=args.decimal_places)
        triplet_rng = random.Random(example_seed)

        treatments, outcome, evidence = _sample_cf_spec(
            triplet_rng,
            scm.nodes,
            args.max_treatments,
            args.max_evidence,
        )

        abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
        naive_cond_cap_reached = num_naive_cond_so_far >= max_naive_cond_count

        if abnormal_cap_reached or naive_cond_cap_reached:
            def _reject_reason(dag, treatments, outcome, evidence):
                if abnormal_cap_reached and is_abnormal_counterfactual(dag, treatments, outcome):
                    return 'abnormal'
                if naive_cond_cap_reached and has_naive_conditioning_counterfactual(dag, treatments, evidence):
                    return 'naive_cond'
                return None

            attempt = 0
            num_rejected_abnormal = 0
            num_rejected_naive_cond = 0
            fallback_spec = None
            reason = _reject_reason(dag, treatments, outcome, evidence)
            if reason == 'naive_cond' and fallback_spec is None:
                fallback_spec = (treatments, outcome, evidence)

            while reason is not None and attempt < max_retries:
                if reason == 'abnormal':
                    num_rejected_abnormal += 1
                else:
                    num_rejected_naive_cond += 1
                treatments, outcome, evidence = _sample_cf_spec(
                    triplet_rng,
                    scm.nodes,
                    args.max_treatments,
                    args.max_evidence,
                )
                attempt += 1
                reason = _reject_reason(dag, treatments, outcome, evidence)
                if reason == 'naive_cond' and fallback_spec is None:
                    fallback_spec = (treatments, outcome, evidence)

            if attempt >= max_retries and reason is not None:
                if reason == 'abnormal' and fallback_spec is not None:
                    treatments, outcome, evidence = fallback_spec
                print(
                    f'  Warning: idx={idx}, could not find acceptable spec after '
                    f'{max_retries} retries (rejected: {num_rejected_abnormal} abnormal, '
                    f'{num_rejected_naive_cond} naive_cond). Accepting current spec.'
                )

        cf_result = compute_counterfactual_result(
            scm,
            treatments=treatments,
            outcome=outcome,
            evidence=evidence,
            atol=args.atol,
        )
        is_abnormal = is_abnormal_counterfactual(dag, treatments, outcome)
        if is_abnormal:
            num_abnormal_so_far += 1

        treatment_nodes = sorted(treatments.keys())
        treatment_outcome_depth = compute_treatment_outcome_depth(
            dag,
            treatment_nodes,
            outcome,
        )
        effective_prob_conds = compute_counterfactual_effective_prob_conds(
            dag,
            treatment_nodes,
            outcome,
            evidence,
        )
        naive_conditioning = has_naive_conditioning_counterfactual(
            dag,
            treatments,
            evidence,
        )
        if naive_conditioning:
            num_naive_cond_so_far += 1

        record = {
            'idx': idx,
            'num_nodes': len(scm.nodes),
            'num_edges': dag.number_of_edges(),
            'dag_edges': sorted([list(edge) for edge in dag.edges()]),
            'cpts': _serialize_cpts(scm),
            'treatments': {str(k): v for k, v in treatments.items()},
            'outcome': outcome,
            'evidence': {str(k): v for k, v in evidence.items()},
            'cf_result': {
                **cf_result.to_json_dict(),
                'is_abnormal': is_abnormal,
            },
            'treatment_outcome_depth': treatment_outcome_depth,
            'effective_prob_conds': effective_prob_conds,
            'has_naive_conditioning': naive_conditioning,
            'example_seed': example_seed,
            'decimal_places': args.decimal_places,
        }
        triplet_records.append(record)

    print()
    print('Writing outputs...')
    filename = build_filename(args)
    write_outputs(args, filename, triplet_records, num_unique)
    print()

    abnormal_count = sum(1 for record in triplet_records if record['cf_result']['is_abnormal'])
    normalization_false_count = sum(
        1 for record in triplet_records if record['cf_result']['normalization_ok'] is False
    )
    print('=== Summary ===')
    print(f'  Total examples: {args.num_examples}')
    print(f'  Unique DAGs: {num_unique}')
    print(f'  Abnormal count: {abnormal_count}')
    print(f'  Normalization failures: {normalization_false_count}')
    print('=== Done ===')


if __name__ == '__main__':
    main()
