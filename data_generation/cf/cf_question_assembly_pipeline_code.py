"""
cf_question_assembly_pipeline_code.py

Counterfactual code-domain question assembly pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from data_generation.cf.cf_query_sampling import CounterfactualQueryResult
from data_generation.cf.cf_question_assembly_code import (
    AssembledCodeQuestion,
    assemble_all_questions_code,
)
from data_generation.dataset_utils.code_pipeline_shared import (
    ensure_output_dirs,
    write_outputs,
)
from data_generation.scm_mechanism_sampling import RLCausalSCM


def _deserialize_cpts(cpts_json: Dict) -> Dict:
    cpts = {}
    for node_str, entry in cpts_json.items():
        node = int(node_str)
        cpts[node] = {
            'parents': entry['parents'],
            'table': {ast.literal_eval(key): value for key, value in entry['table'].items()},
        }
    return cpts


def reconstruct_from_triplet(record: Dict):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(record['num_nodes']))
    dag.add_edges_from([tuple(edge) for edge in record['dag_edges']])
    decimal_places = record['decimal_places']
    scm = RLCausalSCM.from_cpts(dag, _deserialize_cpts(record['cpts']), decimal_places=decimal_places)
    treatments = {int(key): value for key, value in record['treatments'].items()}
    outcome = record['outcome']
    evidence = {int(key): value for key, value in record['evidence'].items()}
    cf_result = CounterfactualQueryResult.from_json_dict(record['cf_result'])
    example_seed = record['example_seed']
    return scm, treatments, outcome, evidence, cf_result, decimal_places, example_seed


def load_triplet_corpus(triplet_corpus_path: str) -> List[Dict]:
    with open(triplet_corpus_path, 'r', encoding='utf-8') as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def load_triplet_stats(triplet_corpus_path: str) -> Dict:
    stats_dir = os.path.join(os.path.dirname(os.path.dirname(triplet_corpus_path)), 'stats')
    stats_name = os.path.basename(triplet_corpus_path).replace('.jsonl', '.json')
    with open(os.path.join(stats_dir, stats_name), 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def build_filename(args, triplet_corpus_path: str, start_idx: int, end_idx: int) -> str:
    corpus_basename = os.path.splitext(os.path.basename(triplet_corpus_path))[0]
    return f"{args.run_timestamp}-start{start_idx}-end{end_idx}-from-{corpus_basename}"


def main():
    parser = argparse.ArgumentParser(description='CF code-domain question assembly pipeline')
    parser.add_argument('--triplet_corpus_path', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--run_timestamp', type=str, required=True)
    args = parser.parse_args()

    print('=== CF Code-Domain Question Assembly Pipeline Config ===')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()

    print('Loading triplet corpus...')
    all_records = load_triplet_corpus(args.triplet_corpus_path)
    print(f'  Loaded {len(all_records)} triplet records')
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_records)
    records_to_process = [record for record in all_records if start_idx <= record['idx'] < end_idx]

    num_examples = len(records_to_process)
    print(f'  Processing {num_examples} records (idx range [{start_idx}, {end_idx}))')
    print()

    ensure_output_dirs(args.output_dir)

    print('Assembling code-domain questions...')
    all_variant_records: Dict[str, List[Dict]] = {'full': [], 'surgery_ancestor_prune': []}
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc='Assembling code-domain questions'):
        scm, treatments, outcome, evidence, cf_result, decimal_places, example_seed = reconstruct_from_triplet(record)
        seen_dag_edge_sets.add(tuple(tuple(edge) for edge in record['dag_edges']))
        questions = assemble_all_questions_code(
            scm,
            treatments,
            outcome,
            evidence,
            cf_result,
            decimal_places=decimal_places,
            example_seed=example_seed,
            treatment_outcome_depth=record['treatment_outcome_depth'],
            effective_prob_conds=record['effective_prob_conds'],
            has_naive_conditioning=record['has_naive_conditioning'],
        )
        for question in questions:
            all_variant_records[question.variant_name].append(
                {
                    'idx': record['idx'],
                    'question': question.full_question_text,
                    'answer': question.answer,
                    'metadata': question.metadata,
                }
            )

    filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
    write_outputs(
        args.output_dir,
        args.triplet_corpus_path,
        filename,
        all_variant_records,
        len(seen_dag_edge_sets),
        triplet_stats['num_nodes'],
        triplet_stats['max_abnormal_ratio'],
        start_idx,
        end_idx,
    )

    print()
    print('=== Summary ===')
    print(f'  Total examples: {num_examples}')
    print(f'  Source: {args.triplet_corpus_path}')
    print('=== Done ===')


if __name__ == '__main__':
    main()
