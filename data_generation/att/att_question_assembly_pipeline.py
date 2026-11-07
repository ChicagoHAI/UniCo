"""
att_question_assembly_pipeline.py

Loads ATT triplets and assembles textual questions for each variant.
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from data_generation.ate.ate_question_assembly_pipeline import (
    _deserialize_cpts,
    build_filename,
    ensure_output_dirs,
    load_triplet_corpus,
    load_triplet_stats,
)
from data_generation.att.att_query_sampling import ATTResult
from data_generation.att.att_question_assembly import (
    AssembledQuestion,
    assemble_all_questions,
)
from data_generation.scm_mechanism_sampling import RLCausalSCM


VARIANT_NAMES = ['full', 'surgery_ancestor_prune']


def reconstruct_from_triplet(record: Dict):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(record['num_nodes']))
    dag.add_edges_from([tuple(edge) for edge in record['dag_edges']])

    cpts = _deserialize_cpts(record['cpts'])
    decimal_places = record['decimal_places']
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=decimal_places)
    att_result = ATTResult.from_json_dict(record['att_result'])
    example_seed = record['example_seed']

    return scm, att_result, decimal_places, example_seed


def compute_variant_stats(
    variant_records: List[Dict],
    num_unique_dags: int,
    num_nodes: int,
    scms: List[RLCausalSCM],
    questions: List[AssembledQuestion],
    max_abnormal_ratio: float,
    source_triplet_corpus: str,
    start_idx: int,
    end_idx: int,
) -> Dict:
    num_examples = len(variant_records)
    if num_examples == 0:
        return {
            'num_examples': 0,
            'num_unique_dags': num_unique_dags,
            'avg_num_nodes': num_nodes,
            'abnormal_count': 0,
            'abnormal_rate': 0.0,
            'normalization_fail_count_do1': 0,
            'normalization_fail_rate_do1': 0.0,
            'normalization_fail_count_do0': 0,
            'normalization_fail_rate_do0': 0.0,
            'avg_num_cpts': 0.0,
            'avg_num_cpts_pa0': 0.0,
            'avg_num_cpts_pa1': 0.0,
            'avg_num_cpts_pa2': 0.0,
            'avg_treatment_outcome_depth': 0.0,
            'avg_effective_prob_conds': 0.0,
            'naive_conditioning_rate': 0.0,
            'graph_variant_counts': {1: 0, 2: 0, 3: 0},
            'graph_variant_rates': {1: 0.0, 2: 0.0, 3: 0.0},
            'query_variant_counts': {1: 0, 2: 0, 3: 0},
            'query_variant_rates': {1: 0.0, 2: 0.0, 3: 0.0},
            'max_abnormal_ratio': max_abnormal_ratio,
            'source_triplet_corpus': source_triplet_corpus,
            'start_idx': start_idx,
            'end_idx': end_idx,
        }

    abnormal_count = sum(1 for r in variant_records if r['metadata']['is_abnormal'])
    normalization_fail_count_do1 = sum(
        1
        for r in variant_records
        if r['metadata']['cf_result_do1']['normalization_ok'] is False
    )
    normalization_fail_count_do0 = sum(
        1
        for r in variant_records
        if r['metadata']['cf_result_do0']['normalization_ok'] is False
    )

    total_cpts = 0
    total_pa0 = 0
    total_pa1 = 0
    total_pa2 = 0
    for question, scm in zip(questions, scms):
        nodes_kept = question.nodes_kept
        total_cpts += len(nodes_kept)
        for node in nodes_kept:
            num_parents = len(scm.cpts[node]['parents'])
            if num_parents == 0:
                total_pa0 += 1
            elif num_parents == 1:
                total_pa1 += 1
            else:
                total_pa2 += 1

    graph_variant_counts = {1: 0, 2: 0, 3: 0}
    query_variant_counts = {1: 0, 2: 0, 3: 0}
    for record in variant_records:
        graph_variant_counts[record['metadata']['graph_variant']] += 1
        query_variant_counts[record['metadata']['query_variant']] += 1

    total_depth = sum(
        r['metadata']['treatment_outcome_depth'] for r in variant_records
    )
    total_eff_conds = sum(
        r['metadata']['effective_prob_conds'] for r in variant_records
    )
    naive_count = sum(
        1 for r in variant_records if r['metadata']['has_naive_conditioning']
    )

    return {
        'num_examples': num_examples,
        'num_unique_dags': num_unique_dags,
        'avg_num_nodes': num_nodes,
        'abnormal_count': abnormal_count,
        'abnormal_rate': abnormal_count / num_examples,
        'normalization_fail_count_do1': normalization_fail_count_do1,
        'normalization_fail_rate_do1': normalization_fail_count_do1 / num_examples,
        'normalization_fail_count_do0': normalization_fail_count_do0,
        'normalization_fail_rate_do0': normalization_fail_count_do0 / num_examples,
        'avg_num_cpts': total_cpts / num_examples,
        'avg_num_cpts_pa0': total_pa0 / num_examples,
        'avg_num_cpts_pa1': total_pa1 / num_examples,
        'avg_num_cpts_pa2': total_pa2 / num_examples,
        'avg_treatment_outcome_depth': total_depth / num_examples,
        'avg_effective_prob_conds': total_eff_conds / num_examples,
        'naive_conditioning_rate': naive_count / num_examples,
        'graph_variant_counts': graph_variant_counts,
        'graph_variant_rates': {
            key: value / num_examples for key, value in graph_variant_counts.items()
        },
        'query_variant_counts': query_variant_counts,
        'query_variant_rates': {
            key: value / num_examples for key, value in query_variant_counts.items()
        },
        'max_abnormal_ratio': max_abnormal_ratio,
        'source_triplet_corpus': source_triplet_corpus,
        'start_idx': start_idx,
        'end_idx': end_idx,
    }


def write_outputs(
    args,
    filename: str,
    all_variant_records: Dict[str, List[Dict]],
    all_variant_questions: Dict[str, List[AssembledQuestion]],
    scms: List[RLCausalSCM],
    num_unique_dags: int,
    num_nodes: int,
    max_abnormal_ratio: float,
    start_idx: int,
    end_idx: int,
) -> None:
    for variant_name in VARIANT_NAMES:
        records = all_variant_records[variant_name]
        questions = all_variant_questions[variant_name]

        jsonl_path = os.path.join(args.output_dir, variant_name, 'jsonls', f'{filename}.jsonl')
        csv_path = os.path.join(args.output_dir, variant_name, 'csvs', f'{filename}.csv')
        stats_path = os.path.join(args.output_dir, variant_name, 'stats', f'{filename}.json')

        with open(jsonl_path, 'w', encoding='utf-8') as file_obj:
            for record in records:
                file_obj.write(json.dumps(record, ensure_ascii=False) + '\n')

        with open(csv_path, 'w', newline='', encoding='utf-8') as file_obj:
            writer = csv.DictWriter(
                file_obj, fieldnames=['idx', 'question', 'answer', 'metadata']
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        'idx': record['idx'],
                        'question': record['question'],
                        'answer': record['answer'],
                        'metadata': json.dumps(record['metadata'], ensure_ascii=False),
                    }
                )

        stats = compute_variant_stats(
            records,
            num_unique_dags,
            num_nodes,
            scms,
            questions,
            max_abnormal_ratio,
            args.triplet_corpus_path,
            start_idx,
            end_idx,
        )
        with open(stats_path, 'w', encoding='utf-8') as file_obj:
            json.dump(stats, file_obj, indent=2, ensure_ascii=False)
            file_obj.write('\n')

        print(f'  {variant_name}: {jsonl_path}')


def main():
    parser = argparse.ArgumentParser(description='ATT question assembly pipeline')
    parser.add_argument('--triplet_corpus_path', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--run_timestamp', type=str, required=True)
    args = parser.parse_args()

    print('=== ATT Question Assembly Pipeline Config ===')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()

    print('Loading triplet corpus...')
    all_records = load_triplet_corpus(args.triplet_corpus_path)
    print(f'  Loaded {len(all_records)} triplet records')
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_records)
    records_to_process = [
        record for record in all_records if start_idx <= record['idx'] < end_idx
    ]
    num_examples = len(records_to_process)
    print(f'  Processing {num_examples} records (idx range [{start_idx}, {end_idx}))')
    print()

    ensure_output_dirs(args.output_dir)

    print('Assembling questions...')
    all_variant_records: Dict[str, List[Dict]] = {name: [] for name in VARIANT_NAMES}
    all_variant_questions: Dict[str, List[AssembledQuestion]] = {
        name: [] for name in VARIANT_NAMES
    }
    scms: List[RLCausalSCM] = []
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc='Assembling questions'):
        scm, att_result, decimal_places, example_seed = reconstruct_from_triplet(record)
        scms.append(scm)
        dag_edges_key = tuple(tuple(edge) for edge in record['dag_edges'])
        seen_dag_edge_sets.add(dag_edges_key)

        questions = assemble_all_questions(
            scm,
            att_result,
            decimal_places=decimal_places,
            example_seed=example_seed,
            treatment_outcome_depth=record['treatment_outcome_depth'],
            effective_prob_conds=record['effective_prob_conds'],
            has_naive_conditioning=record['has_naive_conditioning'],
        )

        for question in questions:
            output_record = {
                'idx': record['idx'],
                'question': question.full_question_text,
                'answer': question.att_ground_truth,
                'metadata': question.metadata,
            }
            all_variant_records[question.variant_name].append(output_record)
            all_variant_questions[question.variant_name].append(question)

    num_unique_dags = len(seen_dag_edge_sets)
    num_nodes = triplet_stats['num_nodes']
    max_abnormal_ratio = triplet_stats['max_abnormal_ratio']
    print()

    print('Writing outputs...')
    filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
    write_outputs(
        args,
        filename,
        all_variant_records,
        all_variant_questions,
        scms,
        num_unique_dags,
        num_nodes,
        max_abnormal_ratio,
        start_idx,
        end_idx,
    )
    print()

    sample_records = all_variant_records[VARIANT_NAMES[0]]
    abnormal_count = sum(1 for r in sample_records if r['metadata']['is_abnormal'])
    print('=== Summary ===')
    print(f'  Total examples: {num_examples}')
    print(f'  Unique DAGs: {num_unique_dags}')
    print(f'  Max abnormal ratio: {max_abnormal_ratio}')
    print(f'  Abnormal count: {abnormal_count}')
    print(f'  Source: {args.triplet_corpus_path}')
    print('=== Done ===')


if __name__ == '__main__':
    main()
