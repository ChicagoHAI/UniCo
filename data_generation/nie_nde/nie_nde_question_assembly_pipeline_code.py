"""
nie_nde_question_assembly_pipeline_code.py

NIE/NDE code-domain question assembly pipeline.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from tqdm import tqdm

from data_generation.dataset_utils.code_pipeline_shared import ensure_output_dirs, write_outputs
from data_generation.nie_nde.nie_nde_question_assembly_code import assemble_all_questions_code
from data_generation.nie_nde.nie_nde_question_assembly_pipeline import (
    EFFECT_TYPES,
    derive_effect_output_dirs,
    reconstruct_from_triplet,
    load_triplet_corpus,
    load_triplet_stats,
    build_filename,
)


def main():
    parser = argparse.ArgumentParser(description='NIE/NDE code-domain question assembly pipeline')
    parser.add_argument('--triplet_corpus_path', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_dir_nie', type=str, default=None)
    parser.add_argument('--output_dir_nde', type=str, default=None)
    parser.add_argument('--run_timestamp', type=str, required=True)
    args = parser.parse_args()

    effect_output_dirs = derive_effect_output_dirs(args.output_dir, args.output_dir_nie, args.output_dir_nde)
    for effect in EFFECT_TYPES:
        ensure_output_dirs(effect_output_dirs[effect])

    all_records = load_triplet_corpus(args.triplet_corpus_path)
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_records)
    records_to_process = [record for record in all_records if start_idx <= record['idx'] < end_idx]

    all_variant_records: Dict[str, Dict[str, List[Dict]]] = {
        effect: {'full': [], 'surgery_ancestor_prune': []}
        for effect in EFFECT_TYPES
    }
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc='Assembling code-domain questions'):
        scm, result, decimal_places, example_seed = reconstruct_from_triplet(record)
        seen_dag_edge_sets.add(tuple(tuple(edge) for edge in record['dag_edges']))
        for effect in EFFECT_TYPES:
            questions = assemble_all_questions_code(
                scm,
                result,
                effect_type=effect,
                decimal_places=decimal_places,
                example_seed=example_seed,
                treatment_outcome_depth=record['treatment_outcome_depth'],
                effective_prob_conds=record['effective_prob_conds'],
                has_naive_conditioning=record['has_naive_conditioning'],
            )
            for question in questions:
                all_variant_records[effect][question.variant_name].append({
                    'idx': record['idx'],
                    'question': question.full_question_text,
                    'answer': question.answer,
                    'metadata': question.metadata,
                })

    for effect in EFFECT_TYPES:
        filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
        write_outputs(
            effect_output_dirs[effect],
            args.triplet_corpus_path,
            filename,
            all_variant_records[effect],
            len(seen_dag_edge_sets),
            triplet_stats['num_nodes'],
            1.0,
            start_idx,
            end_idx,
        )


if __name__ == '__main__':
    main()
