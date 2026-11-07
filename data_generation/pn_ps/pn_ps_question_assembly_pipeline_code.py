"""
pn_ps_question_assembly_pipeline_code.py

PN/PS code-domain question assembly pipeline.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from tqdm import tqdm

from data_generation.dataset_utils.code_pipeline_shared import ensure_output_dirs, write_outputs
from data_generation.pn_ps.pn_ps_question_assembly_code import assemble_all_questions_code
from data_generation.pn_ps.pn_ps_question_assembly_pipeline import (
    EFFECT_TYPES,
    build_filename,
    derive_effect_output_dirs,
    load_triplet_corpus,
    load_triplet_stats,
    reconstruct_from_triplet,
)


CODE_VARIANT_RECORDS = ["full", "surgery_ancestor_prune"]


def main() -> None:
    parser = argparse.ArgumentParser(description="PN/PS code-domain question assembly pipeline")
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_dir_pn", type=str, default=None)
    parser.add_argument("--output_dir_ps", type=str, default=None)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    effect_output_dirs = derive_effect_output_dirs(
        args.output_dir,
        args.output_dir_pn,
        args.output_dir_ps,
    )
    for effect in EFFECT_TYPES:
        ensure_output_dirs(effect_output_dirs[effect])

    all_records = load_triplet_corpus(args.triplet_corpus_path)
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_records)
    records_to_process = [
        record for record in all_records
        if start_idx <= record["idx"] < end_idx
    ]

    print("=== PN/PS Code Question Assembly Pipeline Config ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  resolved_output_dir_pn: {effect_output_dirs['pn']}")
    print(f"  resolved_output_dir_ps: {effect_output_dirs['ps']}")
    print(f"  Processing {len(records_to_process)} triplets (idx range [{start_idx}, {end_idx}))")
    print()

    all_variant_records: Dict[str, Dict[str, List[Dict]]] = {
        effect: {variant_name: [] for variant_name in CODE_VARIANT_RECORDS}
        for effect in EFFECT_TYPES
    }
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc="Assembling PN/PS code-domain questions"):
        scm, pn_ps_result, decimal_places, example_seed = reconstruct_from_triplet(record)
        seen_dag_edge_sets.add(tuple(tuple(edge) for edge in record["dag_edges"]))
        for effect in EFFECT_TYPES:
            questions = assemble_all_questions_code(
                scm,
                pn_ps_result,
                effect_type=effect,
                decimal_places=decimal_places,
                example_seed=example_seed,
                treatment_outcome_depth=record["treatment_outcome_depth"],
                effective_prob_conds=record["effective_prob_conds"],
                has_naive_conditioning=record["has_naive_conditioning"],
            )
            for question in questions:
                all_variant_records[effect][question.variant_name].append({
                    "idx": record["idx"],
                    "question": question.full_question_text,
                    "answer": question.answer,
                    "metadata": question.metadata,
                })

    for effect in EFFECT_TYPES:
        filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
        write_outputs(
            effect_output_dirs[effect],
            args.triplet_corpus_path,
            filename,
            all_variant_records[effect],
            len(seen_dag_edge_sets),
            triplet_stats["num_nodes"],
            triplet_stats.get("max_abnormal_ratio", 1.0),
            start_idx,
            end_idx,
        )

    total_questions = sum(
        len(all_variant_records[effect][variant_name])
        for effect in EFFECT_TYPES
        for variant_name in CODE_VARIANT_RECORDS
    )
    print()
    print("=== Summary ===")
    print(f"  Triplets: {len(records_to_process)}")
    print(f"  Questions emitted (all effects/variants): {total_questions}")
    for effect in EFFECT_TYPES:
        per_effect = sum(
            len(all_variant_records[effect][variant_name])
            for variant_name in CODE_VARIANT_RECORDS
        )
        print(f"  {effect.upper()} questions: {per_effect}")
    print(f"  Unique DAGs: {len(seen_dag_edge_sets)}")
    print(f"  Source: {args.triplet_corpus_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
