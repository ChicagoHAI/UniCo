"""marginal_prob_question_assembly_pipeline_code.py"""

from __future__ import annotations

import argparse
from typing import Dict, List

from tqdm import tqdm

from data_generation.dataset_utils.association_code_assembly_shared import ASSOCIATION_CODE_VARIANT_NAMES
from data_generation.dataset_utils.code_pipeline_shared import ensure_output_dirs, write_outputs
from data_generation.marginal_prob.marginal_prob_question_assembly_code import assemble_all_questions_code
from data_generation.marginal_prob.marginal_prob_question_assembly_pipeline import (
    build_filename,
    load_triplet_corpus,
    reconstruct_from_triplet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Marginal Prob code-domain question assembly pipeline")
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    records = load_triplet_corpus(args.triplet_corpus_path)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(records)
    selected = [record for record in records if start_idx <= record["idx"] < end_idx]
    ensure_output_dirs(args.output_dir, variant_names=ASSOCIATION_CODE_VARIANT_NAMES)

    outputs: Dict[str, List[Dict]] = {variant: [] for variant in ASSOCIATION_CODE_VARIANT_NAMES}
    seen_dag_edge_sets = set()
    for record in tqdm(selected, desc="Assembling marginal_prob code questions"):
        scm, result, decimal_places, example_seed, example_idx = reconstruct_from_triplet(record)
        seen_dag_edge_sets.add(tuple(tuple(edge) for edge in record["dag_edges"]))
        questions = assemble_all_questions_code(
            scm,
            result,
            decimal_places=decimal_places,
            example_seed=example_seed,
            example_idx=example_idx,
            is_abnormal=record.get("is_abnormal"),
            treatment_outcome_depth=record.get("treatment_outcome_depth", 0.0),
            effective_prob_conds=record.get("effective_prob_conds", 0),
            has_naive_conditioning=record.get("has_naive_conditioning", False),
        )
        for question in questions:
            outputs[question.variant_name].append({
                "idx": record["idx"],
                "question": question.full_question_text,
                "answer": question.answer,
                "metadata": question.metadata,
            })

    filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
    num_nodes = selected[0]["num_nodes"] if selected else 0
    write_outputs(
        args.output_dir,
        args.triplet_corpus_path,
        filename,
        outputs,
        len(seen_dag_edge_sets),
        num_nodes,
        1.0,
        start_idx,
        end_idx,
        variant_names=ASSOCIATION_CODE_VARIANT_NAMES,
    )


if __name__ == "__main__":
    main()
