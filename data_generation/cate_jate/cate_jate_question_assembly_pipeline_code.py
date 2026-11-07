"""
cate_jate_question_assembly_pipeline_code.py

Pipeline: CATE/JATE code-domain question assembly — loads a triplet corpus
(DAG, SCM, CATEJATEResult) and assembles code-domain questions across two
condition variants (full-code, surgery_ancestor_prune-code).

Reads triplet JSONL produced by cate_jate_query_sampling_pipeline.py (v1.1),
reconstructs SCMs from saved CPTs via RLCausalSCM.from_cpts(), and
generates question JSONL, CSV, and stats per variant.
"""

import argparse
from typing import Dict, List

from tqdm import tqdm

from data_generation.cate_jate.cate_jate_question_assembly_pipeline import (
    load_triplet_corpus,
    load_triplet_stats,
    reconstruct_from_triplet,
    build_filename,
)
from data_generation.cate_jate.cate_jate_question_assembly_code import (
    assemble_all_questions_code,
    CODE_VARIANT_NAMES,
)
from data_generation.dataset_utils.code_pipeline_shared import (
    ensure_output_dirs,
    write_outputs,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_treatment_outcome_depth,
    compute_effective_prob_conds,
    has_naive_conditioning,
)


def main():
    parser = argparse.ArgumentParser(
        description="CATE/JATE code-domain question assembly pipeline"
    )
    parser.add_argument("--triplet_corpus_path", type=str, required=True,
                        help="Path to input triplet JSONL (v1.1)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First index to process (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Last index to process (exclusive); -1 means all")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    print("=== CATE/JATE Code-Domain Question Assembly Pipeline Config ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    # Load triplet corpus
    print("Loading triplet corpus...")
    all_records = load_triplet_corpus(args.triplet_corpus_path)
    print(f"  Loaded {len(all_records)} triplet records")

    # Load companion stats
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)

    # Filter by index range
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_records)
    records_to_process = [
        r for r in all_records if start_idx <= r["idx"] < end_idx
    ]
    num_examples = len(records_to_process)
    print(f"  Processing {num_examples} records (idx range [{start_idx}, {end_idx}))")
    print()

    # Create output directories
    ensure_output_dirs(args.output_dir)

    # Assemble questions
    print("Assembling code-domain questions...")

    all_variant_records: Dict[str, List[Dict]] = {v: [] for v in CODE_VARIANT_NAMES}
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc="Assembling code questions"):
        (scm, cate_jate_result, treatments, outcome,
         evidence, decimal_places, example_seed) = reconstruct_from_triplet(record)

        dag_edges_key = tuple(tuple(e) for e in record["dag_edges"])
        seen_dag_edge_sets.add(dag_edges_key)

        # Extract difficulty metrics from triplet record
        if "treatment_outcome_depth" in record:
            tod = record["treatment_outcome_depth"]
            epc = record["effective_prob_conds"]
            if "has_naive_conditioning" in record:
                ns = record["has_naive_conditioning"]
            else:
                ns = record["has_naive_shortcut"]
        else:
            dag = scm.dag
            tod = compute_treatment_outcome_depth(dag, treatments, outcome)
            epc = compute_effective_prob_conds(dag, treatments, outcome, evidence=evidence)
            evidence_nodes = sorted(evidence.keys()) if evidence else []
            ns = has_naive_conditioning(dag, treatments, outcome, evidence_nodes=evidence_nodes)

        questions = assemble_all_questions_code(
            scm, treatments, outcome, evidence, cate_jate_result,
            decimal_places=decimal_places,
            example_seed=example_seed,
            treatment_outcome_depth=tod,
            effective_prob_conds=epc,
            has_naive_conditioning=ns,
        )

        for q in questions:
            output_record = {
                "idx": record["idx"],
                "question": q.full_question_text,
                "answer": q.effect_ground_truth,
                "metadata": q.metadata,
            }
            all_variant_records[q.variant_name].append(output_record)

    num_unique_dags = len(seen_dag_edge_sets)
    num_nodes = triplet_stats["num_nodes"]
    max_abnormal_ratio = triplet_stats["max_abnormal_ratio"]
    max_naive_conditioning_ratio = triplet_stats.get("max_naive_conditioning_ratio", 1.0)
    print()

    # Write outputs
    print("Writing outputs...")
    filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
    write_outputs(
        args.output_dir, args.triplet_corpus_path,
        filename, all_variant_records,
        num_unique_dags, num_nodes, max_abnormal_ratio,
        start_idx, end_idx,
    )
    print()

    # Summary
    sample_records = all_variant_records[CODE_VARIANT_NAMES[0]]
    abnormal_count = sum(
        1 for r in sample_records if r["metadata"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in sample_records if r["metadata"]["verification_ok"] is False
    )
    naive_conditioning_count = sum(
        1 for r in sample_records if r["metadata"]["has_naive_conditioning"]
    )
    print("=== Summary ===")
    print(f"  Total examples: {num_examples}")
    print(f"  Unique DAGs: {num_unique_dags}")
    print(f"  Abnormal count: {abnormal_count}")
    print(f"  Max abnormal ratio: {max_abnormal_ratio}")
    print(f"  Naive conditioning count: {naive_conditioning_count}")
    print(f"  Max naive conditioning ratio: {max_naive_conditioning_ratio}")
    print(f"  Verification false count: {verification_false_count}")
    print(f"  Source: {args.triplet_corpus_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
