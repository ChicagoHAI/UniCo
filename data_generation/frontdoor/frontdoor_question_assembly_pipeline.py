"""
frontdoor_question_assembly_pipeline.py

Loads frontdoor triplets and assembles textual frontdoor questions.
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from data_generation.ate.ate_question_assembly_pipeline import (
    build_filename,
    load_triplet_corpus,
    load_triplet_stats,
)
from data_generation.frontdoor.frontdoor_query_sampling import FrontdoorResult
from data_generation.frontdoor.frontdoor_question_assembly import (
    VARIANT_NAME,
    assemble_all_questions,
    format_answer_for_export,
)


VARIANT_NAMES = [VARIANT_NAME]


def ensure_output_dirs(output_dir: str) -> None:
    for variant_name in VARIANT_NAMES:
        for subdir in ("jsonls", "csvs", "stats"):
            os.makedirs(os.path.join(output_dir, variant_name, subdir), exist_ok=True)


def reconstruct_from_triplet(record: Dict):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(record["num_nodes"]))
    dag.add_edges_from([tuple(edge) for edge in record["dag_edges"]])
    frontdoor_result = FrontdoorResult.from_json_dict(record["frontdoor_result"])
    return dag, frontdoor_result, record["example_seed"]


def compute_variant_stats(
    variant_records: List[Dict],
    num_unique_dags: int,
    num_nodes: int,
    max_abnormal_ratio: float,
    source_triplet_corpus: str,
    start_idx: int,
    end_idx: int,
) -> Dict:
    num_examples = len(variant_records)
    abnormal_count = sum(1 for record in variant_records if record["metadata"]["is_abnormal"])
    total_depth = sum(record["metadata"]["treatment_outcome_depth"] for record in variant_records)
    total_eff = sum(record["metadata"]["effective_prob_conds"] for record in variant_records)
    naive_count = sum(1 for record in variant_records if record["metadata"]["has_naive_conditioning"])

    graph_variant_counts = {1: 0, 2: 0, 3: 0}
    query_variant_counts = {1: 0, 2: 0, 3: 0}
    num_correct_options_counts = {count: 0 for count in range(5)}
    for record in variant_records:
        graph_variant_counts[record["metadata"]["graph_variant"]] += 1
        query_variant_counts[record["metadata"]["query_variant"]] += 1
        num_correct_options_counts[record["metadata"]["num_correct_options"]] += 1

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_nodes": num_nodes,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples if num_examples else 0.0,
        "avg_treatment_outcome_depth": total_depth / num_examples if num_examples else 0.0,
        "avg_effective_prob_conds": total_eff / num_examples if num_examples else 0.0,
        "naive_conditioning_rate": naive_count / num_examples if num_examples else 0.0,
        "graph_variant_counts": graph_variant_counts,
        "query_variant_counts": query_variant_counts,
        "num_correct_options_counts": num_correct_options_counts,
        "max_abnormal_ratio": max_abnormal_ratio,
        "query_format": variant_records[0]["metadata"].get("query_format") if num_examples else None,
        "source_triplet_corpus": source_triplet_corpus,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def write_outputs(
    args,
    filename: str,
    all_variant_records: Dict[str, List[Dict]],
    num_unique_dags: int,
    num_nodes: int,
    max_abnormal_ratio: float,
    start_idx: int,
    end_idx: int,
) -> None:
    for variant_name in VARIANT_NAMES:
        records = all_variant_records[variant_name]
        jsonl_path = os.path.join(args.output_dir, variant_name, "jsonls", f"{filename}.jsonl")
        csv_path = os.path.join(args.output_dir, variant_name, "csvs", f"{filename}.csv")
        stats_path = os.path.join(args.output_dir, variant_name, "stats", f"{filename}.json")

        with open(jsonl_path, "w", encoding="utf-8") as file_obj:
            for record in records:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["idx", "question", "answer", "metadata"])
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "idx": record["idx"],
                        "question": record["question"],
                        "answer": record["answer"],
                        "metadata": json.dumps(record["metadata"], ensure_ascii=False),
                    }
                )

        stats = compute_variant_stats(
            records,
            num_unique_dags,
            num_nodes,
            max_abnormal_ratio,
            args.triplet_corpus_path,
            start_idx,
            end_idx,
        )
        with open(stats_path, "w", encoding="utf-8") as file_obj:
            json.dump(stats, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")

        print(f"  {variant_name}: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Frontdoor question assembly pipeline")
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    print("=== Frontdoor Question Assembly Pipeline Config ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    all_records = load_triplet_corpus(args.triplet_corpus_path)
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(all_records)
    records_to_process = [
        record for record in all_records if start_idx <= record["idx"] < end_idx
    ]
    print(f"  Processing {len(records_to_process)} records (idx range [{start_idx}, {end_idx}))")
    print()

    ensure_output_dirs(args.output_dir)

    all_variant_records: Dict[str, List[Dict]] = {name: [] for name in VARIANT_NAMES}
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc="Assembling questions"):
        dag, frontdoor_result, example_seed = reconstruct_from_triplet(record)
        assembled_questions = assemble_all_questions(
            dag,
            frontdoor_result,
            example_seed=example_seed,
            treatment_outcome_depth=record["treatment_outcome_depth"],
            effective_prob_conds=record["effective_prob_conds"],
            has_naive_conditioning=record["has_naive_conditioning"],
        )
        seen_dag_edge_sets.add(tuple(tuple(edge) for edge in record["dag_edges"]))

        for question in assembled_questions:
            output_record = {
                "idx": record["idx"],
                "question": question.full_question_text,
                "answer": format_answer_for_export(
                    question.metadata["query_format"],
                    question.correct_options,
                ),
                "metadata": question.metadata,
            }
            all_variant_records[question.variant_name].append(output_record)

    num_nodes = records_to_process[0]["num_nodes"] if records_to_process else 0
    filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
    write_outputs(
        args,
        filename,
        all_variant_records,
        len(seen_dag_edge_sets),
        num_nodes,
        triplet_stats.get("max_abnormal_ratio", 0.0),
        start_idx,
        end_idx,
    )
    print("=== Done ===")


if __name__ == "__main__":
    main()
