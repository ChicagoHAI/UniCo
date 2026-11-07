"""
nie_nde_question_assembly_pipeline.py

Assembly pipeline for NIE/NDE question corpora.
"""

from __future__ import annotations

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
    load_triplet_corpus,
    load_triplet_stats,
)
from data_generation.nie_nde.nie_nde_query_sampling import NIENDEResult
from data_generation.nie_nde.nie_nde_question_assembly import (
    AssembledQuestion,
    assemble_all_questions,
)
from data_generation.scm_mechanism_sampling import RLCausalSCM


VARIANT_NAMES = ["full", "surgery_only", "surgery_ancestor_prune"]
EFFECT_TYPES = ["nie", "nde"]
def _replace_path_segment(path: str, old: str, new: str) -> str:
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    for idx, part in enumerate(parts):
        if part == old:
            parts[idx] = new
            if parts[0] == "":
                return os.sep + os.path.join(*parts[1:])
            return os.path.join(*parts)
    return os.path.join(norm, new)


def derive_effect_output_dirs(
    output_dir: str,
    output_dir_nie: str | None,
    output_dir_nde: str | None,
) -> Dict[str, str]:
    return {
        "nie": output_dir_nie or _replace_path_segment(output_dir, "nie_nde", "nie"),
        "nde": output_dir_nde or _replace_path_segment(output_dir, "nie_nde", "nde"),
    }


def ensure_output_dirs(effect_output_dirs: Dict[str, str]) -> None:
    for effect in EFFECT_TYPES:
        base_dir = effect_output_dirs[effect]
        for variant_name in VARIANT_NAMES:
            for subdir in ("jsonls", "csvs", "stats"):
                os.makedirs(
                    os.path.join(base_dir, variant_name, subdir), exist_ok=True
                )


def reconstruct_from_triplet(record: Dict):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(record["num_nodes"]))
    dag.add_edges_from([tuple(edge) for edge in record["dag_edges"]])
    cpts = _deserialize_cpts(record["cpts"])
    decimal_places = record["decimal_places"]
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=decimal_places)
    nie_nde_result = NIENDEResult.from_json_dict(record["nie_nde_result"])
    return scm, nie_nde_result, decimal_places, record["example_seed"]


def compute_variant_stats(
    variant_records: List[Dict],
    num_unique_dags: int,
    num_nodes: int,
    scms: List[RLCausalSCM],
    questions: List[AssembledQuestion],
    triplet_stats: Dict,
    source_triplet_corpus: str,
    start_idx: int,
    end_idx: int,
) -> Dict:
    num_examples = len(variant_records)

    if num_examples == 0:
        return {
            "num_examples": 0,
            "num_unique_dags": num_unique_dags,
            "avg_num_nodes": num_nodes,
            "source_triplet_corpus": source_triplet_corpus,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "triplet_stats": triplet_stats,
        }

    abnormal_count = sum(
        1 for record in variant_records if record["metadata"]["is_abnormal"]
    )

    total_cpts = 0
    total_pa0 = 0
    total_pa1 = 0
    total_pa2 = 0
    for question, scm in zip(questions, scms):
        total_cpts += len(question.nodes_kept)
        for node in question.nodes_kept:
            num_parents = len(scm.cpts[node]["parents"])
            if num_parents == 0:
                total_pa0 += 1
            elif num_parents == 1:
                total_pa1 += 1
            else:
                total_pa2 += 1

    graph_variant_counts = {1: 0, 2: 0, 3: 0}
    query_variant_counts = {1: 0, 2: 0, 3: 0}
    total_depth = 0.0
    total_eff_conds = 0
    naive_count = 0

    for record in variant_records:
        metadata = record["metadata"]
        graph_variant_counts[metadata["graph_variant"]] += 1
        query_variant_counts[metadata["query_variant"]] += 1
        total_depth += metadata["treatment_outcome_depth"]
        total_eff_conds += metadata["effective_prob_conds"]
        if metadata["has_naive_conditioning"]:
            naive_count += 1

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_nodes": num_nodes,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "avg_num_cpts": total_cpts / num_examples,
        "avg_num_cpts_pa0": total_pa0 / num_examples,
        "avg_num_cpts_pa1": total_pa1 / num_examples,
        "avg_num_cpts_pa2": total_pa2 / num_examples,
        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,
        "graph_variant_counts": graph_variant_counts,
        "graph_variant_rates": {
            key: value / num_examples for key, value in graph_variant_counts.items()
        },
        "query_variant_counts": query_variant_counts,
        "query_variant_rates": {
            key: value / num_examples for key, value in query_variant_counts.items()
        },
        "source_triplet_corpus": source_triplet_corpus,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "triplet_stats": triplet_stats,
    }


def _write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path: str, records: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "question", "answer", "metadata"]
        )
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


def write_outputs(
    effect_output_dirs: Dict[str, str],
    filename_by_effect: Dict[str, str],
    all_records: Dict[str, Dict[str, List[Dict]]],
    all_questions: Dict[str, Dict[str, List[AssembledQuestion]]],
    all_scms: Dict[str, Dict[str, List[RLCausalSCM]]],
    num_unique_dags: int,
    num_nodes: int,
    triplet_stats: Dict,
    source_triplet_corpus: str,
    start_idx: int,
    end_idx: int,
) -> None:
    for effect in EFFECT_TYPES:
        print(f"  [{effect.upper()}]")
        base_dir = effect_output_dirs[effect]
        filename = filename_by_effect[effect]
        for variant_name in VARIANT_NAMES:
            records = all_records[effect][variant_name]
            questions = all_questions[effect][variant_name]
            scms = all_scms[effect][variant_name]
            variant_dir = os.path.join(base_dir, variant_name)

            jsonl_path = os.path.join(variant_dir, "jsonls", f"{filename}.jsonl")
            csv_path = os.path.join(variant_dir, "csvs", f"{filename}.csv")
            stats_path = os.path.join(variant_dir, "stats", f"{filename}.json")

            _write_jsonl(jsonl_path, records)
            _write_csv(csv_path, records)

            stats = compute_variant_stats(
                records,
                num_unique_dags,
                num_nodes,
                scms,
                questions,
                triplet_stats,
                source_triplet_corpus,
                start_idx,
                end_idx,
            )
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                f.write("\n")

            print(f"    {variant_name}:")
            print(f"      JSONL: {jsonl_path}")
            print(f"      CSV: {csv_path}")
            print(f"      Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NIE/NDE question assembly pipeline"
    )
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_dir_nie", type=str, default=None)
    parser.add_argument("--output_dir_nde", type=str, default=None)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    effect_output_dirs = derive_effect_output_dirs(
        args.output_dir, args.output_dir_nie, args.output_dir_nde
    )

    print("=== NIE/NDE Question Assembly Pipeline Config ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"  resolved_output_dir_nie: {effect_output_dirs['nie']}")
    print(f"  resolved_output_dir_nde: {effect_output_dirs['nde']}")
    print()

    ensure_output_dirs(effect_output_dirs)

    print("Loading triplet corpus...")
    triplets = load_triplet_corpus(args.triplet_corpus_path)
    triplet_stats = load_triplet_stats(args.triplet_corpus_path)
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx >= 0 else len(triplets)
    filtered = [record for record in triplets if start_idx <= record["idx"] < end_idx]
    print(f"  Processing {len(filtered)} triplets (idx range [{start_idx}, {end_idx}))")
    print()

    print("Assembling questions...")
    all_records: Dict[str, Dict[str, List[Dict]]] = {
        effect: {variant_name: [] for variant_name in VARIANT_NAMES}
        for effect in EFFECT_TYPES
    }
    all_questions: Dict[str, Dict[str, List[AssembledQuestion]]] = {
        effect: {variant_name: [] for variant_name in VARIANT_NAMES}
        for effect in EFFECT_TYPES
    }
    all_scms: Dict[str, Dict[str, List[RLCausalSCM]]] = {
        effect: {variant_name: [] for variant_name in VARIANT_NAMES}
        for effect in EFFECT_TYPES
    }
    seen_dags = set()

    for record in tqdm(filtered, desc="Assembling questions"):
        scm, nie_nde_result, decimal_places, example_seed = reconstruct_from_triplet(record)
        seen_dags.add(tuple(tuple(edge) for edge in record["dag_edges"]))

        for effect in EFFECT_TYPES:
            questions = assemble_all_questions(
                scm,
                nie_nde_result,
                effect_type=effect,
                decimal_places=decimal_places,
                example_seed=example_seed,
                treatment_outcome_depth=record["treatment_outcome_depth"],
                effective_prob_conds=record["effective_prob_conds"],
                has_naive_conditioning=record["has_naive_conditioning"],
            )
            for question in questions:
                output_record = {
                    "idx": record["idx"],
                    "question": question.full_question_text,
                    "answer": question.answer_value,
                    "metadata": question.metadata,
                }
                all_records[effect][question.variant_name].append(output_record)
                all_questions[effect][question.variant_name].append(question)
                all_scms[effect][question.variant_name].append(scm)

    num_unique_dags = len(seen_dags)
    num_nodes = triplet_stats["num_nodes"]
    filename_by_effect = {
        effect: build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
        for effect in EFFECT_TYPES
    }
    print()
    print("Writing outputs...")
    write_outputs(
        effect_output_dirs,
        filename_by_effect,
        all_records,
        all_questions,
        all_scms,
        num_unique_dags,
        num_nodes,
        triplet_stats,
        args.triplet_corpus_path,
        start_idx,
        end_idx,
    )
    print()

    total_questions = sum(
        len(all_records[effect][variant_name])
        for effect in EFFECT_TYPES
        for variant_name in VARIANT_NAMES
    )
    print("=== Summary ===")
    print(f"  Triplets: {len(filtered)}")
    print(f"  Questions emitted (all effects/variants): {total_questions}")
    for effect in EFFECT_TYPES:
        per_effect = sum(
            len(all_records[effect][variant_name]) for variant_name in VARIANT_NAMES
        )
        print(f"  {effect.upper()} questions: {per_effect}")
    print(f"  Unique DAGs: {num_unique_dags}")
    print(f"  Source: {args.triplet_corpus_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
