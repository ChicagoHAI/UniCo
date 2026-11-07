from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from data_generation.ate.ate_question_assembly_pipeline import _deserialize_cpts, build_filename, ensure_output_dirs, load_triplet_corpus
from data_generation.marginal_prob.marginal_prob_query_sampling import MarginalProbResult
from data_generation.marginal_prob.marginal_prob_question_assembly import assemble_all_questions
from data_generation.scm_mechanism_sampling import RLCausalSCM

VARIANT_NAMES = ["full", "query_ancestor_prune"]


def reconstruct_from_triplet(record: Dict):
    dag = nx.DiGraph(); dag.add_nodes_from(range(record["num_nodes"])); dag.add_edges_from([tuple(edge) for edge in record["dag_edges"]])
    scm = RLCausalSCM.from_cpts(dag, _deserialize_cpts(record["cpts"]), decimal_places=record["decimal_places"])
    return scm, MarginalProbResult.from_json_dict(record["marginal_prob_result"]), record["decimal_places"], record["example_seed"], record["idx"]


def compute_variant_stats(rows: List[Dict], num_unique_dags: int, num_nodes: int, source_triplet_corpus: str, start_idx: int, end_idx: int) -> Dict:
    num_examples = len(rows)
    if num_examples == 0:
        return {"query_type": "marginal_prob", "num_examples": 0, "num_unique_dags": num_unique_dags, "avg_num_nodes": num_nodes, "source_triplet_corpus": source_triplet_corpus, "start_idx": start_idx, "end_idx": end_idx}
    abnormal_count = sum(1 for row in rows if row["metadata"].get("is_abnormal"))
    graph_variant_counts = {1: 0, 2: 0, 3: 0}
    query_variant_counts = {1: 0, 2: 0, 3: 0}
    total_kept = 0
    for row in rows:
        graph_variant_counts[row["metadata"]["graph_variant"]] += 1
        query_variant_counts[row["metadata"]["query_variant"]] += 1
        total_kept += len(row["metadata"].get("nodes_kept", []))
    return {
        "query_type": "marginal_prob",
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_nodes": num_nodes,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "avg_num_cpts": total_kept / num_examples,
        "graph_variant_counts": graph_variant_counts,
        "query_variant_counts": query_variant_counts,
        "source_triplet_corpus": source_triplet_corpus,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Marginal Prob question assembly pipeline")
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
    ensure_output_dirs(args.output_dir)
    outputs = {variant: [] for variant in VARIANT_NAMES}
    seen_dag_edge_sets = set()
    for record in tqdm(selected, desc="Assembling questions"):
        scm, result, decimal_places, example_seed, example_idx = reconstruct_from_triplet(record)
        seen_dag_edge_sets.add(tuple(tuple(edge) for edge in record["dag_edges"]))
        for question in assemble_all_questions(scm, result, decimal_places=decimal_places, example_seed=example_seed, example_idx=example_idx):
            question.metadata["is_abnormal"] = record.get("is_abnormal")
            question.metadata["num_nodes"] = record["num_nodes"]
            question.metadata["dag_edges"] = record["dag_edges"]
            outputs[question.variant_name].append({"idx": record["idx"], "question": question.full_question_text, "answer": question.answer, "metadata": question.metadata})
    filename = build_filename(args, args.triplet_corpus_path, start_idx, end_idx)
    num_nodes = selected[0]["num_nodes"] if selected else 0
    for variant, rows in outputs.items():
        variant_dir = Path(args.output_dir) / variant
        (variant_dir / "jsonls").mkdir(parents=True, exist_ok=True)
        (variant_dir / "csvs").mkdir(parents=True, exist_ok=True)
        (variant_dir / "stats").mkdir(parents=True, exist_ok=True)
        jsonl_path = variant_dir / "jsonls" / f"{filename}.jsonl"
        csv_path = variant_dir / "csvs" / f"{filename}.csv"
        stats_path = variant_dir / "stats" / f"{filename}.json"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "question", "answer", "metadata"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"idx": row["idx"], "question": row["question"], "answer": row["answer"], "metadata": json.dumps(row["metadata"], ensure_ascii=False)})
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(compute_variant_stats(rows, len(seen_dag_edge_sets), num_nodes, args.triplet_corpus_path, start_idx, end_idx), f, indent=2, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
