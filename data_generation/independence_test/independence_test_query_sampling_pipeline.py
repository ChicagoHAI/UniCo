from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

from tqdm import tqdm

from data_generation.ate.ate_query_sampling_pipeline import _serialize_cpts, ensure_output_dirs
from data_generation.dag_cache_utils import load_complete_unique_dags, save_complete_unique_dags
from data_generation.rlcausal_dag_sampling import CausalGraphSampler
from data_generation.scm_mechanism_sampling import RLCausalSCM
from data_generation.independence_test.independence_test_query_sampling import QUERY_FORMATS, build_independence_test_query
from data_generation.dataset_utils.graph_only_balance import BINARY_LABELS, MULTI_CHOICE_LABELS, compute_target_answer_counts, count_answer_labels, select_most_underfilled_label

MAX_QUERY_RETRIES = 100


def enumerate_unique_dags(num_nodes: int, seed: int, max_unique_dags: int):
    cached = load_complete_unique_dags(num_nodes)
    if cached is not None:
        return cached[:max_unique_dags]
    sampler = CausalGraphSampler(num_nodes=num_nodes, max_attempts=10000, seed=seed)
    dags = []
    exhausted = False
    with tqdm(total=max_unique_dags, desc="Sampling unique DAGs") as pbar:
        while len(dags) < max_unique_dags:
            try:
                dags.append(sampler.sample())
                pbar.update(1)
            except RuntimeError:
                exhausted = True
                break
    if exhausted and len(dags) < max_unique_dags:
        save_complete_unique_dags(num_nodes, dags)
    return dags


def compute_stats(records: List[Dict], num_unique_dags: int, args) -> Dict:
    num_examples = len(records)
    abnormal_count = sum(record["independence_test_result"]["is_abnormal"] for record in records)
    stats = {"query_type": "independence_test", "query_format": args.query_format, "num_examples": num_examples, "num_unique_dags": num_unique_dags, "num_nodes": args.num_nodes, "avg_num_edges": sum(r["num_edges"] for r in records) / num_examples, "abnormal_count": abnormal_count, "abnormal_rate": abnormal_count / num_examples, "max_abnormal_ratio": args.max_abnormal_ratio, "decimal_places": args.decimal_places, "max_conditioning": args.max_conditioning}
    if args.query_format in {"binary", "multi_choice"}:
        answer_label_counts = count_answer_labels(records, "independence_test_result")
        stats["answer_label_counts"] = answer_label_counts
        stats["answer_label_rates"] = {label: count / num_examples for label, count in answer_label_counts.items()}
    return stats


def main():
    parser = argparse.ArgumentParser(description="Independence Test query sampling pipeline")
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decimal_places", type=int, default=2)
    parser.add_argument("--max_conditioning", type=int, default=2)
    parser.add_argument("--query_format", type=str, choices=QUERY_FORMATS, default="binary")
    parser.add_argument("--max_abnormal_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()
    if not (0.0 <= args.max_abnormal_ratio <= 1.0): parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")
    ensure_output_dirs(args.output_dir)
    dags = enumerate_unique_dags(args.num_nodes, args.seed, args.max_unique_dags)
    max_abnormal_count = math.floor(args.max_abnormal_ratio * args.num_examples)
    num_abnormal_so_far = 0
    target_answer_counts = None
    current_answer_counts = None
    if args.query_format == "binary":
        target_answer_counts = compute_target_answer_counts(BINARY_LABELS, args.num_examples)
        current_answer_counts = {label: 0 for label in BINARY_LABELS}
    elif args.query_format == "multi_choice":
        target_answer_counts = compute_target_answer_counts(MULTI_CHOICE_LABELS, args.num_examples)
        current_answer_counts = {label: 0 for label in MULTI_CHOICE_LABELS}
    records = []
    traversal_pass = 0
    with tqdm(total=args.num_examples, desc="Generating triplets") as progress_bar:
        while len(records) < args.num_examples:
            progress_this_pass = False
            target_answer_label = None
            if current_answer_counts is not None:
                target_answer_label = select_most_underfilled_label(target_answer_counts, current_answer_counts)
            for dag_offset in range(len(dags)):
                if len(records) >= args.num_examples: break
                idx = len(records)
                dag = dags[(traversal_pass + dag_offset) % len(dags)]
                example_seed = args.seed + traversal_pass * len(dags) + dag_offset
                scm = RLCausalSCM(dag, seed=example_seed, decimal_places=args.decimal_places)
                query_rng = random.Random(example_seed)
                abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
                result = None
                attempt = 0
                while attempt < MAX_QUERY_RETRIES:
                    result = build_independence_test_query(scm, query_format=args.query_format, option_order_rng=query_rng, max_conditioning=args.max_conditioning, target_answer_label=target_answer_label)
                    attempt += 1
                    if result is None: continue
                    if abnormal_cap_reached and result.is_abnormal:
                        result = None
                        continue
                    break
                if result is None: continue
                if result.is_abnormal: num_abnormal_so_far += 1
                records.append({"idx": idx, "query_type": "independence_test", "query_format": args.query_format, "num_nodes": len(scm.nodes), "num_edges": dag.number_of_edges(), "dag_edges": sorted([list(edge) for edge in dag.edges()]), "cpts": _serialize_cpts(scm), "independence_test_result": result.to_json_dict(), "example_seed": example_seed, "decimal_places": args.decimal_places})
                if current_answer_counts is not None: current_answer_counts[result.correct_options] += 1
                progress_bar.update(1)
                progress_this_pass = True
                if current_answer_counts is not None and current_answer_counts[target_answer_label] >= target_answer_counts[target_answer_label]: break
            if not progress_this_pass:
                raise RuntimeError(f"Could not construct enough independence_test queries to reach num_examples under query_format={args.query_format}.")
            traversal_pass += 1
    filename = f"{args.run_timestamp}-independence_test-nodes{args.num_nodes}-examples{args.num_examples}-maxdags{args.max_unique_dags}-max_abnormal{args.max_abnormal_ratio}-fmt{args.query_format}-max_conditioning{args.max_conditioning}-dp{args.decimal_places}-seed{args.seed}"
    jsonl_path = os.path.join(args.output_dir, "triplets", "jsonls", f"{filename}.jsonl")
    stats_path = os.path.join(args.output_dir, "triplets", "stats", f"{filename}.json")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(compute_stats(records, len(dags), args), f, indent=2, ensure_ascii=False); f.write("\n")


if __name__ == "__main__":
    main()
