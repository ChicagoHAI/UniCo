from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List

from tqdm import tqdm

from data_generation.ate.ate_query_sampling_pipeline import (
    _serialize_cpts,
    assign_dags_to_examples,
    ensure_output_dirs,
)
from data_generation.dag_cache_utils import load_complete_unique_dags, save_complete_unique_dags
from data_generation.rlcausal_dag_sampling import CausalGraphSampler
from data_generation.observed_difference.observed_difference_query_sampling import ObservedDifferenceResult, sample_observed_difference_query, is_abnormal_observed_difference_query
from data_generation.scm_mechanism_sampling import RLCausalSCM

MAX_QUERY_RETRIES = 10000


def enumerate_unique_dags(num_nodes: int, seed: int, max_unique_dags: int):
    cached_dags = load_complete_unique_dags(num_nodes)
    if cached_dags is not None:
        return cached_dags[:max_unique_dags]
    sampler = CausalGraphSampler(num_nodes=num_nodes, max_attempts=10000, seed=seed)
    unique_dags = []
    exhausted = False
    with tqdm(total=max_unique_dags, desc="Sampling unique DAGs") as pbar:
        while len(unique_dags) < max_unique_dags:
            try:
                unique_dags.append(sampler.sample())
                pbar.update(1)
            except RuntimeError:
                exhausted = True
                break
    if exhausted and len(unique_dags) < max_unique_dags:
        save_complete_unique_dags(num_nodes, unique_dags)
    return unique_dags


def compute_stats(records: List[Dict], num_unique_dags: int, args) -> Dict:
    num_examples = len(records)
    abnormal_count = sum(1 for record in records if record["is_abnormal"])
    return {
        "query_type": "observed_difference",
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_edges": sum(r["num_edges"] for r in records) / num_examples,
        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "num_nodes": args.num_nodes,
        "seed": args.seed,
        "max_unique_dags": args.max_unique_dags,
        "max_abnormal_ratio": args.max_abnormal_ratio,
        "decimal_places": args.decimal_places,
    }


def build_filename(args) -> str:
    return (
        f"{args.run_timestamp}-observed_difference-nodes{args.num_nodes}-examples{args.num_examples}"
        f"-maxdags{args.max_unique_dags}-max_abnormal{args.max_abnormal_ratio}"
        f"-dp{args.decimal_places}-seed{args.seed}"
    )


def write_outputs(args, filename: str, records: List[Dict], num_unique_dags: int) -> None:
    jsonl_path = os.path.join(args.output_dir, "triplets", "jsonls", f"{filename}.jsonl")
    stats_path = os.path.join(args.output_dir, "triplets", "stats", f"{filename}.json")
    with open(jsonl_path, "w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(stats_path, "w", encoding="utf-8") as file_obj:
        json.dump(compute_stats(records, num_unique_dags, args), file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="observed_difference query sampling pipeline")
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--max_unique_dags", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decimal_places", type=int, default=2)
    parser.add_argument("--max_abnormal_ratio", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    if not (0.0 <= args.max_abnormal_ratio <= 1.0):
        parser.error("--max_abnormal_ratio must be in [0.0, 1.0]")

    print("=== Observed Difference Query Sampling Pipeline Config ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    ensure_output_dirs(args.output_dir)
    unique_dags = enumerate_unique_dags(args.num_nodes, args.seed, args.max_unique_dags)
    dag_assignment = assign_dags_to_examples(unique_dags, args.num_examples, args.seed)

    records: List[Dict] = []
    max_abnormal_count = math.floor(args.max_abnormal_ratio * args.num_examples)
    num_abnormal_so_far = 0

    for idx in tqdm(range(args.num_examples), desc="Generating triplets"):
        dag = dag_assignment[idx]
        example_seed = args.seed + idx
        scm = RLCausalSCM(dag, seed=example_seed, decimal_places=args.decimal_places)
        query_rng = random.Random(example_seed)

        abnormal_cap_reached = num_abnormal_so_far >= max_abnormal_count
        attempt = 0
        query_spec, result = sample_observed_difference_query(scm, query_rng)
        while attempt < MAX_QUERY_RETRIES:
            candidate_is_abnormal = is_abnormal_observed_difference_query(dag, result.source, result.target)
            if abnormal_cap_reached and candidate_is_abnormal:
                query_spec, result = sample_observed_difference_query(scm, query_rng)
                attempt += 1
                continue
            break
        if attempt >= MAX_QUERY_RETRIES:
            print(f"  Warning: idx={idx}, could not find acceptable observed_difference query after {MAX_QUERY_RETRIES} retries. Accepting final sampled query.")

        is_abnormal = is_abnormal_observed_difference_query(dag, result.source, result.target)
        if is_abnormal:
            num_abnormal_so_far += 1

        records.append({"idx": idx, "query_type": "observed_difference", "num_nodes": len(scm.nodes), "num_edges": dag.number_of_edges(), "dag_edges": sorted([list(edge) for edge in dag.edges()]), "cpts": _serialize_cpts(scm), "observed_difference_result": result.to_json_dict(), "query_spec": query_spec, "is_abnormal": is_abnormal, "example_seed": example_seed, "decimal_places": args.decimal_places})

    filename = build_filename(args)
    write_outputs(args, filename, records, len(unique_dags))


if __name__ == "__main__":
    main()
