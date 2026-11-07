"""
ate_question_assembly_pipeline.py

Pipeline 2: ATE question assembly — loads a triplet corpus (DAG, SCM,
ATEResult) and assembles textual questions across three condition variants
(full, surgery_only, surgery_ancestor_prune).

Reads triplet JSONL produced by ate_query_sampling_pipeline.py, reconstructs
SCMs from saved CPTs via RLCausalSCM.from_cpts(), and generates question
JSONL, CSV, and stats per variant.
"""

import argparse
import ast
import csv
import json
import os
from typing import Dict, List

import networkx as nx
from tqdm import tqdm

from data_generation.scm_mechanism_sampling import RLCausalSCM
from data_generation.ate.ate_query_sampling import ATEResult
from data_generation.ate.ate_question_assembly import (
    assemble_all_questions,
    AssembledQuestion,
)
from data_generation.dataset_utils.difficulty_metrics import (
    compute_treatment_outcome_depth,
    compute_effective_prob_conds,
    has_naive_conditioning,
)


VARIANT_NAMES = ["full", "surgery_only", "surgery_ancestor_prune"]


def ensure_output_dirs(output_dir: str) -> None:
    """Create output directory structure for all variants.

    Creates jsonls/, csvs/, stats/ subdirectories under each variant.
    The .logs/ directory is created by the shell runner (needed by tee
    before Python starts).
    """
    for variant_name in VARIANT_NAMES:
        for subdir in ("jsonls", "csvs", "stats"):
            os.makedirs(
                os.path.join(output_dir, variant_name, subdir), exist_ok=True
            )


def _deserialize_cpts(cpts_json: Dict) -> Dict:
    """Deserialize CPTs from a triplet JSONL record.

    Converts string node IDs back to ints and string tuple keys back to
    actual tuples via ast.literal_eval.
    """
    cpts = {}
    for node_str, entry in cpts_json.items():
        node = int(node_str)
        parents = entry['parents']
        table = {ast.literal_eval(k): v for k, v in entry['table'].items()}
        cpts[node] = {'parents': parents, 'table': table}
    return cpts


def reconstruct_from_triplet(record: Dict):
    """Reconstruct DAG, SCM, and ATEResult from a triplet JSONL record.

    The SCM is reconstructed from saved CPTs via RLCausalSCM.from_cpts()
    — no re-sampling occurs.

    Returns:
        (scm, ate_result, treatment, outcome, decimal_places, example_seed)
    """
    # Reconstruct DAG
    dag = nx.DiGraph()
    dag.add_nodes_from(range(record["num_nodes"]))
    dag.add_edges_from([tuple(e) for e in record["dag_edges"]])

    # Reconstruct SCM from saved CPTs
    cpts = _deserialize_cpts(record["cpts"])
    decimal_places = record["decimal_places"]
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=decimal_places)

    # Reconstruct ATEResult
    ar = record["ate_result"]
    ate_result = ATEResult(
        treatment=record["treatment"],
        outcome=record["outcome"],
        ate_surgery=ar["ate_surgery"],
        ate_causal_query=ar["ate_causal_query"],
        is_abnormal=ar["is_abnormal"],
        verification_ok=ar["verification_ok"],
    )

    return scm, ate_result, record["treatment"], record["outcome"], decimal_places, record["example_seed"]


def load_triplet_corpus(triplet_corpus_path: str) -> List[Dict]:
    """Load all records from a triplet JSONL file."""
    records = []
    with open(triplet_corpus_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_triplet_stats(triplet_corpus_path: str) -> Dict:
    """Load the companion stats JSON for a triplet corpus.

    Infers the stats path from the JSONL path: same basename, stats/
    directory instead of jsonls/, .json instead of .jsonl.
    """
    # .../triplets/jsonls/foo.jsonl → .../triplets/stats/foo.json
    jsonl_dir = os.path.dirname(triplet_corpus_path)
    stats_dir = os.path.join(os.path.dirname(jsonl_dir), "stats")
    basename = os.path.basename(triplet_corpus_path)
    stats_basename = basename.replace(".jsonl", ".json")
    stats_path = os.path.join(stats_dir, stats_basename)

    with open(stats_path, "r") as f:
        return json.load(f)


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
    """Compute dataset-level statistics for one variant."""
    num_examples = len(variant_records)

    abnormal_count = sum(
        1 for r in variant_records if r["metadata"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in variant_records if r["metadata"]["verification_ok"] is False
    )

    # avg_num_cpts and parent-count breakdowns
    total_cpts = 0
    total_pa0 = 0
    total_pa1 = 0
    total_pa2 = 0

    for q, scm in zip(questions, scms):
        nodes_kept = q.nodes_kept
        total_cpts += len(nodes_kept)
        for node in nodes_kept:
            num_parents = len(scm.cpts[node]["parents"])
            if num_parents == 0:
                total_pa0 += 1
            elif num_parents == 1:
                total_pa1 += 1
            else:
                total_pa2 += 1

    # Graph/query variant distributions
    graph_variant_counts = {1: 0, 2: 0, 3: 0}
    query_variant_counts = {1: 0, 2: 0, 3: 0}
    for r in variant_records:
        gv = r["metadata"]["graph_variant"]
        qv = r["metadata"]["query_variant"]
        graph_variant_counts[gv] += 1
        query_variant_counts[qv] += 1

    # Difficulty metrics
    total_depth = sum(
        r["metadata"]["treatment_outcome_depth"] for r in variant_records
    )
    total_eff_conds = sum(
        r["metadata"]["effective_prob_conds"] for r in variant_records
    )
    naive_count = sum(
        1 for r in variant_records if r["metadata"]["has_naive_conditioning"]
    )

    return {
        "num_examples": num_examples,
        "num_unique_dags": num_unique_dags,
        "avg_num_nodes": num_nodes,

        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "verification_false_count": verification_false_count,
        "verification_false_rate": verification_false_count / num_examples,

        "avg_num_cpts": total_cpts / num_examples,
        "avg_num_cpts_pa0": total_pa0 / num_examples,
        "avg_num_cpts_pa1": total_pa1 / num_examples,
        "avg_num_cpts_pa2": total_pa2 / num_examples,

        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,

        "graph_variant_counts": graph_variant_counts,
        "graph_variant_rates": {k: v / num_examples for k, v in graph_variant_counts.items()},
        "query_variant_counts": query_variant_counts,
        "query_variant_rates": {k: v / num_examples for k, v in query_variant_counts.items()},

        "max_abnormal_ratio": max_abnormal_ratio,
        "source_triplet_corpus": source_triplet_corpus,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def build_filename(
    args, triplet_corpus_path: str, start_idx: int, end_idx: int,
) -> str:
    """Build the base filename (without extension).

    Format: {timestamp}-start{S}-end{E}-from-{corpus_basename}.
    The end_idx should be the resolved value (never -1).
    """
    corpus_basename = os.path.splitext(os.path.basename(triplet_corpus_path))[0]
    return (
        f"{args.run_timestamp}"
        f"-start{start_idx}"
        f"-end{end_idx}"
        f"-from-{corpus_basename}"
    )


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
    """Write JSONL, CSV, and stats JSON for each variant."""
    for variant_name in VARIANT_NAMES:
        records = all_variant_records[variant_name]
        questions = all_variant_questions[variant_name]

        # Paths
        jsonl_path = os.path.join(
            args.output_dir, variant_name, "jsonls", f"{filename}.jsonl"
        )
        csv_path = os.path.join(
            args.output_dir, variant_name, "csvs", f"{filename}.csv"
        )
        stats_path = os.path.join(
            args.output_dir, variant_name, "stats", f"{filename}.json"
        )

        # JSONL
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # CSV
        with open(csv_path, "w", newline="") as f:
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
                        "metadata": json.dumps(
                            record["metadata"], ensure_ascii=False
                        ),
                    }
                )

        # Stats JSON
        stats = compute_variant_stats(
            records, num_unique_dags, num_nodes, scms, questions,
            max_abnormal_ratio, args.triplet_corpus_path,
            start_idx, end_idx,
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  {variant_name}: {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ATE question assembly pipeline — triplet corpus to questions"
    )
    parser.add_argument("--triplet_corpus_path", type=str, required=True,
                        help="Path to input triplet JSONL")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First index to process (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Last index to process (exclusive); -1 means all")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    args = parser.parse_args()

    # Print config
    print("=== ATE Question Assembly Pipeline Config ===")
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
    print("Assembling questions...")

    # Collect results per variant
    all_variant_records: Dict[str, List[Dict]] = {v: [] for v in VARIANT_NAMES}
    all_variant_questions: Dict[str, List[AssembledQuestion]] = {
        v: [] for v in VARIANT_NAMES
    }
    scms: List[RLCausalSCM] = []

    # Track unique DAGs
    seen_dag_edge_sets = set()

    for record in tqdm(records_to_process, desc="Assembling questions"):
        scm, ate_result, treatment, outcome, decimal_places, example_seed = (
            reconstruct_from_triplet(record)
        )
        scms.append(scm)

        # Track unique DAGs
        dag_edges_key = tuple(tuple(e) for e in record["dag_edges"])
        seen_dag_edge_sets.add(dag_edges_key)

        # Extract difficulty metrics from triplet record (fallback: compute on-the-fly)
        if "treatment_outcome_depth" in record:
            tod = record["treatment_outcome_depth"]
            epc = record["effective_prob_conds"]
            # Support both old ("has_naive_shortcut") and new ("has_naive_conditioning") key
            if "has_naive_conditioning" in record:
                ns = record["has_naive_conditioning"]
            else:
                ns = record["has_naive_shortcut"]
        else:
            dag = scm.dag
            tod = compute_treatment_outcome_depth(dag, [treatment], outcome)
            epc = compute_effective_prob_conds(dag, [treatment], outcome, evidence={})
            ns = has_naive_conditioning(dag, [treatment], outcome, evidence_nodes=[])

        questions = assemble_all_questions(
            scm, treatment, outcome, ate_result,
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
                "answer": q.ate_ground_truth,
                "metadata": q.metadata,
            }
            all_variant_records[q.variant_name].append(output_record)
            all_variant_questions[q.variant_name].append(q)

    num_unique_dags = len(seen_dag_edge_sets)
    num_nodes = triplet_stats["num_nodes"]
    max_abnormal_ratio = triplet_stats["max_abnormal_ratio"]
    print()

    # Write outputs
    print("Writing outputs...")
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

    # Summary
    sample_records = all_variant_records[VARIANT_NAMES[0]]
    abnormal_count = sum(
        1 for r in sample_records if r["metadata"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in sample_records if r["metadata"]["verification_ok"] is False
    )
    print("=== Summary ===")
    print(f"  Total examples: {num_examples}")
    print(f"  Unique DAGs: {num_unique_dags}")
    print(f"  Max abnormal ratio: {max_abnormal_ratio}")
    print(f"  Abnormal count: {abnormal_count}")
    print(f"  Verification false count: {verification_false_count}")
    print(f"  Source: {args.triplet_corpus_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
