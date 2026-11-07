"""
code_pipeline_shared.py

Shared utilities for ATE and CATE/JATE code-domain question assembly
pipelines. Contains output directory creation, variant folder mapping,
statistics computation, and file writing functions.

Both ate_question_assembly_pipeline_code.py and
cate_jate_question_assembly_pipeline_code.py use these functions
identically, so they are consolidated here.
"""

import csv
import json
import os
from typing import Dict, List


CODE_VARIANT_NAMES = ["full", "surgery_ancestor_prune"]
CODE_FOLDER_NAMES = ["full-code", "surgery_ancestor_prune-code"]


def ensure_output_dirs(output_dir: str, variant_names: List[str] | None = None) -> None:
    """Create output directory structure for code-domain variants."""
    folder_names = [variant_to_folder(name) for name in (variant_names or CODE_VARIANT_NAMES)]
    for folder_name in folder_names:
        for subdir in ("jsonls", "csvs", "stats"):
            os.makedirs(
                os.path.join(output_dir, folder_name, subdir), exist_ok=True
            )


def variant_to_folder(variant_name: str) -> str:
    """Map variant_name to code-domain folder name."""
    return f"{variant_name}-code"


def compute_variant_stats(
    variant_records: List[Dict],
    num_unique_dags: int,
    num_nodes: int,
    max_abnormal_ratio: float,
    source_triplet_corpus: str,
    start_idx: int,
    end_idx: int,
) -> Dict:
    """Compute dataset-level statistics for one code-domain variant."""
    num_examples = len(variant_records)
    if num_examples == 0:
        return {
            "num_examples": 0,
            "num_unique_dags": num_unique_dags,
            "avg_num_nodes": num_nodes,
            "abnormal_count": 0,
            "abnormal_rate": 0.0,
            "verification_false_count": 0,
            "verification_false_rate": 0.0,
            "avg_treatment_outcome_depth": 0.0,
            "avg_effective_prob_conds": 0.0,
            "naive_conditioning_rate": 0.0,
            "cf_variant_counts": {"A": 0, "B": 0, "C": 0},
            "cf_variant_rates": {"A": 0.0, "B": 0.0, "C": 0.0},
            "si_variant_counts": {"A": 0, "B": 0, "C": 0},
            "si_variant_rates": {"A": 0.0, "B": 0.0, "C": 0.0},
            "vn_variant_counts": {"A": 0, "B": 0},
            "vn_variant_rates": {"A": 0.0, "B": 0.0},
            "qv_variant_counts": {1: 0, 2: 0, 3: 0},
            "qv_variant_rates": {1: 0.0, 2: 0.0, 3: 0.0},
            "structure_counts": {"single": 0, "multi": 0},
            "structure_rates": {"single": 0.0, "multi": 0.0},
            "max_abnormal_ratio": max_abnormal_ratio,
            "source_triplet_corpus": source_triplet_corpus,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

    abnormal_count = sum(
        1 for r in variant_records if r["metadata"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in variant_records if r["metadata"]["verification_ok"] is False
    )

    # Code-domain axis distributions
    cf_counts = {"A": 0, "B": 0, "C": 0}
    si_counts = {"A": 0, "B": 0, "C": 0}
    vn_counts = {"A": 0, "B": 0}
    qv_counts = {1: 0, 2: 0, 3: 0}
    structure_counts = {"single": 0, "multi": 0}

    for r in variant_records:
        m = r["metadata"]
        cf_counts[m["cf_variant"]] += 1
        si_counts[m["si_variant"]] += 1
        vn_counts[m["vn_variant"]] += 1
        qv_counts[m["qv_variant"]] += 1
        structure_counts[m["code_structure"]] += 1

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

        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,

        "cf_variant_counts": cf_counts,
        "cf_variant_rates": {k: v / num_examples for k, v in cf_counts.items()},
        "si_variant_counts": si_counts,
        "si_variant_rates": {k: v / num_examples for k, v in si_counts.items()},
        "vn_variant_counts": vn_counts,
        "vn_variant_rates": {k: v / num_examples for k, v in vn_counts.items()},
        "qv_variant_counts": qv_counts,
        "qv_variant_rates": {k: v / num_examples for k, v in qv_counts.items()},
        "structure_counts": structure_counts,
        "structure_rates": {k: v / num_examples for k, v in structure_counts.items()},

        "max_abnormal_ratio": max_abnormal_ratio,
        "source_triplet_corpus": source_triplet_corpus,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def write_outputs(
    output_dir: str,
    triplet_corpus_path: str,
    filename: str,
    all_variant_records: Dict[str, List[Dict]],
    num_unique_dags: int,
    num_nodes: int,
    max_abnormal_ratio: float,
    start_idx: int,
    end_idx: int,
    variant_names: List[str] | None = None,
) -> None:
    """Write JSONL, CSV, and stats JSON for each code-domain variant."""
    for variant_name in (variant_names or CODE_VARIANT_NAMES):
        records = all_variant_records[variant_name]
        folder_name = variant_to_folder(variant_name)

        jsonl_path = os.path.join(
            output_dir, folder_name, "jsonls", f"{filename}.jsonl"
        )
        csv_path = os.path.join(
            output_dir, folder_name, "csvs", f"{filename}.csv"
        )
        stats_path = os.path.join(
            output_dir, folder_name, "stats", f"{filename}.json"
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
                writer.writerow({
                    "idx": record["idx"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "metadata": json.dumps(
                        record["metadata"], ensure_ascii=False
                    ),
                })

        # Stats JSON
        stats = compute_variant_stats(
            records, num_unique_dags, num_nodes,
            max_abnormal_ratio, triplet_corpus_path,
            start_idx, end_idx,
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  {folder_name}: {jsonl_path}")
