"""Prompt loading, grading, metrics, and reports for the unified OOD evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ood_eval.benchmarks.bbeh_causal_understanding.eval_utils import grade_bbeh_answer
from ood_eval.benchmarks.calm_lite.eval_utils import grade_calm_lite_answer
from ood_eval.benchmarks.cladder.eval_utils import grade_cladder_answer
from ood_eval.benchmarks.com_2.eval_utils import grade_com2_answer
from ood_eval.benchmarks.corr2cause.eval_utils import (
    compute_binary_f1,
    grade_corr2cause_answer,
)
from ood_eval.benchmarks.counterbench.eval_utils import grade_counterbench_answer
from ood_eval.benchmarks.exec_cf.eval_utils import (
    extract_exec_cf_answer,
    grade_exec_cf_answer,
)
from shared_utils.grading import extract_last_boxed
from shared_utils.vllm_backends import infer_model_dir_name

SUPPORTED_DATASETS: Tuple[str, ...] = (
    "bbeh_causal_understanding",
    "calm_lite",
    "cladder",
    "com_2",
    "corr2cause",
    "counterbench",
    "exec_cf",
)

REQUIRED_FIELDS_BY_DATASET: Dict[str, Tuple[str, ...]] = {
    "bbeh_causal_understanding": ("idx", "question", "answer"),
    "calm_lite": ("idx", "question", "answer", "question_type"),
    "cladder": ("idx", "question", "answer"),
    "com_2": ("idx", "type", "question", "answer"),
    "corr2cause": ("idx", "question", "answer"),
    "counterbench": ("idx", "question", "answer", "type"),
    "exec_cf": ("idx", "question", "answer"),
}

PRIMARY_METRIC_BY_DATASET: Dict[str, str] = {
    "bbeh_causal_understanding": "accuracy",
    "calm_lite": "overall_accuracy",
    "cladder": "accuracy",
    "com_2": "global_average_f1",
    "corr2cause": "f1",
    "counterbench": "accuracy_rate",
    "exec_cf": "f1_mean",
}

NO_PROMPT_LIMIT = int(1e18)
EVALUATION_FAMILY = "ood"
PRIMARY_METRIC_NOTE = (
    "Primary metric by dataset: accuracy for BBEH/CaLM-Lite/CLadder/CounterBench; "
    "F1 for Com2/Corr2Cause/ExecCF."
)

CSV_LEADING_COLUMNS: Tuple[str, ...] = (
    "global_idx",
    "source_index",
    "source_dataset",
    "source_idx",
    "source_line_idx",
    "idx",
    "model_name",
    "run_timestamp",
    "round",
    "round_seed",
    "question",
    "answer",
    "reference_answer",
    "prediction",
    "extraction_success",
    "score",
    "f1_score",
    "exact_match",
    "correct",
    "completion_length",
    "response",
)

FormatPromptFn = Callable[[str], str]
BatchQueryFn = Callable[[Any, List[str], Any, Optional[int]], Tuple[List[str], List[int]]]


@dataclass(frozen=True)
class PromptSource:
    source_index: int
    prompts_path: str
    source_dataset: str


def infer_source_dataset(prompts_path: str) -> str:
    """Infer the benchmark from the one supported dataset folder in a prompt path."""
    parts = [part for part in Path(os.path.normpath(prompts_path)).parts if part != "."]
    matches = list(dict.fromkeys(part for part in parts if part in SUPPORTED_DATASETS))
    if len(matches) != 1:
        raise ValueError(
            "Could not infer exactly one supported source dataset from "
            f"{prompts_path!r}; matches={matches}"
        )
    return matches[0]


def build_prompt_sources(prompts_path_list: Sequence[str]) -> List[PromptSource]:
    """Validate the prompt files and record which benchmark each one belongs to."""
    if not prompts_path_list:
        raise ValueError("prompts_path_list must not be empty")

    sources: List[PromptSource] = []
    for source_index, prompts_path in enumerate(prompts_path_list):
        if not os.path.isfile(prompts_path):
            raise FileNotFoundError(f"Prompt file not found: {prompts_path}")
        sources.append(
            PromptSource(
                source_index=source_index,
                prompts_path=prompts_path,
                source_dataset=infer_source_dataset(prompts_path),
            )
        )
    return sources


def unique_dataset_order(sources: Sequence[PromptSource]) -> List[str]:
    """List the benchmarks once each, in the order of the prompt files."""
    return list(dict.fromkeys(source.source_dataset for source in sources))


def load_integrated_ood_prompts(
    sources: Sequence[PromptSource],
    format_prompt: FormatPromptFn,
    max_prompts_per_file: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load every prompt file into one list and render its prompts."""
    all_data: List[Dict[str, Any]] = []
    limit = max_prompts_per_file if max_prompts_per_file is not None else NO_PROMPT_LIMIT

    for source in sources:
        num_loaded_for_source = 0
        with open(source.prompts_path, "r", encoding="utf-8") as f:
            for source_line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                if num_loaded_for_source >= limit:
                    break

                ex = json.loads(line)
                for required_field in REQUIRED_FIELDS_BY_DATASET[source.source_dataset]:
                    if required_field not in ex:
                        raise ValueError(
                            f"Missing required field {required_field!r} in {source.prompts_path} "
                            f"at source_line_idx={source_line_idx}"
                        )

                entry = dict(ex)
                entry.update(
                    {
                        "global_idx": len(all_data),
                        "source_index": source.source_index,
                        "source_dataset": source.source_dataset,
                        "source_prompts_path": source.prompts_path,
                        "source_line_idx": source_line_idx,
                        "source_idx": ex["idx"],
                        "formatted_prompt": format_prompt(ex["question"]),
                    }
                )
                all_data.append(entry)
                num_loaded_for_source += 1

        print(
            f"Loaded {num_loaded_for_source} examples from {source.prompts_path} "
            f"({source.source_dataset})",
            flush=True,
        )

    print(f"\nLoaded {len(all_data)} integrated OOD examples from {len(sources)} files.\n")
    return all_data


def _safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return mean(values_list) if values_list else 0.0


def _letters_to_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    return {str(item) for item in value}


def grade_ood_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and grade one response with the rules of its benchmark."""
    dataset = record["source_dataset"]
    response = record["response"]
    answer = record["answer"]

    graded = {key: value for key, value in record.items() if key != "formatted_prompt"}
    details: Dict[str, Any] = {}

    if dataset == "bbeh_causal_understanding":
        pred_str, extraction_success = extract_last_boxed(response)
        score, details = grade_bbeh_answer(pred_str, str(answer))
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": "actual_causation",
                "prediction": pred_str,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": score,
                "correct": bool(details.get("correct")),
            }
        )

    elif dataset == "calm_lite":
        pred_str, extraction_success = extract_last_boxed(response)
        question_type = record["question_type"]
        score, details = grade_calm_lite_answer(pred_str, answer, question_type)
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": question_type,
                "prediction": pred_str,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": score,
                "correct": bool(score),
            }
        )

    elif dataset == "cladder":
        pred_str, extraction_success = extract_last_boxed(response)
        score, details = grade_cladder_answer(pred_str, str(answer))
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": record.get("query_type"),
                "prediction": pred_str,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": score,
                "correct": bool(score),
            }
        )

    elif dataset == "com_2":
        pred_str, extraction_success = extract_last_boxed(response)
        example_type = record["type"]
        score, details = grade_com2_answer(pred_str, answer, example_type)
        if example_type == "decision":
            pred_set = _letters_to_set(details.get("pred_letters"))
            ref_set = _letters_to_set(details.get("ref_letters"))
            exact_match = 1.0 if pred_set == ref_set else 0.0
        else:
            exact_match = float(score)
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": example_type,
                "prediction": pred_str,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": score,
                "f1_score": score,
                "exact_match": exact_match,
                "correct": bool(exact_match),
            }
        )

    elif dataset == "corr2cause":
        pred_str, extraction_success = extract_last_boxed(response)
        correct, details = grade_corr2cause_answer(pred_str, str(answer))
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": "binary_label",
                "prediction": pred_str,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": 1.0 if correct else 0.0,
                "correct": correct,
            }
        )

    elif dataset == "counterbench":
        pred_str, extraction_success = extract_last_boxed(response)
        correct = grade_counterbench_answer(pred_str, str(answer))
        details = {"correct": correct}
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": record.get("type"),
                "prediction": pred_str,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": 1.0 if correct else 0.0,
                "correct": correct,
            }
        )

    elif dataset == "exec_cf":
        pred_answers, extraction_success = extract_exec_cf_answer(response)
        f1_score, exact_match, details = grade_exec_cf_answer(pred_answers, answer)
        graded.update(
            {
                "evaluation_family": EVALUATION_FAMILY,
                "evaluation_query_type": "counterfactual_execution",
                "prediction": pred_answers,
                "reference_answer": answer,
                "extraction_success": extraction_success,
                "score": f1_score,
                "f1_score": f1_score,
                "exact_match": exact_match,
                "correct": bool(exact_match),
            }
        )

    else:
        raise ValueError(f"Unsupported source_dataset: {dataset}")

    graded.update(details)
    return graded


def _base_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"total": 0, "extraction_rate": 0.0, "avg_completion_length": 0.0}
    return {
        "total": len(records),
        "extraction_rate": _safe_mean(
            1.0 if record.get("extraction_success", False) else 0.0 for record in records
        ),
        "avg_completion_length": _safe_mean(
            float(record.get("completion_length", 0)) for record in records
        ),
    }


def compute_dataset_round_metrics(
    dataset: str,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute one benchmark's metrics for one round, including its primary metric."""
    metrics = _base_metrics(records)
    primary_metric_name = PRIMARY_METRIC_BY_DATASET[dataset]
    metrics["primary_metric_name"] = primary_metric_name
    if not records:
        metrics["primary_metric"] = None
        return metrics

    if dataset in {"bbeh_causal_understanding", "cladder"}:
        metrics["accuracy"] = _safe_mean(float(record["score"]) for record in records)

    elif dataset == "calm_lite":
        for question_type, metric_prefix in (
            ("binary classification", "binary_classification"),
            ("choice selection", "choice_selection"),
            ("probability calculation", "probability_calculation"),
        ):
            group = [record for record in records if record["question_type"] == question_type]
            if group:
                metrics[f"{metric_prefix}_accuracy"] = _safe_mean(
                    float(record["score"]) for record in group
                )
                metrics[f"{metric_prefix}_count"] = len(group)
        metrics["overall_accuracy"] = _safe_mean(float(record["score"]) for record in records)

    elif dataset == "com_2":
        counterfactual_group = [r for r in records if r["type"] == "counterfactual"]
        if counterfactual_group:
            metrics["counterfactual_accuracy"] = _safe_mean(
                float(record["score"]) for record in counterfactual_group
            )
            metrics["counterfactual_count"] = len(counterfactual_group)

        decision_group = [r for r in records if r["type"] == "decision"]
        if decision_group:
            metrics["decision_f1"] = _safe_mean(
                float(record["f1_score"]) for record in decision_group
            )
            metrics["decision_em"] = _safe_mean(
                float(record["exact_match"]) for record in decision_group
            )
            metrics["decision_count"] = len(decision_group)

        metrics["global_average_f1"] = _safe_mean(float(record["score"]) for record in records)
        metrics["global_average_em"] = _safe_mean(
            float(record["exact_match"]) for record in records
        )

    elif dataset == "corr2cause":
        predictions = [str(record.get("prediction", "")) for record in records]
        references = [str(record.get("answer", "")) for record in records]
        f1_results = compute_binary_f1(predictions, references, positive_label="Yes")
        metrics.update(
            {
                "accuracy": _safe_mean(
                    1.0 if record.get("correct", False) else 0.0 for record in records
                ),
                **{
                    key: f1_results[key]
                    for key in ("f1", "precision", "recall", "tp", "fp", "fn")
                },
            }
        )

    elif dataset == "counterbench":
        metrics["accuracy_rate"] = _safe_mean(
            1.0 if record.get("correct", False) else 0.0 for record in records
        )
        type_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            type_groups[str(record.get("type", "<missing>"))].append(record)
        for type_name, group in sorted(type_groups.items()):
            metric_name = type_name.replace("-", "_").replace(" ", "_")
            metrics[f"{metric_name}_accuracy_rate"] = _safe_mean(
                1.0 if record.get("correct", False) else 0.0 for record in group
            )
            metrics[f"{metric_name}_count"] = len(group)

    elif dataset == "exec_cf":
        metrics["f1_mean"] = _safe_mean(float(record["f1_score"]) for record in records)
        metrics["exact_match_rate"] = _safe_mean(
            float(record["exact_match"]) for record in records
        )

    else:
        raise ValueError(f"Unsupported source_dataset: {dataset}")

    metrics["primary_metric"] = metrics.get(primary_metric_name)
    return metrics


def get_primary_metric_value(dataset: str, metrics: Dict[str, Any]) -> Optional[float]:
    """Read a benchmark's primary metric from its metrics dictionary."""
    metric_name = metrics.get("primary_metric_name")
    if not isinstance(metric_name, str):
        metric_name = PRIMARY_METRIC_BY_DATASET[dataset]
    value = metrics.get(metric_name)
    return None if value is None else float(value)


def compute_integrated_round_metrics(
    records: Sequence[Dict[str, Any]],
    by_dataset_metrics: Dict[str, Dict[str, Any]],
    dataset_order: Sequence[str],
) -> Dict[str, Any]:
    """Compute the suite-level metrics of one round, including the macro primary metric."""
    metrics = _base_metrics(records)
    primary_values = [
        value
        for dataset in dataset_order
        for value in [get_primary_metric_value(dataset, by_dataset_metrics[dataset])]
        if value is not None
    ]
    metrics["num_datasets"] = len(primary_values)
    metrics["macro_primary_metric"] = _safe_mean(primary_values)
    return metrics


def _group_records_by_dataset(
    records: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_dataset"])].append(record)
    return grouped


def _dataset_counts(all_data: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for item in all_data:
        counts[str(item["source_dataset"])] += 1
    return dict(counts)


def _signature(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "max_new_tokens": args.max_new_tokens,
    }


def _mean_numeric_metrics(round_stats: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    numeric_values: Dict[str, List[float]] = defaultdict(list)
    for item in round_stats:
        for key, value in item.get("metrics", {}).items():
            if key == "total" or key.endswith("_count"):
                continue
            if isinstance(value, (int, float)):
                numeric_values[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in numeric_values.items() if values}


def summarize_cross_round_primary_metrics(
    round_stats: Sequence[Dict[str, Any]],
    dataset_order: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute the mean, max, and min of each benchmark's primary metric across rounds."""
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for row_name, agg_fn in (
        ("mean", lambda vals: sum(vals) / len(vals)),
        ("max", max),
        ("min", min),
    ):
        row: Dict[str, Optional[float]] = {}
        for dataset in dataset_order:
            vals = [
                value
                for round_item in round_stats
                if dataset in round_item.get("by_dataset", {})
                for value in [get_primary_metric_value(dataset, round_item["by_dataset"][dataset])]
                if value is not None
            ]
            row[dataset] = agg_fn(vals) if vals else None
        summary[row_name] = row
    return summary


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    df = pd.DataFrame(records)
    leading_columns = [column for column in CSV_LEADING_COLUMNS if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in leading_columns]
    df.to_csv(path, index=False, columns=leading_columns + remaining_columns)


def _format_metric(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "NA"


def render_ood_table(
    rows: Sequence[Tuple[str, Dict[str, Optional[float]]]],
    dataset_order: Sequence[str],
    *,
    include_macro_average: bool = False,
) -> str:
    column_names = list(dataset_order)
    if include_macro_average:
        column_names.append("macro_avg")

    lines = [
        "| Row | " + " | ".join(column_names) + " |",
        "|---|" + "|".join(["---"] * len(column_names)) + "|",
    ]
    for row_name, values in rows:
        cells = [_format_metric(values.get(dataset)) for dataset in dataset_order]
        if include_macro_average:
            dataset_values = [
                value
                for dataset in dataset_order
                for value in [values.get(dataset)]
                if value is not None
            ]
            cells.append(_format_metric(mean(dataset_values) if dataset_values else None))
        lines.append(f"| {row_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_per_round_markdown(
    *,
    model_name: str,
    round_index: int,
    by_dataset: Dict[str, Dict[str, Any]],
    shared_filename: str,
    dataset_order: Sequence[str],
) -> str:
    values = {
        dataset: get_primary_metric_value(dataset, by_dataset[dataset])
        for dataset in dataset_order
    }
    parts = [
        f"# Unified OOD Evaluation Round {round_index}",
        "",
        f"- Model: `{model_name}`",
        f"- Run: `{shared_filename}`",
        "",
        PRIMARY_METRIC_NOTE,
        "",
        render_ood_table([(f"round_{round_index}", values)], dataset_order),
        "",
    ]
    return "\n".join(parts)


def render_cross_round_markdown(
    *,
    model_name: str,
    cross_round_values: Dict[str, Dict[str, Optional[float]]],
    shared_filename: str,
    rounds: Sequence[int],
    dataset_order: Sequence[str],
) -> str:
    rows = [(name, cross_round_values[name]) for name in ("mean", "max", "min")]
    parts = [
        "# Unified OOD Evaluation Cross-Round Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Run: `{shared_filename}`",
        f"- Rounds: `{', '.join(str(round_index) for round_index in rounds)}`",
        "",
        PRIMARY_METRIC_NOTE,
        "The `macro_avg` column is the equal-weight mean of the displayed dataset values in each row; max/min rows can combine extrema from different rounds.",
        "",
        render_ood_table(rows, dataset_order, include_macro_average=True),
        "",
    ]
    return "\n".join(parts)


def run_unified_ood_evaluation(
    *,
    args: argparse.Namespace,
    sources: Sequence[PromptSource],
    all_data: Sequence[Dict[str, Any]],
    model: Any,
    batch_query: BatchQueryFn,
) -> None:
    """Generate and grade every round, then write per-benchmark artifacts and reports."""
    eval_start_time = time.time()
    dataset_order = unique_dataset_order(sources)
    model_name = infer_model_dir_name(args.model)
    timestamp = args.run_timestamp or time.strftime("%Y%m%d_%H%M%S")

    results_root = Path(args.results_base_dir)
    dataset_outputs_root = results_root / "dataset-specific-outputs"
    integrated_model_root = results_root / "integrated-outputs" / model_name
    integrated_stats_dir = integrated_model_root / "stats"
    integrated_mds_dir = integrated_model_root / "mds"
    integrated_stats_dir.mkdir(parents=True, exist_ok=True)
    integrated_mds_dir.mkdir(parents=True, exist_ok=True)

    shared_filename = (
        f"{timestamp}-temp{args.temperature}-top_p{args.top_p}"
        f"-top_k{args.top_k}-presence_penalty{args.presence_penalty}"
        f"-max_tokens{args.max_new_tokens}-seed{args.sampling_seed}"
    )

    print("Running unified OOD evaluation.", flush=True)
    print(f"Results root: {results_root}", flush=True)
    print("Prompt sources:", flush=True)
    for source in sources:
        print(f"  - {json.dumps(asdict(source), sort_keys=True)}", flush=True)

    prompts_for_vllm = [item["formatted_prompt"] for item in all_data]
    source_counts = _dataset_counts(all_data)
    all_round_stats: List[Dict[str, Any]] = []
    rounds_run: List[int] = []
    dataset_round_stats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def paths_of_dataset(dataset: str) -> List[str]:
        return [source.prompts_path for source in sources if source.source_dataset == dataset]

    for round_index in range(args.start_round, args.budget + 1):
        rounds_run.append(round_index)
        round_seed = round_index * args.sampling_seed
        print(
            f"\n=== Unified OOD round {round_index}/{args.budget} "
            f"({len(prompts_for_vllm)} prompts, seed={round_seed}) ===",
            flush=True,
        )
        round_start_time = time.time()
        responses, completion_lengths = batch_query(model, prompts_for_vllm, args, round_seed)
        round_duration_s = time.time() - round_start_time
        print(
            f"Round {round_index} generation done in {round_duration_s:.2f}s "
            f"({round_duration_s / len(prompts_for_vllm):.2f}s/prompt)",
            flush=True,
        )
        if len(responses) != len(prompts_for_vllm) or len(completion_lengths) != len(
            prompts_for_vllm
        ):
            raise ValueError(
                f"Expected {len(prompts_for_vllm)} completions, got {len(responses)} responses "
                f"and {len(completion_lengths)} lengths."
            )

        round_records: List[Dict[str, Any]] = []
        for item_index, data_item in enumerate(all_data):
            record = {
                key: value for key, value in data_item.items() if key != "formatted_prompt"
            }
            record.update(
                {
                    "response": responses[item_index],
                    "completion_length": completion_lengths[item_index],
                    "model": args.model,
                    "model_name": model_name,
                    "run_timestamp": timestamp,
                    "round": round_index,
                    "round_seed": round_seed,
                }
            )
            round_records.append(grade_ood_record(record))

        records_by_dataset = _group_records_by_dataset(round_records)
        by_dataset_metrics: Dict[str, Dict[str, Any]] = {}

        for dataset in dataset_order:
            dataset_records = records_by_dataset.get(dataset, [])
            run_dir = dataset_outputs_root / dataset / model_name / timestamp
            jsonl_dir = run_dir / "jsonls"
            csv_dir = run_dir / "csvs"
            stats_dir = run_dir / "stats"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            csv_dir.mkdir(parents=True, exist_ok=True)
            stats_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(jsonl_dir / f"{shared_filename}-round{round_index}.jsonl", dataset_records)
            write_csv(csv_dir / f"{shared_filename}-round{round_index}.csv", dataset_records)

            dataset_metrics = compute_dataset_round_metrics(dataset, dataset_records)
            by_dataset_metrics[dataset] = dataset_metrics
            source_round_stats = {
                "round": round_index,
                "source_dataset": dataset,
                "source_prompts_paths": paths_of_dataset(dataset),
                "primary_metric_name": PRIMARY_METRIC_BY_DATASET[dataset],
                "metrics": dataset_metrics,
            }
            dataset_round_stats[dataset].append(source_round_stats)
            (stats_dir / f"{shared_filename}-stats_round{round_index}.json").write_text(
                json.dumps(source_round_stats, indent=2),
                encoding="utf-8",
            )

        round_stats = {
            "round": round_index,
            "signature": _signature(args),
            "base_seed": args.sampling_seed,
            "round_seed": round_seed,
            "metrics": compute_integrated_round_metrics(
                round_records,
                by_dataset_metrics,
                dataset_order,
            ),
            "by_dataset": by_dataset_metrics,
            "primary_metric_by_dataset": {
                dataset: get_primary_metric_value(dataset, by_dataset_metrics[dataset])
                for dataset in dataset_order
            },
            "sources": [asdict(source) for source in sources],
            "source_counts": source_counts,
        }
        all_round_stats.append(round_stats)

        (integrated_stats_dir / f"{shared_filename}-integrated-stats_round{round_index}.json").write_text(
            json.dumps(round_stats, indent=2),
            encoding="utf-8",
        )
        (integrated_mds_dir / f"{shared_filename}-round{round_index}.md").write_text(
            render_per_round_markdown(
                model_name=model_name,
                round_index=round_index,
                by_dataset=by_dataset_metrics,
                shared_filename=shared_filename,
                dataset_order=dataset_order,
            ),
            encoding="utf-8",
        )

        print(f"\n=== Unified OOD Round {round_index} Stats ===")
        print(json.dumps(round_stats, indent=2))
        print(flush=True)

    cross_round_values = summarize_cross_round_primary_metrics(all_round_stats, dataset_order)
    (integrated_stats_dir / f"{shared_filename}-integrated-stats.json").write_text(
        json.dumps(
            {
                "signature": _signature(args),
                "base_seed": args.sampling_seed,
                "model": args.model,
                "model_name": model_name,
                "dataset_order": dataset_order,
                "primary_metric_by_dataset": {
                    dataset: PRIMARY_METRIC_BY_DATASET[dataset] for dataset in dataset_order
                },
                "sources": [asdict(source) for source in sources],
                "source_counts": source_counts,
                "global_average": _mean_numeric_metrics(all_round_stats),
                "cross_round_primary_metrics": cross_round_values,
                "rounds": all_round_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (integrated_mds_dir / f"{shared_filename}-cross_round.md").write_text(
        render_cross_round_markdown(
            model_name=model_name,
            cross_round_values=cross_round_values,
            shared_filename=shared_filename,
            rounds=rounds_run,
            dataset_order=dataset_order,
        ),
        encoding="utf-8",
    )

    for dataset in dataset_order:
        stats_dir = dataset_outputs_root / dataset / model_name / timestamp / "stats"
        (stats_dir / f"{shared_filename}-stats.json").write_text(
            json.dumps(
                {
                    "signature": _signature(args),
                    "base_seed": args.sampling_seed,
                    "source_dataset": dataset,
                    "source_prompts_paths": paths_of_dataset(dataset),
                    "primary_metric_name": PRIMARY_METRIC_BY_DATASET[dataset],
                    "global_average": _mean_numeric_metrics(dataset_round_stats[dataset]),
                    "rounds": dataset_round_stats[dataset],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    total_time_s = time.time() - eval_start_time
    print("\nUnified OOD evaluation complete.", flush=True)
    print(f"Dataset-specific outputs: {dataset_outputs_root}", flush=True)
    print(f"Integrated stats: {integrated_stats_dir}", flush=True)
    print(f"Integrated markdown: {integrated_mds_dir}", flush=True)
    print(f"Total time: {total_time_s:.2f}s ({total_time_s / 60:.2f} min)", flush=True)
