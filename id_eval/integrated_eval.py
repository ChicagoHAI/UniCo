"""Prompt loading, subsampling, metrics, and reports for the unified ID evaluation.

All prompt files are concatenated into one vLLM workload per round. Results are then split
back into per-file artifacts and integrated reports with one table per representation form.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from id_eval.grading import extract_and_grade_prediction
from shared_utils.vllm_backends import infer_model_dir_name

ASSOCIATION_QUERY_TYPES = (
    "conditional_prob",
    "joint_prob",
    "marginal_prob",
    "observed_difference",
    "independence_test",
    "markov_blanket",
)
INTERVENTION_QUERY_TYPES = (
    "ate",
    "cate",
    "jate",
    "identification",
    "backdoor",
    "frontdoor",
)
COUNTERFACTUAL_QUERY_TYPES = (
    "cf",
    "att",
    "nde",
    "nie",
    "pn",
    "ps",
)
QUERY_TYPE_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("association", ASSOCIATION_QUERY_TYPES),
    ("intervention", INTERVENTION_QUERY_TYPES),
    ("counterfactual", COUNTERFACTUAL_QUERY_TYPES),
)

DOMAINS = ("symbolic", "code", "nl")
TABLE_SCOPES = ("all_domains", *DOMAINS)
NO_PROMPT_LIMIT = int(1e18)

FormatPromptFn = Callable[[str], str]
BatchQueryFn = Callable[[Any, List[str], Any, Optional[int]], Tuple[List[str], List[int]]]


@dataclass(frozen=True)
class PromptSource:
    source_index: int
    prompts_path: str
    dataset_name: str
    domain: str
    query_type: str


def _strip_dataset_stem(stem: str) -> Tuple[str, str]:
    """Parse ``[<timestamp>-]<query_type>[-code|-nl]-eval[-subsetN]`` into query type and domain.

    Files downloaded from Hugging Face omit the timestamp; freshly generated files include it.
    """
    stem = re.sub(r"-subset\d+$", "", stem)
    stem = re.sub(r"^\d{8}_\d{6}-", "", stem)

    if not stem.endswith("-eval"):
        raise ValueError(f"Cannot infer query type/domain from filename stem {stem!r}")
    stem = stem[: -len("-eval")]

    if stem.endswith("-code"):
        return stem[: -len("-code")], "code"
    if stem.endswith("-nl"):
        return stem[: -len("-nl")], "nl"
    return stem, "symbolic"


def infer_query_type_and_domain(
    prompts_path: str,
    first_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Read the query type and domain from metadata, falling back to the filename.

    Symbolic records of several query types carry neither field, so the filename is required.
    """
    stem_query_type, stem_domain = _strip_dataset_stem(Path(prompts_path).stem)
    metadata = first_metadata or {}
    query_type = str(metadata.get("query_type") or stem_query_type)
    domain = str(metadata.get("domain") or stem_domain)
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain {domain!r} inferred from {prompts_path}")
    return query_type, domain


def _read_first_json_record(path: str) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def build_prompt_sources(prompts_path_list: Sequence[str]) -> List[PromptSource]:
    """Validate the prompt files and describe each one; names must be unique across the suite."""
    dataset_names: Dict[str, str] = {}
    sources: List[PromptSource] = []

    for source_index, prompts_path in enumerate(prompts_path_list):
        if not os.path.isfile(prompts_path):
            raise FileNotFoundError(
                f"Prompt file not found: {prompts_path}. "
                "See README.md for how to download the UniCo evaluation data."
            )

        first_record = _read_first_json_record(prompts_path)
        if first_record is None:
            raise ValueError(f"Prompt file is empty: {prompts_path}")
        metadata = first_record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"First record in {prompts_path} has no metadata object")

        dataset_name = Path(prompts_path).stem
        if dataset_name in dataset_names:
            raise ValueError(
                f"Duplicate dataset name {dataset_name!r} for both "
                f"{dataset_names[dataset_name]!r} and {prompts_path!r}."
            )
        dataset_names[dataset_name] = prompts_path

        query_type, domain = infer_query_type_and_domain(prompts_path, metadata)
        sources.append(
            PromptSource(
                source_index=source_index,
                prompts_path=prompts_path,
                dataset_name=dataset_name,
                domain=domain,
                query_type=query_type,
            )
        )
    return sources


def _make_prompt_entry(
    *,
    source: PromptSource,
    ex: Dict[str, Any],
    format_prompt: FormatPromptFn,
    global_idx: int,
) -> Dict[str, Any]:
    for required_field in ("idx", "question", "answer", "metadata"):
        if required_field not in ex:
            raise ValueError(f"Missing required field {required_field!r} in {source.prompts_path}")

    metadata = ex["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Record in {source.prompts_path} has no metadata object")
    for required_field in ("variant_name", "is_abnormal", "num_nodes"):
        if required_field not in metadata:
            raise ValueError(
                f"Missing required metadata field {required_field!r} in {source.prompts_path}"
            )

    raw_prompt = ex["question"]
    return {
        "global_idx": global_idx,
        "source_index": source.source_index,
        "source_prompts_path": source.prompts_path,
        "source_dataset_name": source.dataset_name,
        "source_domain": source.domain,
        "source_query_type": source.query_type,
        "idx": ex["idx"],
        "question": raw_prompt,
        "formatted_prompt": format_prompt(raw_prompt),
        "answer": ex["answer"],
        "query_type": metadata.get("query_type", source.query_type),
        "variant_name": metadata["variant_name"],
        "decimal_places": metadata.get("decimal_places"),
        "is_abnormal": metadata["is_abnormal"],
        "query_format": metadata.get("query_format"),
        "has_naive_conditioning": metadata.get("has_naive_conditioning"),
        "num_nodes": metadata["num_nodes"],
    }


def load_integrated_prompts(
    sources: Sequence[PromptSource],
    format_prompt: FormatPromptFn,
    max_prompts_per_file: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load every prompt file into one list, up to ``max_prompts_per_file`` records per file."""
    all_data: List[Dict[str, Any]] = []
    limit = max_prompts_per_file if max_prompts_per_file is not None else NO_PROMPT_LIMIT

    for source in sources:
        num_loaded_for_source = 0
        with open(source.prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if num_loaded_for_source >= limit:
                    break
                all_data.append(
                    _make_prompt_entry(
                        source=source,
                        ex=json.loads(line),
                        format_prompt=format_prompt,
                        global_idx=len(all_data),
                    )
                )
                num_loaded_for_source += 1

        print(
            f"Loaded {num_loaded_for_source} examples from {source.prompts_path} "
            f"({source.domain}, {source.query_type})",
            flush=True,
        )

    print(f"\nLoaded {len(all_data)} integrated examples from {len(sources)} files.\n")
    return all_data


def compute_round_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy overall and by variant/decimals, abnormality, naivety, and graph size."""
    if not records:
        return {
            "total": 0,
            "extraction_rate": 0.0,
            "overall_accuracy": 0.0,
            "avg_completion_length": 0.0,
        }

    total = len(records)
    scores = [float(r["score"]) for r in records]
    num_extracted = sum(1 for r in records if r.get("extraction_success", False))
    completion_lengths = [int(r.get("completion_length", 0)) for r in records]

    metrics: Dict[str, Any] = {
        "total": total,
        "extraction_rate": num_extracted / total,
        "avg_completion_length": mean(completion_lengths) if completion_lengths else 0.0,
        "overall_accuracy": mean(scores) if scores else 0.0,
    }

    combo_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        decimal_places = (
            record["decimal_places"] if record.get("decimal_places") is not None else "na"
        )
        combo_groups[f"{record['variant_name']}_dp{decimal_places}"].append(record)

    for key, group in sorted(combo_groups.items()):
        extracted = sum(1 for r in group if r.get("extraction_success", False))
        metrics[f"{key}_accuracy"] = mean(float(r["score"]) for r in group)
        metrics[f"{key}_count"] = len(group)
        metrics[f"{key}_extraction_rate"] = extracted / len(group)

    for abnormal_val in (True, False):
        label = "abnormal" if abnormal_val else "normal"
        group = [r for r in records if r.get("is_abnormal") == abnormal_val]
        if group:
            metrics[f"{label}_accuracy"] = mean(float(r["score"]) for r in group)
            metrics[f"{label}_count"] = len(group)

    for naive_val in (True, False):
        label = "naive" if naive_val else "not_naive"
        group = [r for r in records if r.get("has_naive_conditioning") == naive_val]
        if group:
            metrics[f"{label}_accuracy"] = mean(float(r["score"]) for r in group)
            metrics[f"{label}_count"] = len(group)

    nodes_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        nodes_groups[int(record["num_nodes"])].append(record)
    for num_nodes, group in sorted(nodes_groups.items()):
        metrics[f"nodes{num_nodes}_accuracy"] = mean(float(r["score"]) for r in group)
        metrics[f"nodes{num_nodes}_count"] = len(group)

    return metrics


def iter_query_types_for_tables() -> List[str]:
    """List query types in table order: association, intervention, counterfactual."""
    query_types: List[str] = []
    for _, group_query_types in QUERY_TYPE_GROUPS:
        query_types.extend(group_query_types)
    return query_types


def _select_records(
    records: Sequence[Dict[str, Any]],
    domain: Optional[str],
    query_type: Optional[str],
) -> List[Dict[str, Any]]:
    selected = list(records)
    if domain is not None:
        selected = [r for r in selected if r.get("source_domain") == domain]
    if query_type is not None:
        selected = [r for r in selected if r.get("source_query_type") == query_type]
    return selected


def compute_group_accuracy(
    records: Sequence[Dict[str, Any]],
    domain: Optional[str] = None,
    query_type: Optional[str] = None,
) -> Optional[float]:
    """Mean score of one report cell, or None when the cell has no records."""
    selected = _select_records(records, domain, query_type)
    if not selected:
        return None
    return mean(float(r["score"]) for r in selected)


def compute_group_count(
    records: Sequence[Dict[str, Any]],
    domain: Optional[str] = None,
    query_type: Optional[str] = None,
) -> int:
    """Count the records in one report cell."""
    return len(_select_records(records, domain, query_type))


def compute_round_report_counts(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Count the examples contributing to every report-table cell."""
    counts: Dict[str, Dict[str, int]] = {}
    query_types = iter_query_types_for_tables()
    for scope in TABLE_SCOPES:
        domain = None if scope == "all_domains" else scope
        scope_counts = {
            query_type: compute_group_count(records, domain=domain, query_type=query_type)
            for query_type in query_types
        }
        num_scope_records = compute_group_count(records, domain=domain)
        num_column_records = sum(scope_counts.values())
        if num_column_records != num_scope_records:
            raise ValueError(
                f"{scope} report columns cover {num_column_records} records, but the scope "
                f"contains {num_scope_records}. Check for an unsupported source_query_type."
            )
        scope_counts["micro_avg"] = num_scope_records
        counts[scope] = scope_counts
    return counts


def _weighted_query_type_micro_average(
    values: Dict[str, Optional[float]],
    counts: Dict[str, int],
) -> Optional[float]:
    """Derive one table row's micro average from its query-type cells and their counts."""
    weighted_score_sum = 0.0
    num_examples_total = 0
    for query_type in iter_query_types_for_tables():
        num_examples = counts.get(query_type, 0)
        value = values.get(query_type)
        if num_examples < 0:
            raise ValueError(f"Example count cannot be negative for {query_type}: {num_examples}")
        if num_examples == 0:
            if value is not None:
                raise ValueError(f"{query_type} has a report value but zero examples.")
            continue
        if value is None:
            raise ValueError(f"{query_type} has {num_examples} examples but no report value.")
        weighted_score_sum += value * num_examples
        num_examples_total += num_examples

    expected_total = counts.get("micro_avg", num_examples_total)
    if expected_total != num_examples_total:
        raise ValueError(
            "Query-type counts do not sum to the micro-average count: "
            f"{num_examples_total} != {expected_total}."
        )
    if num_examples_total == 0:
        return None
    return weighted_score_sum / num_examples_total


def compute_round_report_values(
    records: Sequence[Dict[str, Any]],
    report_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute one round's report cells with a row-consistent weighted micro average."""
    counts = report_counts or compute_round_report_counts(records)
    values: Dict[str, Dict[str, Optional[float]]] = {}
    for scope in TABLE_SCOPES:
        domain = None if scope == "all_domains" else scope
        scope_values: Dict[str, Optional[float]] = {
            query_type: compute_group_accuracy(records, domain=domain, query_type=query_type)
            for query_type in iter_query_types_for_tables()
        }
        scope_values["micro_avg"] = _weighted_query_type_micro_average(
            scope_values,
            counts[scope],
        )
        values[scope] = scope_values
    return values


def summarize_cross_round_values(
    round_values: Sequence[Dict[str, Dict[str, Optional[float]]]],
    report_counts: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Aggregate query-type cells across rounds, then derive each row's weighted micro average."""
    summary: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for scope in TABLE_SCOPES:
        summary[scope] = {}
        for row_name, agg_fn in (
            ("mean", lambda vals: sum(vals) / len(vals)),
            ("max", max),
            ("min", min),
        ):
            row: Dict[str, Optional[float]] = {}
            for column in iter_query_types_for_tables():
                vals = [
                    rv[scope][column]
                    for rv in round_values
                    if rv.get(scope, {}).get(column) is not None
                ]
                row[column] = agg_fn(vals) if vals else None
            row["micro_avg"] = _weighted_query_type_micro_average(row, report_counts[scope])
            summary[scope][row_name] = row
    return summary


def summarize_cross_round_counts(
    round_counts: Sequence[Dict[str, Dict[str, int]]],
) -> Dict[str, Dict[str, int]]:
    """Return the report counts shared by all rounds after checking that they match."""
    if not round_counts:
        raise ValueError("At least one round of report counts is required.")
    expected_counts = round_counts[0]
    for round_offset, counts in enumerate(round_counts[1:], start=2):
        if counts != expected_counts:
            raise ValueError(
                "Report example counts must remain identical across rounds; "
                f"round position {round_offset} differs from round position 1."
            )
    return {scope: dict(expected_counts[scope]) for scope in TABLE_SCOPES}


def write_jsonl(path: str, records: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: str, records: Sequence[Dict[str, Any]]) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def _format_metric(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "NA"


def _table_title(scope: str) -> str:
    if scope == "all_domains":
        return "Query Type Micro Average"
    return f"{scope.capitalize()} Domain"


def render_report_table(
    rows: Sequence[Tuple[str, Dict[str, Optional[float]]]],
    counts: Dict[str, int],
) -> str:
    columns = iter_query_types_for_tables() + ["micro_avg"]
    lines = [
        "| Row | " + " | ".join(columns) + " |",
        "|---|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row_name, values in rows:
        row_cells = [_format_metric(values.get(column)) for column in columns]
        lines.append(f"| {row_name} | " + " | ".join(row_cells) + " |")
    count_cells = [str(counts.get(column, 0)) for column in columns]
    lines.append("| num_examples | " + " | ".join(count_cells) + " |")
    return "\n".join(lines)


def render_per_round_markdown(
    *,
    model_name: str,
    round_index: int,
    values: Dict[str, Dict[str, Optional[float]]],
    counts: Dict[str, Dict[str, int]],
    shared_filename: str,
    sources: Sequence[PromptSource],
) -> str:
    parts = [
        f"# Integrated Evaluation Round {round_index}",
        "",
        f"- Model: `{model_name}`",
        f"- Run: `{shared_filename}`",
        f"- Prompt files: {len(sources)}",
        "",
        "Column order groups query types by causal ladder: association, intervention, counterfactual.",
        "Each `micro_avg` is the record-count-weighted average of the query-type cells in its row; `num_examples` gives the weights.",
        "",
    ]
    for scope in TABLE_SCOPES:
        parts.extend(
            [
                f"## {_table_title(scope)}",
                "",
                render_report_table([(model_name, values[scope])], counts[scope]),
                "",
            ]
        )
    return "\n".join(parts)


def render_cross_round_markdown(
    *,
    model_name: str,
    cross_round_values: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    counts: Dict[str, Dict[str, int]],
    shared_filename: str,
    rounds: Sequence[int],
    sources: Sequence[PromptSource],
) -> str:
    parts = [
        "# Integrated Evaluation Cross-Round Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Run: `{shared_filename}`",
        f"- Rounds: `{', '.join(str(r) for r in rounds)}`",
        f"- Prompt files: {len(sources)}",
        "",
        "Each query-type cell summarizes its per-round values. The `micro_avg` in each row is then recomputed as the record-count-weighted average of that row's query-type cells; `num_examples` gives the shared round weights.",
        "",
    ]
    for scope in TABLE_SCOPES:
        rows = [(name, cross_round_values[scope][name]) for name in ("mean", "max", "min")]
        parts.extend(
            [
                f"## {_table_title(scope)}",
                "",
                render_report_table(rows, counts[scope]),
                "",
            ]
        )
    return "\n".join(parts)


def _mean_numeric_metrics(round_stats: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    numeric_values: Dict[str, List[float]] = defaultdict(list)
    for item in round_stats:
        for key, value in item.get("metrics", {}).items():
            if key == "total" or key.endswith("_count"):
                continue
            if isinstance(value, (int, float)):
                numeric_values[key].append(float(value))
    return {key: sum(values) / len(values) for key, values in numeric_values.items() if values}


def _make_record(
    *,
    data_item: Dict[str, Any],
    response: str,
    completion_length: int,
) -> Dict[str, Any]:
    pred_str, extraction_success, ref_answer, score, grade_details = extract_and_grade_prediction(
        data_item,
        response,
    )
    record: Dict[str, Any] = {
        "global_idx": data_item["global_idx"],
        "source_index": data_item["source_index"],
        "source_prompts_path": data_item["source_prompts_path"],
        "source_dataset_name": data_item["source_dataset_name"],
        "source_domain": data_item["source_domain"],
        "source_query_type": data_item["source_query_type"],
        "idx": data_item["idx"],
        "query_type": data_item["query_type"],
        "variant_name": data_item["variant_name"],
        "decimal_places": data_item["decimal_places"],
        "is_abnormal": data_item["is_abnormal"],
        "has_naive_conditioning": data_item.get("has_naive_conditioning"),
        "num_nodes": data_item["num_nodes"],
        "query_format": data_item.get("query_format"),
        "question": data_item["question"],
        "answer": ref_answer,
        "prediction": pred_str,
        "extraction_success": extraction_success,
        "score": score,
        "response": response,
        "completion_length": completion_length,
    }
    record.update(grade_details)
    return record


def _group_records_by_source(records: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["source_index"])].append(record)
    return grouped


def _source_counts_by_name(
    sources: Sequence[PromptSource],
    all_data: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    counts: Dict[int, int] = defaultdict(int)
    for item in all_data:
        counts[int(item["source_index"])] += 1
    source_by_index = {source.source_index: source for source in sources}
    return {source_by_index[index].dataset_name: count for index, count in counts.items()}


def _signature(args: Any) -> Dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "max_new_tokens": args.max_new_tokens,
        "max_prompts_per_file": args.max_prompts_per_file,
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_integrated_evaluation(
    *,
    args: Any,
    sources: Sequence[PromptSource],
    all_data: Sequence[Dict[str, Any]],
    model: Any,
    batch_query: BatchQueryFn,
) -> None:
    """Generate and grade every round, then write per-file artifacts and integrated reports."""
    eval_start_time = time.time()
    model_name = infer_model_dir_name(args.model)
    timestamp = args.run_timestamp or time.strftime("%Y%m%d_%H%M%S")
    shared_filename = (
        f"{timestamp}-temp{args.temperature}-top_p{args.top_p}"
        f"-top_k{args.top_k}-presence_penalty{args.presence_penalty}"
        f"-max_tokens{args.max_new_tokens}-seed{args.sampling_seed}"
    )

    results_root = args.results_base_dir
    dataset_outputs_root = os.path.join(results_root, "dataset-specific-outputs")
    integrated_model_root = os.path.join(results_root, "integrated-outputs", model_name)
    integrated_stats_dir = os.path.join(integrated_model_root, "stats")
    integrated_mds_dir = os.path.join(integrated_model_root, "mds")
    os.makedirs(integrated_stats_dir, exist_ok=True)
    os.makedirs(integrated_mds_dir, exist_ok=True)

    print("Running integrated in-distribution evaluation.", flush=True)
    print(f"Results root: {results_root}", flush=True)
    print("Prompt sources:", flush=True)
    for source in sources:
        print(f"  - {json.dumps(asdict(source), sort_keys=True)}", flush=True)

    prompts = [item["formatted_prompt"] for item in all_data]
    source_count_names = _source_counts_by_name(sources, all_data)
    signature = _signature(args)

    all_round_stats: List[Dict[str, Any]] = []
    round_report_values: List[Dict[str, Dict[str, Optional[float]]]] = []
    round_report_counts: List[Dict[str, Dict[str, int]]] = []
    rounds_run: List[int] = []
    dataset_round_stats: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for round_index in range(args.start_round, args.budget + 1):
        rounds_run.append(round_index)
        round_seed = round_index * args.sampling_seed
        print(
            f"\n=== Integrated round {round_index}/{args.budget} "
            f"({len(prompts)} prompts, seed={round_seed}) ===",
            flush=True,
        )
        round_start_time = time.time()
        responses, completion_lengths = batch_query(model, prompts, args, round_seed)
        round_duration_s = time.time() - round_start_time
        print(
            f"Round {round_index} generation done in {round_duration_s:.2f}s "
            f"({round_duration_s / len(prompts):.2f}s/prompt)",
            flush=True,
        )
        if len(responses) != len(prompts) or len(completion_lengths) != len(prompts):
            raise ValueError(
                f"Expected {len(prompts)} completions, got {len(responses)} responses "
                f"and {len(completion_lengths)} lengths."
            )

        round_records = [
            _make_record(
                data_item=data_item,
                response=responses[item_index],
                completion_length=completion_lengths[item_index],
            )
            for item_index, data_item in enumerate(all_data)
        ]

        by_source = _group_records_by_source(round_records)
        for source in sources:
            source_records = by_source.get(source.source_index, [])
            run_dir = os.path.join(
                dataset_outputs_root,
                source.dataset_name,
                model_name,
                timestamp,
            )
            jsonl_dir = os.path.join(run_dir, "jsonls")
            csv_dir = os.path.join(run_dir, "csvs")
            stats_dir = os.path.join(run_dir, "stats")
            os.makedirs(jsonl_dir, exist_ok=True)
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

            write_jsonl(
                os.path.join(jsonl_dir, f"{shared_filename}-round{round_index}.jsonl"),
                source_records,
            )
            write_csv(
                os.path.join(csv_dir, f"{shared_filename}-round{round_index}.csv"),
                source_records,
            )

            source_round_stats = {
                "round": round_index,
                "source": asdict(source),
                "metrics": compute_round_metrics(source_records),
            }
            dataset_round_stats[source.source_index].append(source_round_stats)
            _write_json(
                os.path.join(stats_dir, f"{shared_filename}-stats_round{round_index}.json"),
                source_round_stats,
            )

        counts = compute_round_report_counts(round_records)
        values = compute_round_report_values(round_records, report_counts=counts)
        round_report_values.append(values)
        round_report_counts.append(counts)
        round_stats = {
            "round": round_index,
            "signature": signature,
            "base_seed": args.sampling_seed,
            "round_seed": round_seed,
            "metrics": compute_round_metrics(round_records),
            "report_values": values,
            "report_counts": counts,
            "sources": [asdict(source) for source in sources],
            "source_counts": source_count_names,
        }
        all_round_stats.append(round_stats)

        _write_json(
            os.path.join(
                integrated_stats_dir,
                f"{shared_filename}-integrated-stats_round{round_index}.json",
            ),
            round_stats,
        )
        with open(
            os.path.join(integrated_mds_dir, f"{shared_filename}-round{round_index}.md"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                render_per_round_markdown(
                    model_name=model_name,
                    round_index=round_index,
                    values=values,
                    counts=counts,
                    shared_filename=shared_filename,
                    sources=sources,
                )
            )

        print(f"\n=== Integrated Round {round_index} Stats ===")
        print(json.dumps(round_stats, indent=2))
        print(flush=True)

    cross_round_counts = summarize_cross_round_counts(round_report_counts)
    cross_round_values = summarize_cross_round_values(round_report_values, cross_round_counts)
    _write_json(
        os.path.join(integrated_stats_dir, f"{shared_filename}-integrated-stats.json"),
        {
            "signature": signature,
            "base_seed": args.sampling_seed,
            "model": args.model,
            "model_name": model_name,
            "sources": [asdict(source) for source in sources],
            "source_counts": source_count_names,
            "global_average": _mean_numeric_metrics(all_round_stats),
            "cross_round_report_values": cross_round_values,
            "cross_round_report_counts": cross_round_counts,
            "rounds": all_round_stats,
        },
    )
    with open(
        os.path.join(integrated_mds_dir, f"{shared_filename}-cross_round.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            render_cross_round_markdown(
                model_name=model_name,
                cross_round_values=cross_round_values,
                counts=cross_round_counts,
                shared_filename=shared_filename,
                rounds=rounds_run,
                sources=sources,
            )
        )

    for source in sources:
        stats_dir = os.path.join(
            dataset_outputs_root,
            source.dataset_name,
            model_name,
            timestamp,
            "stats",
        )
        _write_json(
            os.path.join(stats_dir, f"{shared_filename}-stats.json"),
            {
                "signature": signature,
                "base_seed": args.sampling_seed,
                "source": asdict(source),
                "global_average": _mean_numeric_metrics(dataset_round_stats[source.source_index]),
                "rounds": dataset_round_stats[source.source_index],
            },
        )

    total_time_s = time.time() - eval_start_time
    print("\nIntegrated evaluation complete.", flush=True)
    print(f"Dataset-specific outputs: {dataset_outputs_root}", flush=True)
    print(f"Integrated stats: {integrated_stats_dir}", flush=True)
    print(f"Integrated markdown: {integrated_mds_dir}", flush=True)
    print(f"Total time: {total_time_s:.2f}s ({total_time_s / 60:.2f} min)", flush=True)
