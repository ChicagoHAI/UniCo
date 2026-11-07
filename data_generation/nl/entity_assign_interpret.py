"""NL pipeline — v2 step 1: Combined entity assignment + interpretation.

Loads a triplet corpus and reference passages, builds v2 step 1 prompts
(DAG + CPTs + passage), calls an LLM to jointly assign entities and
interpret binary values, extracts JSON, and saves intermediate results
with incremental checkpointing.

Shared across all query types (ATE, CATE, JATE).

Usage (from the repository root)::

    python3 -m data_generation.nl.entity_assign_interpret \
        --triplet_corpus_path <path.jsonl> \
        --passage_corpus_path <path.jsonl> \
        --output_path <output.jsonl> \
        --model gpt-5.4-mini \
        --provider openai \
        ...
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from data_generation.dataset_utils.nl_prompt_builder import (
    build_dag,
    build_v2_step1_prompt,
)
from data_generation.dataset_utils.nl_pipeline_shared import (
    extract_json_block,
    validate_entity_interp,
    load_processed_indices,
    append_records,
    load_jsonl,
    sample_passage,
)
from data_generation.dataset_utils.nl_query_specs import is_v2_2_query_type
from shared_utils.api_utils_multi_provider import (
    batch_query_with_api,
    validate_api_key,
)


def process_chunk(
    chunk_records: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    query_type_offset: int,
    model: str,
    provider: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    num_processes: int,
    max_retries: int,
    api_retries: int,
    api_retry_sleep: float,
    query_type: str | None = None,
) -> List[Dict[str, Any]]:
    """Process a chunk through combined entity assignment + interpretation."""
    pending = list(range(len(chunk_records)))
    prompts: List[str] = []
    passage_info: List[Dict[str, Any]] = []

    for rec in chunk_records:
        rec_query_type = query_type or rec.get("query_type")
        if rec_query_type is not None and not is_v2_2_query_type(rec_query_type):
            rec_query_type = None
        dag = build_dag(rec["dag_edges"], rec["num_nodes"])
        p = sample_passage(passages, rec["example_seed"], query_type_offset)
        prompts.append(build_v2_step1_prompt(
            dag,
            rec["cpts"],
            rec["decimal_places"],
            p["text_content"],
            query_type=rec_query_type,
        ))
        passage_info.append({
            "passage_idx": p["idx"],
            "passage_source": p["source"],
            "passage_text": p["text_content"],
        })

    results: List[Dict[str, Any]] = [None] * len(chunk_records)  # type: ignore[list-item]
    attempt = 0

    while pending and attempt < max_retries:
        attempt += 1
        batch_prompts = [prompts[i] for i in pending]

        responses, _ = batch_query_with_api(
            batch_prompts,
            model_name=model,
            provider=provider,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            num_processes=min(num_processes, len(batch_prompts)),
            api_retries=api_retries,
            api_retry_sleep_seconds=api_retry_sleep,
        )

        still_pending = []
        for local_idx, global_idx in enumerate(pending):
            raw_response = responses[local_idx]
            entity_interpretation = extract_json_block(raw_response)

            if (entity_interpretation is not None
                    and validate_entity_interp(
                        entity_interpretation,
                        chunk_records[global_idx]["num_nodes"],
                        query_type or chunk_records[global_idx].get("query_type"),
                        chunk_records[global_idx]["cpts"],
                    )):
                rec = chunk_records[global_idx]
                pinfo = passage_info[global_idx]
                results[global_idx] = {
                    "idx": rec["idx"],
                    "num_nodes": rec["num_nodes"],
                    "example_seed": rec["example_seed"],
                    "passage_idx": pinfo["passage_idx"],
                    "passage_source": pinfo["passage_source"],
                    "passage_text": pinfo["passage_text"],
                    "entity_interpretation": entity_interpretation,
                    "raw_response": raw_response,
                    "model": model,
                    "provider": provider,
                    "retry_count": attempt - 1,
                    "status": "success",
                    "failure_reason": None,
                }
            else:
                still_pending.append(global_idx)

        pending = still_pending

    for global_idx in pending:
        rec = chunk_records[global_idx]
        pinfo = passage_info[global_idx]
        results[global_idx] = {
            "idx": rec["idx"],
            "num_nodes": rec["num_nodes"],
            "example_seed": rec["example_seed"],
            "passage_idx": pinfo["passage_idx"],
            "passage_source": pinfo["passage_source"],
            "passage_text": pinfo["passage_text"],
            "entity_interpretation": None,
            "raw_response": "",
            "model": model,
            "provider": provider,
            "retry_count": max_retries,
            "status": "failed",
            "failure_reason": "json_extraction_failed",
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NL v2 step 1: combined entity assignment + interpretation",
    )
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--passage_corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--api_retries", type=int, default=5)
    parser.add_argument("--api_retry_sleep", type=float, default=10.0)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--query_type_offset", type=int, default=0)
    parser.add_argument("--query_type", type=str, default=None,
                        help="Explicit query type for query-aware prompting")
    args = parser.parse_args()

    print("=== NL v2 Step 1: Combined Entity Assignment + Interpretation ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    validate_api_key(args.provider)

    print("\nLoading triplet corpus...")
    all_records = load_jsonl(args.triplet_corpus_path)
    print(f"  Loaded {len(all_records)} triplet records")

    print("Loading passage corpus...")
    passages = load_jsonl(args.passage_corpus_path)
    print(f"  Loaded {len(passages)} passages")

    end_idx = args.end_idx if args.end_idx >= 0 else max(r["idx"] for r in all_records) + 1
    records = [r for r in all_records if args.start_idx <= r["idx"] < end_idx]
    print(f"  Processing idx range [{args.start_idx}, {end_idx}): {len(records)} records")

    processed = load_processed_indices(args.output_path)
    records = [r for r in records if r["idx"] not in processed]
    print(f"  After resume filter: {len(records)} remaining")

    if not records:
        print("Nothing to process. Done.")
        return

    output_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = args.checkpoint_interval
    num_chunks = (len(records) + chunk_size - 1) // chunk_size
    num_success = 0
    num_failed = 0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(records))
        chunk = records[start:end]

        print(f"\nChunk {chunk_idx + 1}/{num_chunks}: "
              f"processing {len(chunk)} records...")

        results = process_chunk(
            chunk, passages, args.query_type_offset,
            args.model, args.provider,
            args.max_completion_tokens, args.reasoning_effort,
            args.num_processes, args.max_retries,
            args.api_retries, args.api_retry_sleep,
            args.query_type,
        )

        append_records(args.output_path, results)

        chunk_success = sum(1 for r in results if r["status"] == "success")
        chunk_failed = len(results) - chunk_success
        num_success += chunk_success
        num_failed += chunk_failed
        print(f"  Saved: {chunk_success} success, {chunk_failed} failed")

    print(f"\n=== Done ===")
    print(f"  Total: {num_success} success, {num_failed} failed")
    print(f"  Output: {args.output_path}")


if __name__ == "__main__":
    main()
