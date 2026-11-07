"""NL pipeline — v3 step 2: Entity interpretation.

Loads a triplet corpus and entity assignment output (from v3 step 1),
builds v3 step 2 prompts (DAG + CPTs + entity mapping), calls an LLM to
interpret binary node values under commonsense, extracts JSON, and saves
intermediate results with incremental checkpointing.

Shared across all query types (ATE, CATE, JATE).

Usage (from the repository root)::

    python3 -m data_generation.nl.entity_interpret \
        --triplet_corpus_path <path.jsonl> \
        --entity_assign_path <step1_output.jsonl> \
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
    build_v3_step2_prompt,
)
from data_generation.dataset_utils.nl_pipeline_shared import (
    extract_json_block,
    validate_entity_interp,
    load_processed_indices,
    append_records,
    load_jsonl,
)
from data_generation.dataset_utils.nl_query_specs import is_v2_2_query_type
from shared_utils.api_utils_multi_provider import (
    batch_query_with_api,
    validate_api_key,
)


def process_chunk(
    chunk_triplets: List[Dict[str, Any]],
    entity_assign_map: Dict[int, Dict[str, Any]],
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
    """Process a chunk through entity interpretation."""
    pending = list(range(len(chunk_triplets)))
    prompts: List[str] = []

    for rec in chunk_triplets:
        dag = build_dag(rec["dag_edges"], rec["num_nodes"])
        assign_rec = entity_assign_map[rec["idx"]]
        rec_query_type = query_type or rec.get("query_type")
        if rec_query_type is not None and not is_v2_2_query_type(rec_query_type):
            rec_query_type = None
        entity_mapping_json = json.dumps(
            assign_rec["entity_mapping"], indent=2, ensure_ascii=False,
        )
        prompts.append(build_v3_step2_prompt(
            dag,
            rec["cpts"],
            rec["decimal_places"],
            entity_mapping_json,
            query_type=rec_query_type,
        ))

    results: List[Dict[str, Any]] = [None] * len(chunk_triplets)  # type: ignore[list-item]
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
                        chunk_triplets[global_idx]["num_nodes"],
                        query_type or chunk_triplets[global_idx].get("query_type"),
                        chunk_triplets[global_idx].get("cpts"),
                    )):
                rec = chunk_triplets[global_idx]
                assign_rec = entity_assign_map[rec["idx"]]
                results[global_idx] = {
                    "idx": rec["idx"],
                    "num_nodes": rec["num_nodes"],
                    "example_seed": rec["example_seed"],
                    "passage_idx": assign_rec["passage_idx"],
                    "passage_source": assign_rec["passage_source"],
                    "passage_text": assign_rec["passage_text"],
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
        rec = chunk_triplets[global_idx]
        assign_rec = entity_assign_map[rec["idx"]]
        results[global_idx] = {
            "idx": rec["idx"],
            "num_nodes": rec["num_nodes"],
            "example_seed": rec["example_seed"],
            "passage_idx": assign_rec["passage_idx"],
            "passage_source": assign_rec["passage_source"],
            "passage_text": assign_rec["passage_text"],
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
        description="NL v3 step 2: entity interpretation",
    )
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--entity_assign_path", type=str, required=True,
                        help="Output from v3 step 1 (entity assignment)")
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
    parser.add_argument("--query_type", type=str, default=None,
                        help="Explicit query type for query-aware prompting")
    args = parser.parse_args()

    print("=== NL v3 Step 2: Entity Interpretation ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    validate_api_key(args.provider)

    print("\nLoading triplet corpus...")
    all_triplets = load_jsonl(args.triplet_corpus_path)
    print(f"  Loaded {len(all_triplets)} triplet records")

    print("Loading entity assignment results...")
    assign_records = load_jsonl(args.entity_assign_path)
    entity_assign_map = {r["idx"]: r for r in assign_records if r["status"] == "success"}
    print(f"  Loaded {len(entity_assign_map)} successful assignments "
          f"(out of {len(assign_records)} total)")

    # Subset
    end_idx = args.end_idx if args.end_idx >= 0 else max(r["idx"] for r in all_triplets) + 1
    records = [
        r for r in all_triplets
        if args.start_idx <= r["idx"] < end_idx and r["idx"] in entity_assign_map
    ]
    print(f"  Processing {len(records)} records with successful assignments")

    # Resume
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
            chunk, entity_assign_map,
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
