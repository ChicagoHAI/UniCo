"""Generic v2.2 NL question generation for association/counterfactual queries."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from data_generation.dataset_utils.nl_pipeline_shared import (
    NL_ANSWER_FORMAT_SENTENCE,
    append_records,
    clean_question_response,
    ensure_intermediate_dirs,
    ensure_output_dirs,
    has_symbolic_leakage,
    load_jsonl,
    load_processed_indices,
    write_nl_outputs,
)
from data_generation.dataset_utils.nl_post_validator import (
    _check_numeric_coverage,
    _extract_scm_values,
)
from data_generation.dataset_utils.nl_prompt_builder import (
    build_question_gen_prompt,
)
from data_generation.dataset_utils.nl_query_specs import get_query_spec
from data_generation.dataset_utils.nl_triplet_adapters import (
    build_metadata,
    build_symbolic_question_text_from_triplet,
    extract_answer,
    get_condition_nodes,
)
from shared_utils.api_utils_multi_provider import (
    batch_query_with_api,
    validate_api_key,
)


ManifestIndex = Dict[Tuple[int, int, str], Set[int]]


def _load_manifest_indices(manifest_path: str, split: str) -> ManifestIndex:
    """Return selected per-node idxs grouped by (num_nodes, dp, variant)."""
    result: ManifestIndex = {}
    position_by_nodes: Dict[int, int] = {}
    entries = []

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    for entry in sorted(entries, key=lambda e: e["triplet_idx"]):
        num_nodes = int(entry["num_nodes"])
        per_nodes_idx = position_by_nodes.get(num_nodes, 0)
        position_by_nodes[num_nodes] = per_nodes_idx + 1

        if split == "train":
            if entry.get("domain_assignment") != "nl":
                continue
        elif not entry.get("selected", False):
            continue

        key = (
            num_nodes,
            int(entry["decimal_places"]),
            entry["condition_variant"],
        )
        result.setdefault(key, set()).add(per_nodes_idx)

    return result


def _parse_seed_from_path(path: str) -> str:
    match = re.search(r"-seed(\d+)", os.path.basename(path))
    return match.group(1) if match else "unknown"


def _final_filename(
    *,
    run_timestamp: str,
    start_idx: int,
    end_idx: int,
    query_type: str,
    num_nodes: int,
    decimal_places: int,
    seed: str,
    variant_assignment: str,
    config_fingerprint: str,
) -> str:
    cfg_suffix = f"-cfg{config_fingerprint}" if config_fingerprint else ""
    return (
        f"{run_timestamp}"
        f"-start{start_idx}-end{end_idx}"
        f"-qt{query_type}"
        f"-n{num_nodes}"
        f"-dp{decimal_places}"
        f"-sd{seed}"
        f"-va_{variant_assignment}"
        f"{cfg_suffix}"
    )


def _build_prompt_for_variant(
    triplet: Dict[str, Any],
    entity_interp: Dict[str, Any],
    query_type: str,
    variant_name: str,
) -> str:
    symbolic_question = build_symbolic_question_text_from_triplet(
        triplet,
        query_type,
        variant_name,
    )
    entity_json = json.dumps(
        entity_interp["entity_interpretation"],
        indent=2,
        ensure_ascii=False,
    )
    return build_question_gen_prompt(
        symbolic_question,
        entity_json,
        variant_name=variant_name,
        query_type=query_type,
    )


def _compute_numeric_coverage(
    nl_question: str,
    triplet: Dict[str, Any],
    query_type: str,
    variant_name: str,
) -> float:
    condition_nodes = get_condition_nodes(triplet, query_type, variant_name)
    scm_values = _extract_scm_values(
        triplet["cpts"],
        condition_nodes,
        triplet.get("decimal_places"),
    )
    return _check_numeric_coverage(nl_question, scm_values)


def process_variant_chunk(
    chunk_triplets: List[Dict[str, Any]],
    entity_interp_map: Dict[int, Dict[str, Any]],
    query_type: str,
    variant_name: str,
    model: str,
    provider: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    num_processes: int,
    max_retries: int,
    api_retries: int,
    api_retry_sleep: float,
    pipeline_version: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pending = list(range(len(chunk_triplets)))
    prompts = [
        _build_prompt_for_variant(
            rec,
            entity_interp_map[rec["idx"]],
            query_type,
            variant_name,
        )
        for rec in chunk_triplets
    ]
    results: List[Optional[Dict[str, Any]]] = [None] * len(chunk_triplets)
    best_candidates: Dict[int, Dict[str, Any]] = {}
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
            nl_question = clean_question_response(raw_response)
            is_non_empty = bool(nl_question)
            has_leakage = has_symbolic_leakage(nl_question) if is_non_empty else True
            no_leakage = not has_leakage

            if is_non_empty:
                rec = chunk_triplets[global_idx]
                coverage = _compute_numeric_coverage(
                    nl_question,
                    rec,
                    query_type,
                    variant_name,
                )
            else:
                coverage = -1.0

            candidate_rank = (int(is_non_empty), int(no_leakage), coverage)
            if not is_non_empty:
                candidate_reason = "empty"
            elif has_leakage:
                candidate_reason = "symbolic_leakage"
            elif coverage < 1.0:
                candidate_reason = "low_numeric_coverage"
            else:
                candidate_reason = None

            prev = best_candidates.get(global_idx)
            if prev is None or candidate_rank > prev["rank"]:
                best_candidates[global_idx] = {
                    "question": nl_question,
                    "numeric_coverage": coverage if coverage >= 0 else 0.0,
                    "failure_reason": candidate_reason or "low_numeric_coverage",
                    "rank": candidate_rank,
                }

            if candidate_reason is None:
                rec = chunk_triplets[global_idx]
                interp = entity_interp_map[rec["idx"]]
                question = nl_question + "\n\n" + NL_ANSWER_FORMAT_SENTENCE
                results[global_idx] = {
                    "idx": rec["idx"],
                    "question": question,
                    "answer": extract_answer(rec, query_type),
                    "metadata": build_metadata(
                        rec,
                        query_type,
                        variant_name,
                        pipeline_version=pipeline_version,
                        entity_interp=interp,
                        question_model=model,
                        question_provider=provider,
                        question_retry_count=attempt - 1,
                    ),
                }
            else:
                still_pending.append(global_idx)

        pending = still_pending

    failed_records = []
    for global_idx in pending:
        rec = chunk_triplets[global_idx]
        best = best_candidates.get(global_idx, {
            "question": "",
            "numeric_coverage": 0.0,
            "failure_reason": "empty",
        })
        failed_records.append({
            "query_type": query_type,
            "idx": rec["idx"],
            "variant_name": variant_name,
            "failure_reason": best["failure_reason"],
            "best_question": best["question"],
            "best_numeric_coverage": best["numeric_coverage"],
            "retry_count": max_retries,
            "model": model,
            "provider": provider,
        })

    return [r for r in results if r is not None], failed_records


def _filter_variant_records(
    records: List[Dict[str, Any]],
    *,
    variant_name: str,
    variant_assignment: str,
    start_idx: int,
    end_idx: int,
    manifest_indices: ManifestIndex | None,
) -> List[Dict[str, Any]]:
    if variant_assignment == "both":
        return records
    if variant_assignment == "split":
        mid_idx = (start_idx + end_idx) // 2
        if variant_name == "full":
            return [r for r in records if r["idx"] < mid_idx]
        return [r for r in records if r["idx"] >= mid_idx]
    if variant_assignment == "manifest":
        if manifest_indices is None:
            raise ValueError("--manifest_path is required with --variant_assignment manifest")
        if not records:
            return []
        first = records[0]
        key = (first["num_nodes"], first["decimal_places"], variant_name)
        selected = manifest_indices.get(key, set())
        return [r for r in records if r["idx"] in selected]
    raise ValueError(f"Unknown variant_assignment: {variant_assignment}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic v2.2 NL question generation step",
    )
    parser.add_argument("--query_type", type=str, required=True)
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--entity_interp_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_timestamp", type=str, required=True)
    parser.add_argument("--pipeline_version", type=str, required=True,
                        choices=["v2", "v3"])
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--provider", type=str, default="openrouter")
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--api_retries", type=int, default=5)
    parser.add_argument("--api_retry_sleep", type=float, default=10.0)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--variant_assignment", type=str, default="manifest",
                        choices=["split", "both", "manifest"])
    parser.add_argument("--config_fingerprint", type=str, default="")
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "eval"])
    args = parser.parse_args()

    spec = get_query_spec(args.query_type)

    print("=== Generic v2.2 NL Question Generation ===")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    validate_api_key(args.provider)

    all_triplets = load_jsonl(args.triplet_corpus_path)
    print(f"\nLoaded {len(all_triplets)} triplet records")

    interp_records = load_jsonl(args.entity_interp_path)
    entity_interp_map = {
        r["idx"]: r for r in interp_records if r.get("status") == "success"
    }
    print(f"Loaded {len(entity_interp_map)} successful interpretations")

    end_idx = (
        args.end_idx if args.end_idx >= 0
        else max(r["idx"] for r in all_triplets) + 1
    )
    records = [
        r for r in all_triplets
        if args.start_idx <= r["idx"] < end_idx
        and r["idx"] in entity_interp_map
    ]
    print(f"Processing {len(records)} records")

    manifest_indices = None
    if args.manifest_path:
        manifest_indices = _load_manifest_indices(args.manifest_path, args.split)

    ensure_output_dirs(args.output_dir, list(spec.variants))
    ensure_intermediate_dirs(args.output_dir)

    num_nodes = all_triplets[0]["num_nodes"] if all_triplets else 0
    decimal_places = all_triplets[0]["decimal_places"] if all_triplets else 0
    seed = _parse_seed_from_path(args.triplet_corpus_path)
    final_filename = _final_filename(
        run_timestamp=args.run_timestamp,
        start_idx=args.start_idx,
        end_idx=end_idx,
        query_type=args.query_type,
        num_nodes=num_nodes,
        decimal_places=decimal_places,
        seed=seed,
        variant_assignment=args.variant_assignment,
        config_fingerprint=args.config_fingerprint,
    )
    checkpoint_basename = args.checkpoint_name or final_filename

    all_variant_records: Dict[str, List[Dict[str, Any]]] = {}

    for variant_name in spec.variants:
        print(f"\n--- Variant: {variant_name} ---")
        variant_input = _filter_variant_records(
            records,
            variant_name=variant_name,
            variant_assignment=args.variant_assignment,
            start_idx=args.start_idx,
            end_idx=end_idx,
            manifest_indices=manifest_indices,
        )
        print(f"  Variant input: {len(variant_input)} records")

        checkpoint_path = os.path.join(
            args.output_dir,
            "intermediates-nl",
            f"{checkpoint_basename}-{variant_name}-checkpoint.jsonl",
        )
        processed = load_processed_indices(checkpoint_path)
        variant_records = [r for r in variant_input if r["idx"] not in processed]
        print(f"  Remaining after resume: {len(variant_records)}")

        all_failed: List[Dict[str, Any]] = []
        if variant_records:
            chunk_size = args.checkpoint_interval
            num_chunks = (len(variant_records) + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(variant_records))
                chunk = variant_records[start:end]
                print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {len(chunk)} records...")
                success, failed = process_variant_chunk(
                    chunk,
                    entity_interp_map,
                    args.query_type,
                    variant_name,
                    args.model,
                    args.provider,
                    args.max_completion_tokens,
                    args.reasoning_effort,
                    args.num_processes,
                    args.max_retries,
                    args.api_retries,
                    args.api_retry_sleep,
                    args.pipeline_version,
                )
                if success:
                    append_records(checkpoint_path, success)
                all_failed.extend(failed)
                print(f"    {len(success)} success, {len(failed)} failed")

        all_records_for_variant = (
            load_jsonl(checkpoint_path) if os.path.isfile(checkpoint_path) else []
        )
        all_variant_records[variant_name] = all_records_for_variant
        print(f"  Variant {variant_name}: {len(all_records_for_variant)} total")

        if all_failed:
            fail_path = os.path.join(
                args.output_dir,
                "intermediates-nl",
                f"{final_filename}-{variant_name}-failures.jsonl",
            )
            with open(fail_path, "w") as f:
                for rec in all_failed:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Failures logged to: {fail_path}")

    print("\nWriting final outputs...")
    write_nl_outputs(
        args.output_dir,
        final_filename,
        all_variant_records,
        num_nodes=num_nodes,
        triplet_corpus_path=args.triplet_corpus_path,
        pipeline_version=args.pipeline_version,
        variant_names=list(spec.variants),
    )
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
