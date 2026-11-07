"""ATE NL pipeline — question generation step (shared v2 step 2 / v3 step 3).

Loads ATE triplet corpus and entity interpretation results, builds
condition-variant-specific symbolic questions, calls an LLM to generate
natural language questions, validates for symbolic leakage, and produces
final JSONL output per condition variant with checkpointing.

Usage (from the repository root)::

    python3 -m data_generation.ate.ate_nl_step_question_gen \
        --triplet_corpus_path <path.jsonl> \
        --entity_interp_path <interp_output.jsonl> \
        --output_dir <nodes6-prompting_v3> \
        --pipeline_version v3 \
        --model gemini-3-flash-preview \
        --provider openrouter \
        ...
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from data_generation.dataset_utils.nl_prompt_builder import (
    build_dag,
    build_symbolic_question_text,
    build_question_gen_prompt,
    derive_query_variant,
    get_full_condition_nodes,
    get_surgery_ancestor_prune_nodes_ate,
)
from data_generation.dataset_utils.nl_pipeline_shared import (
    NL_ANSWER_FORMAT_SENTENCE,
    NL_VARIANT_NAMES,
    clean_question_response,
    has_symbolic_leakage,
    check_numeric_fidelity,
    load_processed_indices,
    append_records,
    ensure_output_dirs,
    ensure_intermediate_dirs,
    write_nl_outputs,
    load_jsonl,
)
from shared_utils.api_utils_multi_provider import (
    batch_query_with_api,
    validate_api_key,
)




def _build_prompts_for_variant(
    triplet: Dict[str, Any],
    entity_interp: Dict[str, Any],
    variant_name: str,
) -> str:
    """Build the question generation prompt for one (triplet, variant) pair."""
    dag = build_dag(triplet["dag_edges"], triplet["num_nodes"])
    qv = derive_query_variant(triplet["example_seed"])

    if variant_name == "full":
        condition_nodes = get_full_condition_nodes(dag)
    else:
        condition_nodes = get_surgery_ancestor_prune_nodes_ate(
            dag, triplet["treatment"], triplet["outcome"],
        )

    symbolic_question = build_symbolic_question_text(
        dag, triplet["cpts"], triplet["decimal_places"],
        [triplet["treatment"]], triplet["outcome"], {},
        qv, condition_nodes,
    )

    entity_json = json.dumps(
        entity_interp["entity_interpretation"],
        indent=2, ensure_ascii=False,
    )

    return build_question_gen_prompt(
        symbolic_question, entity_json, variant_name=variant_name,
    )


def _compute_numeric_coverage(
    nl_question: str,
    rec: Dict[str, Any],
    variant_name: str,
) -> float:
    """Compute numeric coverage for a candidate NL question."""
    from data_generation.dataset_utils.nl_post_validator import (
        _extract_scm_values,
        _check_numeric_coverage,
    )
    dag = build_dag(rec["dag_edges"], rec["num_nodes"])
    if variant_name == "full":
        cond_nodes = get_full_condition_nodes(dag)
    else:
        cond_nodes = get_surgery_ancestor_prune_nodes_ate(
            dag, rec["treatment"], rec["outcome"],
        )
    scm_values = _extract_scm_values(rec["cpts"], cond_nodes, rec.get("decimal_places"))
    return _check_numeric_coverage(nl_question, scm_values)


def process_variant_chunk(
    chunk_triplets: List[Dict[str, Any]],
    entity_interp_map: Dict[int, Dict[str, Any]],
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
) -> List[Dict[str, Any]]:
    """Process a chunk for one condition variant.

    Returns final question records (success) or failure records.
    """
    pending = list(range(len(chunk_triplets)))
    prompts: List[str] = []

    for rec in chunk_triplets:
        interp = entity_interp_map[rec["idx"]]
        prompts.append(_build_prompts_for_variant(rec, interp, variant_name))

    results: List[Optional[Dict[str, Any]]] = [None] * len(chunk_triplets)

    # Best-effort failure caching: track the best candidate per record.
    # Rank tuple: (non_empty, no_leakage, coverage) — higher is better.
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

            # Determine candidate rank and failure reason for caching
            is_non_empty = bool(nl_question)
            has_leakage = has_symbolic_leakage(nl_question) if is_non_empty else True
            no_leakage = not has_leakage

            if is_non_empty:
                rec = chunk_triplets[global_idx]
                coverage = _compute_numeric_coverage(nl_question, rec, variant_name)
            else:
                coverage = -1.0

            candidate_rank = (int(is_non_empty), int(no_leakage), coverage)

            # Determine specific failure reason for this attempt
            if not is_non_empty:
                candidate_reason = "empty"
            elif has_leakage:
                candidate_reason = "symbolic_leakage"
            elif coverage < 1.0:
                candidate_reason = "low_numeric_coverage"
            else:
                candidate_reason = None  # passed all checks

            # Update best candidate if this attempt is better
            prev = best_candidates.get(global_idx)
            if prev is None or candidate_rank > prev["rank"]:
                best_candidates[global_idx] = {
                    "question": nl_question,
                    "numeric_coverage": coverage if coverage >= 0 else 0.0,
                    "failure_reason": candidate_reason if candidate_reason else "low_numeric_coverage",
                    "rank": candidate_rank,
                }

            # If all checks pass, accept the record
            if candidate_reason is None:
                rec = chunk_triplets[global_idx]
                interp = entity_interp_map[rec["idx"]]
                qv = derive_query_variant(rec["example_seed"])

                nl_question = nl_question + "\n\n" + NL_ANSWER_FORMAT_SENTENCE

                results[global_idx] = {
                    "idx": rec["idx"],
                    "question": nl_question,
                    "answer": rec["ate_result"]["ate_surgery"],
                    "metadata": {
                        "num_nodes": rec["num_nodes"],
                        "num_edges": rec["num_edges"],
                        "dag_edges": rec["dag_edges"],
                        "treatment": rec["treatment"],
                        "outcome": rec["outcome"],
                        "query_type": "ate",
                        "domain": "nl",
                        "variant_name": variant_name,
                        "condition_variant": variant_name,
                        "pipeline_version": pipeline_version,
                        "query_variant": qv,
                        "passage_idx": interp["passage_idx"],
                        "passage_source": interp["passage_source"],
                        "entity_mapping": interp["entity_interpretation"],
                        "is_abnormal": rec["ate_result"]["is_abnormal"],
                        "verification_ok": rec["ate_result"]["verification_ok"],
                        "treatment_outcome_depth": rec.get("treatment_outcome_depth", 0),
                        "effective_prob_conds": rec.get("effective_prob_conds", 0),
                        "has_naive_conditioning": rec.get("has_naive_conditioning", False),
                        "decimal_places": rec["decimal_places"],
                        "model_entity": interp["model"],
                        "provider_entity": interp["provider"],
                        "model_question": model,
                        "provider_question": provider,
                        "entity_retry_count": interp["retry_count"],
                        "question_retry_count": attempt - 1,
                    },
                }
            else:
                still_pending.append(global_idx)

        pending = still_pending

    # Mark remaining as failed with specific reasons and cached best question
    failed_records = []
    for global_idx in pending:
        rec = chunk_triplets[global_idx]
        best = best_candidates.get(global_idx, {
            "question": "",
            "numeric_coverage": 0.0,
            "failure_reason": "empty",
        })
        failed_records.append({
            "idx": rec["idx"],
            "variant_name": variant_name,
            "failure_reason": best["failure_reason"],
            "best_question": best["question"],
            "best_numeric_coverage": best["numeric_coverage"],
            "retry_count": max_retries,
            "model": model,
            "provider": provider,
        })
        results[global_idx] = None  # will be filtered out

    success_records = [r for r in results if r is not None]
    return success_records, failed_records  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ATE NL question generation step",
    )
    parser.add_argument("--triplet_corpus_path", type=str, required=True)
    parser.add_argument("--entity_interp_path", type=str, required=True,
                        help="Entity interpretation JSONL (v2 step 1 or v3 step 2 output)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Node-level output dir (e.g., nodes6-prompting_v3)")
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
    parser.add_argument("--checkpoint_name", type=str, default=None,
                        help="Stable name for checkpoint files (enables resume across reruns)")
    parser.add_argument("--variant_assignment", type=str, default="split",
                        choices=["split", "both"],
                        help="Variant assignment mode: 'split' generates only one variant "
                             "per idx half-range, 'both' generates both variants for all records")
    parser.add_argument("--config_fingerprint", type=str, default="",
                        help="Config fingerprint to namespace output files across preset sweeps")
    args = parser.parse_args()

    print("=== ATE NL Question Generation ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    validate_api_key(args.provider)

    # Load data
    print("\nLoading triplet corpus...")
    all_triplets = load_jsonl(args.triplet_corpus_path)
    print(f"  Loaded {len(all_triplets)} records")

    print("Loading entity interpretation results...")
    interp_records = load_jsonl(args.entity_interp_path)
    entity_interp_map = {r["idx"]: r for r in interp_records if r["status"] == "success"}
    print(f"  Loaded {len(entity_interp_map)} successful interpretations")

    # Subset
    end_idx = args.end_idx if args.end_idx >= 0 else max(r["idx"] for r in all_triplets) + 1
    records = [
        r for r in all_triplets
        if args.start_idx <= r["idx"] < end_idx and r["idx"] in entity_interp_map
    ]
    print(f"  Processing {len(records)} records")

    # Ensure output directories (before early return so empty files can be written)
    ensure_output_dirs(args.output_dir)
    ensure_intermediate_dirs(args.output_dir)

    # Build filenames: checkpoint in intermediates-nl/, final in {variant}-nl/
    corpus_basename = os.path.basename(args.triplet_corpus_path).replace(".jsonl", "")
    cfg_suffix = f"-cfg{args.config_fingerprint}" if args.config_fingerprint else ""
    final_filename = (
        f"{args.run_timestamp}"
        f"-start{args.start_idx}-end{end_idx}"
        f"-from-{corpus_basename}"
        f"-va_{args.variant_assignment}"
        f"{cfg_suffix}"
    )

    if not records:
        print("Nothing to process — writing empty output files.")
        write_nl_outputs(
            args.output_dir, final_filename,
            {"full": [], "surgery_ancestor_prune": []},
            num_nodes=all_triplets[0]["num_nodes"] if all_triplets else 0,
            triplet_corpus_path=args.triplet_corpus_path,
            pipeline_version=args.pipeline_version,
        )
        return

    # Checkpoint name: stable across reruns when provided.
    if args.checkpoint_name:
        checkpoint_basename = args.checkpoint_name
    else:
        checkpoint_basename = f"{final_filename}"

    # Collect all variant results
    all_variant_records: Dict[str, List[Dict[str, Any]]] = {}

    # Compute midpoint for split variant assignment
    mid_idx = (args.start_idx + end_idx) // 2

    for variant_name in NL_VARIANT_NAMES:
        print(f"\n--- Variant: {variant_name} ---")

        # Filter by variant assignment: in split mode, each variant processes
        # only its assigned idx half-range
        if args.variant_assignment == "split":
            if variant_name == "full":
                variant_input = [r for r in records if r["idx"] < mid_idx]
            else:
                variant_input = [r for r in records if r["idx"] >= mid_idx]
            print(f"  Split mode: {len(variant_input)} records "
                  f"(idx {'< ' + str(mid_idx) if variant_name == 'full' else '>= ' + str(mid_idx)})")
        else:
            variant_input = records

        # Checkpoint path in intermediates-nl/
        checkpoint_path = os.path.join(
            args.output_dir, "intermediates-nl",
            f"{checkpoint_basename}-{variant_name}-checkpoint.jsonl",
        )

        # Resume
        processed = load_processed_indices(checkpoint_path)
        variant_records = [r for r in variant_input if r["idx"] not in processed]
        print(f"  Remaining after resume: {len(variant_records)}")

        all_success: List[Dict[str, Any]] = []
        all_failed: List[Dict[str, Any]] = []

        if variant_records:
            chunk_size = args.checkpoint_interval
            num_chunks = (len(variant_records) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(variant_records))
                chunk = variant_records[start:end]

                print(f"  Chunk {chunk_idx + 1}/{num_chunks}: "
                      f"{len(chunk)} records...")

                success, failed = process_variant_chunk(
                    chunk, entity_interp_map, variant_name,
                    args.model, args.provider,
                    args.max_completion_tokens, args.reasoning_effort,
                    args.num_processes, args.max_retries,
                    args.api_retries, args.api_retry_sleep,
                    args.pipeline_version,
                )

                if success:
                    append_records(checkpoint_path, success)
                all_success.extend(success)
                all_failed.extend(failed)

                print(f"    {len(success)} success, {len(failed)} failed")

        # Combine checkpointed + newly processed records for final output
        all_records_for_variant = load_jsonl(checkpoint_path) if os.path.isfile(checkpoint_path) else []
        all_variant_records[variant_name] = all_records_for_variant

        print(f"  Variant {variant_name}: {len(all_records_for_variant)} total "
              f"({len(all_failed)} failed this run)")

        if all_failed:
            fail_path = os.path.join(
                args.output_dir, "intermediates-nl",
                f"{final_filename}-{variant_name}-failures.jsonl",
            )
            with open(fail_path, "w") as f:
                for rec in all_failed:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Failures logged to: {fail_path}")

    # Write final outputs (JSONL + CSV + stats) to {variant}-nl/{jsonls,csvs,stats}/
    print("\nWriting final outputs...")
    write_nl_outputs(
        args.output_dir,
        final_filename,
        all_variant_records,
        num_nodes=records[0]["num_nodes"],
        triplet_corpus_path=args.triplet_corpus_path,
        pipeline_version=args.pipeline_version,
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
