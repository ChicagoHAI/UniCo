#!/usr/bin/env python3
"""Unified vLLM evaluation over the seven out-of-distribution causal benchmarks.

Run from the repository root:

    python3 -m ood_eval.eval_ood_vllm \
        --model_family qwen3 \
        --model Qwen/Qwen3-4B \
        --prompts_path_list ./ood_eval/benchmarks/cladder/data/CLadder-full_v1.5_default_processed.jsonl
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from ood_eval.integrated_eval import (
    build_prompt_sources,
    load_integrated_ood_prompts,
    run_unified_ood_evaluation,
)
from shared_utils.eval_cli import build_eval_parser, parse_eval_args
from shared_utils.vllm_backends import (
    MODEL_FAMILIES,
    batch_query_with_vllm,
    clamp_max_new_tokens,
    load_tokenizer,
    load_vllm_model,
    make_prompt_formatter,
    release_vllm_model,
)

DEFAULT_RESULTS_BASE_DIR = "./results/ood_eval"


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the unified out-of-distribution evaluation."""
    parser = build_eval_parser(
        "Unified vLLM evaluation across the OOD causal benchmarks.",
        DEFAULT_RESULTS_BASE_DIR,
    )
    args = parse_eval_args(parser, argv)
    family = MODEL_FAMILIES[args.model_family]
    args.max_new_tokens = clamp_max_new_tokens(args.max_new_tokens, family, args.model)

    print("Script arguments:")
    print(json.dumps(vars(args), indent=2))

    sources = build_prompt_sources(args.prompts_path_list)
    tokenizer = load_tokenizer(args.model)
    all_data = load_integrated_ood_prompts(
        sources,
        make_prompt_formatter(tokenizer, family, args.enable_thinking),
        max_prompts_per_file=args.max_prompts_per_file,
    )
    if not all_data:
        raise ValueError("No prompts were loaded; nothing to evaluate.")

    model = load_vllm_model(args.model, family, args.max_new_tokens)
    try:
        run_unified_ood_evaluation(
            args=args,
            sources=sources,
            all_data=all_data,
            model=model,
            batch_query=batch_query_with_vllm,
        )
    finally:
        release_vllm_model(model)


if __name__ == "__main__":
    main()
