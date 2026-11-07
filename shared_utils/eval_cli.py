"""Command-line arguments shared by the in-distribution and out-of-distribution evaluators."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from shared_utils.vllm_backends import MODEL_FAMILIES

DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_SAMPLING_SEED = 1
DEFAULT_BUDGET = 3
SAMPLING_PRESET_FIELDS = ("temperature", "top_p", "top_k", "presence_penalty")


def build_eval_parser(description: str, default_results_base_dir: str) -> argparse.ArgumentParser:
    """Build the parser with the arguments common to both evaluators."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model_family",
        required=True,
        choices=sorted(MODEL_FAMILIES),
        help="Selects the sampling preset, output cap, and vLLM options of the model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face ID or local checkpoint path (default: the family's base model).",
    )
    parser.add_argument(
        "--prompts_path_list",
        type=str,
        nargs="+",
        required=True,
        help="Prompt JSONL files to concatenate into one vLLM workload per round.",
    )
    parser.add_argument(
        "--results_base_dir",
        type=str,
        default=default_results_base_dir,
        help=f"Root directory of the results (default: {default_results_base_dir}).",
    )
    parser.add_argument(
        "--max_prompts_per_file",
        type=int,
        default=None,
        help="Keep at most this many prompts from each file; use for smoke tests.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=None, help="Default: family preset.")
    parser.add_argument("--top_p", type=float, default=None, help="Default: family preset.")
    parser.add_argument("--top_k", type=int, default=None, help="Default: family preset.")
    parser.add_argument(
        "--presence_penalty", type=float, default=None, help="Default: family preset."
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=DEFAULT_SAMPLING_SEED,
        help="Base seed. Round N generates with seed N * sampling_seed.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Qwen3 only: render prompts with thinking on. The paper keeps it off.",
    )
    parser.add_argument(
        "--run_timestamp",
        type=str,
        default=None,
        help="Timestamp naming this run's outputs (default: the current time).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help="Final round index. The paper's avg@3 is budget 3 from start_round 1.",
    )
    parser.add_argument("--start_round", type=int, default=1, help="First round index.")
    return parser


def parse_eval_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse arguments, fill family defaults, and check the invariants on rounds and caps."""
    args = parser.parse_args(argv)
    family = MODEL_FAMILIES[args.model_family]
    if args.model is None:
        args.model = family.default_model
    for preset_field in SAMPLING_PRESET_FIELDS:
        if getattr(args, preset_field) is None:
            setattr(args, preset_field, getattr(family, preset_field))

    if args.max_prompts_per_file is not None and args.max_prompts_per_file < 1:
        raise ValueError("max_prompts_per_file must be at least 1 when provided.")
    if args.sampling_seed < 1:
        raise ValueError("sampling_seed must be at least 1: round seeds are round * sampling_seed.")
    if args.start_round < 1:
        raise ValueError("start_round must be at least 1.")
    if args.budget < args.start_round:
        raise ValueError("budget must be greater than or equal to start_round.")
    return args
