"""vLLM loading, prompt rendering, and generation for the evaluated model families.

All three families share one code path. They differ only in the values recorded in
``MODEL_FAMILIES``: the sampling preset, the output cap, and a few vLLM keyword arguments.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

from shared_utils.prompt_utils import format_prompt_chat_template

DEFAULT_MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.9
VLLM_ENGINE_SEED = 0
CHECKPOINT_DIR_PREFIX = "checkpoint-"

# Templates that own their BOS token (notably Gemma 3 IT) would receive a second BOS if vLLM
# asked the tokenizer to add special tokens to the already-rendered prompt.
_NO_ADDED_SPECIAL_TOKENS = {"tokenization_kwargs": {"add_special_tokens": False}}


@dataclass(frozen=True)
class ModelFamily:
    """Evaluation settings shared by every model in one family."""

    name: str
    default_model: str
    max_output_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    llm_kwargs: Mapping[str, Any] = field(default_factory=dict)
    generate_kwargs: Mapping[str, Any] = field(default_factory=dict)


MODEL_FAMILIES: Dict[str, ModelFamily] = {
    "qwen3": ModelFamily(
        name="qwen3",
        default_model="Qwen/Qwen3-4B",
        max_output_tokens=38912,  # max_position_embeddings is 40960 for Qwen3.
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        presence_penalty=1.5,
    ),
    "olmo3": ModelFamily(
        name="olmo3",
        default_model="allenai/Olmo-3-7B-Instruct",
        max_output_tokens=DEFAULT_MAX_MODEL_LEN,
        temperature=0.6,
        top_p=0.95,
        top_k=0,
        presence_penalty=0.0,
        generate_kwargs=_NO_ADDED_SPECIAL_TOKENS,
    ),
    "gemma3": ModelFamily(
        name="gemma3",
        default_model="google/gemma-3-27b-it",
        max_output_tokens=DEFAULT_MAX_MODEL_LEN,
        temperature=0.6,
        top_p=0.95,
        top_k=0,
        presence_penalty=0.0,
        # The checkpoint is multimodal; load only its text components.
        llm_kwargs={"language_model_only": True},
        generate_kwargs=_NO_ADDED_SPECIAL_TOKENS,
    ),
}


def infer_model_dir_name(model_ref: str) -> str:
    """Name a results directory after a Hugging Face ID or a local checkpoint path.

    ``Qwen/Qwen3-4B`` gives ``Qwen3-4B``. ``./out/my-run/checkpoint-2082`` gives ``my-run``.
    """
    path = Path(model_ref.rstrip("/"))
    if path.name.startswith(CHECKPOINT_DIR_PREFIX):
        return path.resolve().parent.name
    return path.name


def clamp_max_new_tokens(max_new_tokens: int, family: ModelFamily, model_ref: str) -> int:
    """Limit the output length to the family's cap, warning when the request exceeds it."""
    if max_new_tokens <= family.max_output_tokens:
        return max_new_tokens
    warnings.warn(
        f"max_new_tokens ({max_new_tokens}) exceeds the {family.name} cap "
        f"({family.max_output_tokens}) for {model_ref!r}; clamping to the cap.",
        stacklevel=2,
    )
    return family.max_output_tokens


def determine_max_model_len(model_ref: str, max_new_tokens: int) -> int:
    """Return vLLM's ``max_model_len``: the default, or the model's limit if that is lower."""
    max_model_len = DEFAULT_MAX_MODEL_LEN
    # A config that cannot be read must not stop an evaluation: vLLM validates the context
    # length itself when it loads the model, so fall back to the default and say so.
    try:
        config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - loader errors vary across Transformers releases.
        print(
            f"Could not load the config of {model_ref!r}: {exc!r}. "
            f"Using max_model_len={max_model_len}.",
            flush=True,
        )
    else:
        model_max_positions = getattr(config, "max_position_embeddings", None)
        if model_max_positions is not None and model_max_positions < max_model_len:
            print(
                f"Clamping max_model_len to max_position_embeddings={model_max_positions}.",
                flush=True,
            )
            max_model_len = model_max_positions

    if max_model_len < max_new_tokens:
        warnings.warn(
            f"max_model_len ({max_model_len}) is smaller than max_new_tokens "
            f"({max_new_tokens}); generations may be truncated.",
            stacklevel=2,
        )
    print(f"Using max_model_len={max_model_len} for {model_ref!r}.", flush=True)
    return max_model_len


def load_tokenizer(model_ref: str) -> Any:
    """Load the tokenizer whose chat template renders the prompts."""
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    print(f"Using chat template:\n{tokenizer.chat_template}\n", flush=True)
    return tokenizer


def format_prompt_qwen3(tokenizer: Any, instruction: str, enable_thinking: bool) -> str:
    """Render one user turn with the Qwen3 chat template and an explicit thinking switch."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"Tokenizer {tokenizer.name_or_path!r} has no chat_template.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def make_prompt_formatter(
    tokenizer: Any,
    family: ModelFamily,
    enable_thinking: bool = False,
) -> Callable[[str], str]:
    """Build the function that turns one raw question into a rendered prompt."""
    if family.name == "qwen3":
        print(f"Using enable_thinking={enable_thinking} for the chat template.", flush=True)
        return lambda instruction: format_prompt_qwen3(tokenizer, instruction, enable_thinking)
    if enable_thinking:
        raise ValueError(f"--enable_thinking only applies to qwen3, not {family.name}.")
    return lambda instruction: format_prompt_chat_template(tokenizer, instruction)


def load_vllm_model(model_ref: str, family: ModelFamily, max_new_tokens: int) -> LLM:
    """Load one model with vLLM across all visible GPUs."""
    print(f"Loading {model_ref} with vLLM ...", flush=True)
    model = LLM(
        model=model_ref,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=determine_max_model_len(model_ref, max_new_tokens),
        trust_remote_code=True,
        seed=VLLM_ENGINE_SEED,
        **family.llm_kwargs,
    )
    print("vLLM model loaded.", flush=True)
    return model


def release_vllm_model(model: Any) -> None:
    """Free the model's GPU memory without hiding a failure from the evaluation itself."""
    try:
        if hasattr(model, "shutdown"):
            model.shutdown()
        elif hasattr(model, "llm_engine") and hasattr(
            model.llm_engine, "shutdown_background_loop"
        ):
            model.llm_engine.shutdown_background_loop()
    except Exception as exc:  # noqa: BLE001 - results are already saved at this point.
        print(f"[warn] vLLM shutdown raised {exc!r} (continuing).", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def batch_query_with_vllm(
    model: Any,
    prompts: List[str],
    args: Any,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    """Generate one completion per rendered prompt; return the texts and their token counts."""
    family = MODEL_FAMILIES[args.model_family]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.sampling_seed if seed is None else seed,
        n=1,
    )
    outputs = model.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
        **family.generate_kwargs,
    )
    responses: List[str] = []
    completion_lengths: List[int] = []
    for output in outputs:
        candidate = output.outputs[0]
        responses.append(candidate.text.strip())
        completion_lengths.append(len(candidate.token_ids))
    return responses, completion_lengths
