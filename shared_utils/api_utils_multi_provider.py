"""Multi-provider API utilities using OpenAI SDK >=1.0.

Supports OpenRouter, official OpenAI, Gemini (via OpenAI-compatible
endpoint), and GCP Vertex AI with per-provider reasoning parameter
handling.  Uses ProcessPoolExecutor with per-worker client initialization
and per-prompt tqdm progress tracking.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "reasoning_style": "extra_body",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "reasoning_style": "top_level",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GOOGLE_API_KEY",
        "api_key_env_fallback": "GEMINI_API_KEY",
        "reasoning_style": "top_level",
    },
    "vertex": {
        "base_url": "",  # constructed dynamically from env vars
        "api_key_env": "",  # uses GCP ADC, not a static API key
        "reasoning_style": "top_level",
        "auth_style": "gcp_adc",
        "project_env": "GCP_PROJECT_ID",
        "location_env": "GCP_LOCATION",
        "location_default": "global",
    },
}

SUPPORTED_PROVIDERS = tuple(PROVIDER_CONFIGS.keys())
SUPPORTED_REASONING_EFFORTS = ("none", "low", "medium", "high")

# Model-family prefixes rejected by each direct provider.
# OpenRouter is intentionally permissive (routes to many backends).
_PROVIDER_REJECTED_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "openai": ("gemini-", "gemini_"),
    "gemini": ("gpt-", "o1-", "o3-", "o4-"),
    "openrouter": (),
    "vertex": ("gpt-", "o1-", "o3-", "o4-"),
}

_KNOWN_OPENROUTER_PREFIXES = {"openai", "google", "anthropic"}

_OPENROUTER_PREFIX_FAMILIES = {
    "openai": ("gpt-", "o1", "o3", "o4"),
    "google": ("gemini",),
    "anthropic": ("claude",),
}

# ---------------------------------------------------------------------------
# Module-global worker state (one client per worker process)
# ---------------------------------------------------------------------------

_worker_client = None
_worker_provider: Optional[str] = None


def _openrouter_effective_model(model_name: str) -> str:
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Malformed prefixed model name for OpenRouter: {model_name!r}. "
                "Expected exactly one slash with 'prefix/model'."
            )
        prefix, bare = parts[0], parts[1]
        if prefix not in _KNOWN_OPENROUTER_PREFIXES:
            raise ValueError(
                f"Unsupported OpenRouter prefix {prefix!r} in {model_name!r}. "
                f"Supported prefixes: {', '.join(sorted(_KNOWN_OPENROUTER_PREFIXES))}"
            )
        expected_starts = _OPENROUTER_PREFIX_FAMILIES.get(prefix, ())
        if expected_starts and not bare.lower().startswith(expected_starts):
            raise ValueError(
                f"OpenRouter prefix {prefix!r} does not match model {bare!r} "
                f"in {model_name!r}. The {prefix!r} prefix expects models "
                f"starting with: {', '.join(expected_starts)}"
            )
        return model_name

    lower = model_name.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{model_name}"
    if lower.startswith("gemini"):
        return f"google/{model_name}"
    if lower.startswith("claude"):
        return f"anthropic/{model_name}"
    raise ValueError(
        f"Unsupported model family for OpenRouter: {model_name!r}. "
        "Supported families: gpt/o-series (openai/), gemini (google/), "
        "claude (anthropic/)."
    )


def _get_api_key_from_env(cfg: Dict[str, str]) -> Tuple[Optional[str], Tuple[str, ...]]:
    env_vars = [cfg["api_key_env"]]
    fallback = cfg.get("api_key_env_fallback")
    if fallback:
        env_vars.append(fallback)
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value:
            return value, tuple(env_vars)
    return None, tuple(env_vars)


class _VertexCredentialsRefresher:
    """OpenAI client wrapper with auto-refreshing GCP ADC tokens.

    Vertex AI issues OAuth tokens that expire after 1 hour.  This wrapper
    checks ``self._creds.valid`` on every attribute access and refreshes
    transparently, following the pattern recommended by Google Cloud docs.
    """

    def __init__(self, **kwargs: Any) -> None:
        import google.auth
        import google.auth.transport.requests as _ga_requests

        from openai import OpenAI

        self._client = OpenAI(**kwargs, api_key="PLACEHOLDER")
        self._creds, self._project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._ga_requests = _ga_requests

    def __getattr__(self, name: str) -> Any:
        if not self._creds.valid:
            self._creds.refresh(self._ga_requests.Request())
        if not self._creds.valid:
            raise RuntimeError("Unable to refresh GCP credentials")
        self._client.api_key = self._creds.token
        return getattr(self._client, name)


def _init_worker(provider: str) -> None:
    """ProcessPoolExecutor initializer: create OpenAI client for *provider*."""
    global _worker_client, _worker_provider

    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unsupported provider {provider!r}. "
            f"Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    _worker_provider = provider
    cfg = PROVIDER_CONFIGS[provider]

    if cfg.get("auth_style") == "gcp_adc":
        # Vertex AI: ADC-based auth with auto-refreshing tokens
        project_id = os.environ.get(cfg["project_env"])
        location = os.environ.get(cfg["location_env"], cfg["location_default"])
        if not project_id:
            raise ValueError(
                f"{cfg['project_env']} is not set. "
                "Set it in the environment or load it from a .env file."
            )
        host = (
            f"{location}-aiplatform.googleapis.com"
            if location != "global"
            else "aiplatform.googleapis.com"
        )
        base_url = (
            f"https://{host}/v1/"
            f"projects/{project_id}/locations/{location}/"
            f"endpoints/openapi"
        )
        _worker_client = _VertexCredentialsRefresher(base_url=base_url)
    else:
        # Standard API-key auth (openrouter, openai, gemini)
        api_key, env_vars = _get_api_key_from_env(cfg)
        if not api_key:
            raise ValueError(
                f"None of {', '.join(env_vars)} are set. "
                "Set it in the environment or load it from a .env file."
            )

        from openai import OpenAI

        _worker_client = OpenAI(base_url=cfg["base_url"], api_key=api_key)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_provider(provider: str) -> None:
    """Raise immediately if *provider* is not supported."""
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unsupported provider {provider!r}. "
            f"Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
        )


def _load_dotenv_once() -> None:
    """Try to load ``.env`` from the repo root once (no-op if already loaded)."""
    if getattr(_load_dotenv_once, "_done", False):
        return
    _load_dotenv_once._done = True
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def validate_api_key(provider: str) -> None:
    """Raise immediately if credentials for *provider* are unavailable.

    Loads ``.env`` via ``python-dotenv`` (if installed) before checking,
    so credentials stored only in ``.env`` are picked up.
    """
    _load_dotenv_once()
    validate_provider(provider)
    cfg = PROVIDER_CONFIGS[provider]

    if cfg.get("auth_style") == "gcp_adc":
        # Vertex AI: check project ID env var; ADC availability is
        # verified lazily at worker init time.
        project_env = cfg["project_env"]
        if not os.environ.get(project_env):
            raise ValueError(
                f"{project_env} is not set. "
                "Set it in the environment or load it from a .env file."
            )
    else:
        api_key, env_vars = _get_api_key_from_env(cfg)
        if not api_key:
            raise ValueError(
                f"None of {', '.join(env_vars)} are set. "
                "Set it in the environment or load it from a .env file."
            )


def validate_params(
    provider: str,
    model_name: str,
    reasoning_effort: str,
    temperature: Optional[float],
    top_p: Optional[float],
) -> None:
    """Fail-fast parameter validation before any API request."""
    validate_provider(provider)

    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    stripped = model_name.strip()
    bare_name = stripped
    if "/" in stripped:
        prefix_part, bare_name = stripped.split("/", 1)
        if provider == "openrouter" and prefix_part not in _KNOWN_OPENROUTER_PREFIXES:
            raise ValueError(
                f"Unsupported prefix {prefix_part!r} in model name {model_name!r} "
                "for provider openrouter. Supported prefixes: "
                f"{', '.join(sorted(_KNOWN_OPENROUTER_PREFIXES))}"
            )

    # Model/provider compatibility: reject known-bad model families
    model_lower = bare_name.lower()
    rejected = _PROVIDER_REJECTED_PREFIXES.get(provider, ())
    for prefix in rejected:
        if model_lower.startswith(prefix):
            raise ValueError(
                f"Model {model_name!r} is not compatible with provider "
                f"{provider!r}. Models starting with {prefix!r} should not "
                f"be sent to the {provider} endpoint."
            )

    if provider in ("gemini", "vertex") and not model_lower.startswith("gemini"):
        raise ValueError(
            f"Model {model_name!r} is not compatible with provider "
            f"{provider!r}. Only Gemini-family models can be sent to "
            f"the {provider} endpoint."
        )

    if provider == "openai" and not model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        raise ValueError(
            f"Model {model_name!r} is not compatible with provider "
            f"{provider!r}. Only GPT and o-series models can be sent to "
            "the openai endpoint."
        )

    if provider == "openrouter":
        _openrouter_effective_model(stripped)

    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            f"Unsupported reasoning_effort {reasoning_effort!r}. "
            f"Choose from: {', '.join(SUPPORTED_REASONING_EFFORTS)}"
        )

    if reasoning_effort != "none":
        if temperature is not None:
            raise ValueError(
                "temperature must not be set when reasoning is enabled "
                f"(reasoning_effort={reasoning_effort!r})."
            )
        if top_p is not None:
            raise ValueError(
                "top_p must not be set when reasoning is enabled "
                f"(reasoning_effort={reasoning_effort!r})."
            )


# ---------------------------------------------------------------------------
# Single-prompt call (runs inside worker process)
# ---------------------------------------------------------------------------

def _build_kwargs(
    model_name: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int,
    reasoning_effort: str,
    temperature: Optional[float],
    top_p: Optional[float],
) -> Dict[str, Any]:
    """Assemble provider-aware kwargs for ``chat.completions.create``."""
    provider = _worker_provider
    cfg = PROVIDER_CONFIGS[provider]

    effective_model = model_name
    if "/" in model_name and provider in ("openai", "gemini"):
        effective_model = model_name.split("/", 1)[1]
    if provider == "vertex" and not effective_model.startswith("google/"):
        effective_model = f"google/{effective_model}"
    elif provider == "openrouter":
        effective_model = _openrouter_effective_model(effective_model)

    kwargs: Dict[str, Any] = {
        "model": effective_model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }

    if reasoning_effort != "none":
        if cfg["reasoning_style"] == "extra_body":
            kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
        else:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

    return kwargs


def _call_single_prompt(
    prompt: str,
    model_name: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    temperature: Optional[float],
    top_p: Optional[float],
    retries: int,
    retry_sleep_seconds: float,
) -> Dict[str, Any]:
    """Process one prompt with the worker's OpenAI client.

    Returns
    -------
    dict
        ``{"response": str, "completion_length": int}``
    """
    global _worker_client
    if _worker_client is None:
        raise RuntimeError("Worker client not initialized; was _init_worker called?")

    prompt_text = str(prompt).strip()
    if not prompt_text:
        return {"response": "", "completion_length": 0}

    messages = [{"role": "user", "content": prompt_text}]
    kwargs = _build_kwargs(
        model_name, messages, max_completion_tokens,
        reasoning_effort, temperature, top_p,
    )

    attempts = max(retries, 1)
    while attempts > 0:
        try:
            response = _worker_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            usage = response.usage
            completion_tokens = 0
            if usage is not None:
                completion_tokens = usage.completion_tokens or 0
            return {
                "response": content.strip(),
                "completion_length": int(completion_tokens),
            }
        except Exception as exc:
            attempts -= 1
            if attempts == 0:
                raise RuntimeError(
                    f"API call failed after {retries} attempts "
                    f"for model {model_name} via {_worker_provider}: {exc!r}"
                ) from exc
            print(
                f"[warn] API call failed with {exc!r}; "
                f"retrying in {retry_sleep_seconds:.1f}s "
                f"({attempts} attempts left)...",
                flush=True,
            )
            time.sleep(retry_sleep_seconds)

    raise RuntimeError("Unreachable retry state in _call_single_prompt().")


# ---------------------------------------------------------------------------
# Batch query entry point
# ---------------------------------------------------------------------------

def batch_query_with_api(
    prompts: List[str],
    model_name: str,
    provider: str,
    max_completion_tokens: int,
    reasoning_effort: str = "none",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    num_processes: int = 4,
    api_retries: int = 5,
    api_retry_sleep_seconds: float = 10.0,
) -> Tuple[List[str], List[int]]:
    """Process prompts in parallel through a provider-specific API.

    Parameters
    ----------
    prompts : list[str]
        Prompts to submit (one per API call).
    model_name : str
        Model identifier recognised by the chosen *provider*.
    provider : str
        One of ``"openrouter"``, ``"openai"``, ``"gemini"``, or
        ``"vertex"``.
    max_completion_tokens : int
        Maximum completion tokens.
    reasoning_effort : str
        ``"none"``, ``"low"``, ``"medium"``, or ``"high"``.  Handling
        differs by provider (OpenRouter uses ``extra_body``; OpenAI and
        Gemini use a top-level parameter).
    temperature, top_p : float | None
        Sampling params.  Must be ``None`` when *reasoning_effort* is not
        ``"none"``.
    num_processes : int
        Worker count for ``ProcessPoolExecutor``.
    api_retries : int
        Max retries per prompt on transient failures.
    api_retry_sleep_seconds : float
        Sleep between retries.

    Returns
    -------
    (responses, out_token_lens) : tuple[list[str], list[int]]
        Lists aligned with the input *prompts*.
    """
    # Fail-fast validation before spawning workers
    validate_api_key(provider)
    validate_params(provider, model_name, reasoning_effort, temperature, top_p)

    if not prompts:
        return [], []

    mp_ctx = multiprocessing.get_context("spawn")
    max_workers = max(1, min(num_processes, len(prompts)))

    future_to_idx: Dict[Any, int] = {}
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(provider,),
    ) as executor:
        for i, prompt in enumerate(prompts):
            future = executor.submit(
                _call_single_prompt,
                prompt,
                model_name,
                max_completion_tokens,
                reasoning_effort,
                temperature,
                top_p,
                api_retries,
                api_retry_sleep_seconds,
            )
            future_to_idx[future] = i

        failed_count = 0
        last_exc: RuntimeError | None = None
        with tqdm(total=len(prompts), desc="API calls", unit="prompt") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except RuntimeError as exc:
                    # Exhausted per-prompt retries — isolate single failures
                    # so other prompts' results are preserved for checkpointing.
                    failed_count += 1
                    last_exc = exc
                    print(f"  [WARN] Prompt {idx} failed: {exc}")
                    results[idx] = {"response": "", "completion_length": 0}
                pbar.update(1)

        if failed_count > 0:
            severity = "ERROR" if failed_count == len(prompts) else "WARN"
            print(
                f"  [{severity}] {failed_count}/{len(prompts)} prompts failed in batch. "
                f"Last error: {last_exc}"
            )
        # Always return results (including empty responses for failed
        # prompts) so the caller's own retry loop can handle them.
        # Permanent config/auth errors are caught by validate_api_key()
        # and validate_params() before any API call.

    responses: List[str] = []
    out_token_lens: List[int] = []
    for i, record in enumerate(results):
        if record is None:
            raise ValueError(f"Missing result for prompt index {i}.")
        responses.append(record["response"])
        out_token_lens.append(record["completion_length"])

    return responses, out_token_lens
