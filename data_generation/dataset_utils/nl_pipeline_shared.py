"""NL pipeline shared utilities for ATE and CATE/JATE NL-domain pipelines.

Provides checkpointing, resume logic, JSON extraction from LLM responses,
symbolic leakage detection, output directory management, and stats/IO for
the NL domain translation pipeline.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set

from data_generation.dataset_utils.nl_query_specs import (
    is_v2_2_query_type,
    selector_assumptions_required,
)

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_passage(
    passages: List[Dict[str, Any]],
    example_seed: int,
    query_type_offset: int,
) -> Dict[str, Any]:
    """Deterministically sample a passage given seed and offset."""
    rng = random.Random(example_seed + query_type_offset)
    idx = rng.randint(0, len(passages) - 1)
    return passages[idx]


# ---------------------------------------------------------------------------
# NL domain folder mapping
# ---------------------------------------------------------------------------

NL_VARIANT_NAMES = ["full", "surgery_ancestor_prune"]
NL_FOLDER_NAMES = ["full-nl", "surgery_ancestor_prune-nl"]

NL_ANSWER_FORMAT_SENTENCE = (
    "Think step-by-step to solve this question. "
    "Your final answer should be rounded to four decimal places "
    "and included inside \\boxed{}, as below: \\boxed{<your final answer>}."
)


def variant_to_folder(variant_name: str) -> str:
    """Map condition variant name to NL-domain folder name."""
    return f"{variant_name}-nl"


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def ensure_output_dirs(
    output_dir: str,
    variant_names: Optional[List[str]] = None,
) -> None:
    """Create final output directories for NL-domain variants.

    Creates ``{output_dir}/{variant}-nl/{jsonls,csvs,stats}/`` for each
    condition variant, plus ``.logs/``.
    """
    folder_names = [
        variant_to_folder(v)
        for v in (variant_names if variant_names is not None else NL_VARIANT_NAMES)
    ]
    for folder_name in folder_names:
        for subdir in ("jsonls", "csvs", "stats"):
            os.makedirs(
                os.path.join(output_dir, folder_name, subdir), exist_ok=True,
            )
    os.makedirs(os.path.join(output_dir, ".logs"), exist_ok=True)


def ensure_intermediate_dirs(output_dir: str) -> None:
    """Create intermediate output directory ``{output_dir}/intermediates-nl/``."""
    os.makedirs(
        os.path.join(output_dir, "intermediates-nl"), exist_ok=True,
    )


# ---------------------------------------------------------------------------
# Checkpointing and resume
# ---------------------------------------------------------------------------

def load_processed_indices(output_path: str) -> Set[int]:
    """Read an existing JSONL and return the set of successfully processed ``idx`` values.

    Only records with ``status == "success"`` or no ``status`` field (e.g.,
    final question-gen outputs) are considered processed.  Failed records
    are excluded so they can be retried on resume.

    Returns an empty set if the file does not exist.
    """
    indices: Set[int] = set()
    if not os.path.isfile(output_path):
        return indices

    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            status = record.get("status")
            if status is None or status == "success":
                indices.add(record["idx"])

    return indices


def append_records(output_path: str, records: List[Dict[str, Any]]) -> None:
    """Append *records* to a JSONL file and flush."""
    with open(output_path, "a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# JSON extraction from LLM responses
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from *text*.

    Attempts (in order):
    1. Fenced code block (````json ... ``` `` or ```` ``` ... ``` ````)
    2. First ``{`` to last ``}`` substring
    3. Return ``None`` on failure.
    """
    # Try fenced block first
    match = _JSON_FENCE_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON (first { to last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Entity mapping validation
# ---------------------------------------------------------------------------

def _validate_exogenous_selectors(
    node_data: Dict[str, Any],
    cpt_entry: Dict[str, Any],
) -> bool:
    selectors = node_data.get("exogenous_selectors")
    if not isinstance(selectors, dict):
        return False
    expected_count = 2 ** len(cpt_entry.get("parents", []))
    if len(selectors) != expected_count:
        return False
    for selector_data in selectors.values():
        if not isinstance(selector_data, dict):
            return False
        if not all(key in selector_data for key in ("entity", "0", "1")):
            return False
    return True


def validate_entity_assign(
    entity_json: Dict[str, Any],
    num_nodes: int,
    query_type: Optional[str] = None,
    cpts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Check v3-step1 entity assignment has all node keys with ``entity`` field."""
    for i in range(num_nodes):
        key = f"X{i}"
        if key not in entity_json:
            return False
        node_data = entity_json[key]
        if not isinstance(node_data, dict) or "entity" not in node_data:
            return False
        if selector_assumptions_required(query_type):
            if cpts is None:
                return False
            if not _validate_exogenous_selectors(node_data, cpts[str(i)]):
                return False
    return True


def validate_entity_interp(
    entity_json: Dict[str, Any],
    num_nodes: int,
    query_type: Optional[str] = None,
    cpts: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """Check entity interpretation has all nodes with ``entity``, ``0``, ``1``."""
    for i in range(num_nodes):
        key = f"X{i}"
        if key not in entity_json:
            return False
        node_data = entity_json[key]
        if not isinstance(node_data, dict):
            return False
        # v2 uses "0"/"1"; v3 uses "0_interpretation"/"1_interpretation"
        has_01 = "0" in node_data and "1" in node_data
        has_interp = "0_interpretation" in node_data and "1_interpretation" in node_data
        if "entity" not in node_data or not (has_01 or has_interp):
            return False
        if selector_assumptions_required(query_type):
            if cpts is None:
                return False
            if not _validate_exogenous_selectors(node_data, cpts[str(i)]):
                return False
    return True


# ---------------------------------------------------------------------------
# Numeric fidelity check (inline, for question gen gating)
# ---------------------------------------------------------------------------

def check_numeric_fidelity(
    question: str,
    cpts: Dict[str, Dict[str, Any]],
    condition_nodes: Optional[List[int]],
    decimal_places: Optional[int],
    threshold: float = 1.0,
) -> bool:
    """Return True if *question* mentions enough SCM values.

    Reuses extraction and coverage logic from the post-validator.
    """
    from data_generation.dataset_utils.nl_post_validator import (
        _extract_scm_values,
        _check_numeric_coverage,
    )
    scm_values = _extract_scm_values(cpts, condition_nodes, decimal_places)
    coverage = _check_numeric_coverage(question, scm_values)
    return coverage >= threshold


# ---------------------------------------------------------------------------
# Symbolic leakage detection
# ---------------------------------------------------------------------------

_SYMBOLIC_TOKEN_RE = re.compile(
    r"X\d+(?=[\s,.)=:;}\]\|]|$)"
    r"|"
    r"X\{\d+\}"
)


def count_symbolic_tokens(text: str) -> int:
    """Count occurrences of symbolic node token patterns (uppercase X only).

    Matches ``X<digits>`` followed by a delimiter (space, comma, period,
    closing paren/brace/bracket, equals, colon, semicolon, pipe, or
    end-of-string), and ``X{<digits>}`` with curly-braced indices.
    Case-sensitive to avoid false positives like ``4x100m``.
    """
    return len(_SYMBOLIC_TOKEN_RE.findall(text))


def has_symbolic_leakage(text: str, threshold: int = 1) -> bool:
    """Return ``True`` if the text has > *threshold* symbolic token matches.

    With the default threshold of 1, a single isolated match is tolerated
    (it may be coincidental) while 2 or more matches trigger rejection.
    """
    return count_symbolic_tokens(text) > threshold


# Common LLM boilerplate patterns stripped from question gen responses
_BOILERPLATE_PREFIXES = [
    "Here is the converted question:",
    "Here is the natural language question:",
    "Here is the rewritten question:",
    "Here's the converted question:",
    "Here's the natural language question:",
    "Here's the rewritten question:",
    "Converted question:",
    "Natural language question:",
    "Rewritten question:",
    "Sure, here is the converted question:",
    "Sure, here's the converted question:",
    "Sure! Here is the converted question:",
    "Sure! Here's the converted question:",
]


def clean_question_response(text: str) -> str:
    """Strip common LLM boilerplate from a question generation response.

    Removes known preamble prefixes (case-insensitive) and surrounding
    quotation marks or markdown formatting.
    """
    cleaned = text.strip()

    # Strip known prefixes (case-insensitive)
    lower = cleaned.lower()
    for prefix in _BOILERPLATE_PREFIXES:
        if lower.startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
            lower = cleaned.lower()

    # Strip surrounding quotes (single, double, or backtick)
    if len(cleaned) >= 2:
        if (cleaned[0] == '"' and cleaned[-1] == '"') or \
           (cleaned[0] == "'" and cleaned[-1] == "'") or \
           (cleaned[0] == '`' and cleaned[-1] == '`'):
            cleaned = cleaned[1:-1].strip()

    # Strip triple backtick fencing (with optional language tag)
    if cleaned.startswith("```") and cleaned.endswith("```"):
        # Remove opening ``` and closing ```
        inner = cleaned[3:]
        if inner.endswith("```"):
            inner = inner[:-3]
        # Strip optional language tag on the first line (e.g., ```markdown)
        first_newline = inner.find("\n")
        if first_newline >= 0:
            first_line = inner[:first_newline].strip()
            # If the first line is only a language tag (alphanumeric), remove it
            if first_line.isalpha():
                inner = inner[first_newline + 1:]
        cleaned = inner.strip()

    return cleaned


# ---------------------------------------------------------------------------
# Final NL record schema validation
# ---------------------------------------------------------------------------

# Base metadata fields required for all NL question records
_BASE_REQUIRED_METADATA = frozenset({
    "num_nodes", "num_edges", "dag_edges",
    "query_type", "domain", "variant_name", "condition_variant",
    "pipeline_version", "query_variant",
    "passage_idx", "passage_source", "entity_mapping",
    "is_abnormal", "verification_ok",
    "treatment_outcome_depth", "effective_prob_conds", "has_naive_conditioning",
    "decimal_places",
    "model_entity", "provider_entity",
    "model_question", "provider_question",
    "entity_retry_count", "question_retry_count",
})

# Query-type-specific extra required fields
_QUERY_TYPE_EXTRA_FIELDS: Dict[str, frozenset] = {
    "ate": frozenset({"treatment", "outcome"}),
    "cate": frozenset({"treatment", "evidence", "outcome"}),
    "jate": frozenset({"treatments", "outcome"}),
    "conditional_prob": frozenset({"target", "evidence", "probability"}),
    "joint_prob": frozenset({"target_nodes", "probability"}),
    "marginal_prob": frozenset({"target", "probability"}),
    "observed_difference": frozenset({"source", "target", "observed_difference"}),
    "att": frozenset({"treatment", "outcome", "att_value"}),
    "cf": frozenset({"treatments", "outcome", "evidence", "cf_answer_probability"}),
    "nie": frozenset({"treatment", "mediator", "outcome", "effect_type", "nie_value"}),
    "nde": frozenset({"treatment", "mediator", "outcome", "effect_type", "nde_value"}),
    "pn": frozenset({"treatment", "outcome", "effect_type", "pn_value"}),
    "ps": frozenset({"treatment", "outcome", "effect_type", "ps_value"}),
}

# Top-level required fields (outside metadata)
_TOP_LEVEL_REQUIRED = frozenset({"idx", "question", "answer", "metadata"})


def validate_nl_record_schema(
    record: Dict[str, Any],
    query_type: Optional[str] = None,
) -> List[str]:
    """Validate a final NL question record against the required schema.

    Parameters
    ----------
    record : dict
        A final NL question record with ``idx``, ``question``, ``answer``,
        ``metadata``.
    query_type : str or None
        If provided, also check query-type-specific metadata fields.
        If ``None``, infer from ``record["metadata"]["query_type"]``.

    Returns
    -------
    list[str]
        List of error messages.  Empty list means the record is valid.
    """
    errors: List[str] = []

    # Top-level fields
    missing_top = _TOP_LEVEL_REQUIRED - set(record.keys())
    if missing_top:
        errors.append(f"Missing top-level fields: {sorted(missing_top)}")
        return errors  # can't check metadata if top-level is broken

    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        errors.append("metadata is not a dict")
        return errors

    # Base metadata
    missing_base = _BASE_REQUIRED_METADATA - set(metadata.keys())
    if missing_base:
        errors.append(f"Missing base metadata fields: {sorted(missing_base)}")

    # Query-type-specific
    qt = query_type or metadata.get("query_type")
    if qt and qt in _QUERY_TYPE_EXTRA_FIELDS:
        missing_qt = _QUERY_TYPE_EXTRA_FIELDS[qt] - set(metadata.keys())
        if missing_qt:
            errors.append(
                f"Missing {qt}-specific metadata fields: {sorted(missing_qt)}"
            )
    elif qt and qt not in _QUERY_TYPE_EXTRA_FIELDS and not is_v2_2_query_type(qt):
        errors.append(f"Unknown query_type: {qt!r}")

    return errors


# ---------------------------------------------------------------------------
# Stats and output writing
# ---------------------------------------------------------------------------

def compute_nl_variant_stats(
    variant_records: List[Dict[str, Any]],
    num_nodes: int,
    source_triplet_corpus: str,
    pipeline_version: str,
) -> Dict[str, Any]:
    """Compute dataset-level statistics for one NL-domain variant."""
    num_examples = len(variant_records)
    if num_examples == 0:
        return {"num_examples": 0}

    abnormal_count = sum(
        1 for r in variant_records if r["metadata"]["is_abnormal"]
    )
    verification_false_count = sum(
        1 for r in variant_records
        if r["metadata"]["verification_ok"] is False
    )

    # Query variant distribution
    qv_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    for r in variant_records:
        qv = r["metadata"]["query_variant"]
        qv_counts[qv] = qv_counts.get(qv, 0) + 1

    # Difficulty metrics
    total_depth = sum(
        r["metadata"].get("treatment_outcome_depth", 0) for r in variant_records
    )
    total_eff_conds = sum(
        r["metadata"].get("effective_prob_conds", 0) for r in variant_records
    )
    naive_count = sum(
        1 for r in variant_records
        if r["metadata"]["has_naive_conditioning"]
    )

    # NL-specific: retry counts
    total_entity_retries = sum(
        r["metadata"].get("entity_retry_count", 0) for r in variant_records
    )
    total_question_retries = sum(
        r["metadata"].get("question_retry_count", 0) for r in variant_records
    )

    # Symbolic leakage in final outputs
    leakage_count = sum(
        1 for r in variant_records
        if has_symbolic_leakage(r["question"])
    )

    return {
        "num_examples": num_examples,
        "avg_num_nodes": num_nodes,
        "pipeline_version": pipeline_version,

        "abnormal_count": abnormal_count,
        "abnormal_rate": abnormal_count / num_examples,
        "verification_false_count": verification_false_count,

        "avg_treatment_outcome_depth": total_depth / num_examples,
        "avg_effective_prob_conds": total_eff_conds / num_examples,
        "naive_conditioning_rate": naive_count / num_examples,

        "qv_variant_counts": qv_counts,
        "qv_variant_rates": {
            k: v / num_examples for k, v in qv_counts.items()
        },

        "avg_entity_retries": total_entity_retries / num_examples,
        "avg_question_retries": total_question_retries / num_examples,
        "symbolic_leakage_count": leakage_count,
        "symbolic_leakage_rate": leakage_count / num_examples,

        "source_triplet_corpus": source_triplet_corpus,
    }


def write_nl_outputs(
    output_dir: str,
    filename: str,
    all_variant_records: Dict[str, List[Dict[str, Any]]],
    num_nodes: int,
    triplet_corpus_path: str,
    pipeline_version: str,
    variant_names: Optional[List[str]] = None,
) -> None:
    """Write JSONL, CSV, and stats JSON for each NL-domain variant.

    Always writes all three files even when records is empty, so that
    downstream integration can discover the file and report a clear
    count mismatch rather than a confusing FileNotFoundError.
    """
    for variant_name in (variant_names if variant_names is not None else NL_VARIANT_NAMES):
        records = all_variant_records.get(variant_name, [])

        folder_name = variant_to_folder(variant_name)

        jsonl_path = os.path.join(
            output_dir, folder_name, "jsonls", f"{filename}.jsonl",
        )
        csv_path = os.path.join(
            output_dir, folder_name, "csvs", f"{filename}.csv",
        )
        stats_path = os.path.join(
            output_dir, folder_name, "stats", f"{filename}.json",
        )

        # JSONL
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["idx", "question", "answer", "metadata"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow({
                    "idx": record["idx"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "metadata": json.dumps(
                        record["metadata"], ensure_ascii=False,
                    ),
                })

        # Stats
        stats = compute_nl_variant_stats(
            records, num_nodes, triplet_corpus_path, pipeline_version,
        )
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  {folder_name}: {jsonl_path} ({len(records)} records)")
