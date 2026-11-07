"""NL post-validator for quality checks on generated NL questions.

Runs three checks on final NL question JSONL outputs:
1. **Symbolic token check** (mandatory): flags questions with >= 3 X\\d+
   occurrences.
2. **Entity coverage check** (optional, off by default): checks what
   fraction of entity names appear in the NL question.
3. **Numeric value coverage check** (configurable, default 100%):
   verifies that SCM probability values appear as decimals or
   percentage numbers (including trailing-zero-stripped and
   leading-zero percentage forms).

Usage (from the repository root)::

    python3 -m data_generation.dataset_utils.nl_post_validator \
        --input_path <nl_output.jsonl> \
        --triplet_corpus_path <triplets.jsonl> \
        --report_path <report.json> \
        [--numeric_coverage_threshold 0.8] \
        [--enable_entity_coverage]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

from data_generation.dataset_utils.nl_pipeline_shared import (
    count_symbolic_tokens,
    has_symbolic_leakage,
)
from data_generation.dataset_utils.nl_prompt_builder import (
    build_dag,
    get_full_condition_nodes,
    get_surgery_ancestor_prune_nodes_ate,
    get_surgery_ancestor_prune_nodes_cate_jate,
)
from data_generation.dataset_utils.nl_query_specs import is_v2_2_query_type
from data_generation.dataset_utils.nl_triplet_adapters import (
    get_condition_nodes as get_v2_2_condition_nodes,
)


# -------------------------------------------------------------------
# Numeric value coverage
# -------------------------------------------------------------------

def _extract_scm_values(
    cpts: Dict[str, Dict[str, Any]],
    condition_nodes: List[int] | None = None,
    decimal_places: int | None = None,
) -> List[str]:
    """Extract probability values from CPTs for *condition_nodes*.

    When *condition_nodes* is ``None``, extracts from all nodes (full variant).
    When *decimal_places* is provided, formats values with fixed precision
    (preserving trailing zeros to match the rendered question text).
    Returns a list (not set) to preserve duplicates: repeated CPT values
    from different conditional rows are counted separately for coverage.
    """
    if condition_nodes is not None:
        node_keys = {str(n) for n in condition_nodes}
    else:
        node_keys = None

    values: List[str] = []
    for node_str, entry in cpts.items():
        if node_keys is not None and node_str not in node_keys:
            continue
        for key_str, prob in entry["table"].items():
            if decimal_places is not None:
                values.append(f"{prob:.{decimal_places}f}")
            else:
                values.append(str(prob))
    return values


def _numeric_boundary_match(needle: str, haystack: str) -> bool:
    """Check if *needle* appears in *haystack* at a numeric word boundary.

    A numeric boundary means the character before/after the match (if any)
    is not a digit or period, preventing ``0.2`` from matching inside
    ``120`` or ``0.05`` from matching inside ``0.050001``.
    """
    pattern = r"(?<![0-9.])" + re.escape(needle) + r"(?![0-9]|\.(?=\d))"
    return re.search(pattern, haystack) is not None


def _count_numeric_occurrences(needle: str, haystack: str) -> int:
    """Count non-overlapping boundary-safe occurrences of *needle* in *haystack*.

    Excludes matches immediately followed by a percentage cue (``%`` or
    ``percent``), so that ``0.35`` in ``"0.35%"`` is not counted as a
    decimal match (0.35% = 0.0035, not 0.35).
    """
    pattern = (
        r"(?<![0-9.])" + re.escape(needle)
        + r"(?![0-9]|\.(?=\d))"
        + r"(?!\s*(?:%|percent\b))"
    )
    return len(re.findall(pattern, haystack, re.IGNORECASE))


def _count_pct_occurrences(needle: str, haystack: str) -> int:
    """Count occurrences of *needle* followed by a percentage cue (``%`` or ``percent``).

    Uses ``percent\\b`` to avoid matching ``percentile`` or ``percentage``.
    """
    pattern = r"(?<![0-9.])" + re.escape(needle) + r"(?![0-9]|\.(?=\d))\s*(?:%|percent\b)"
    return len(re.findall(pattern, haystack, re.IGNORECASE))


def _check_numeric_coverage(
    question: str,
    scm_values: List[str],
) -> float:
    """Return fraction of unique SCM values found in *question* (decimal or %).

    For each value like 0.3635, checks for:
    - "0.3635" (decimal form)
    - "36.35" followed by % (percentage form)

    Percentage candidates are generated at precision levels from
    ``max(1, dec_places - 2)`` to ``dec_places + 2``.  The lower bound
    is the minimum precision that preserves the exact numeric value
    (never 0, which would round).  Each candidate is also tried with
    trailing zeros stripped, covering renderings like ``0.7500`` → "75%",
    ``0.110`` → "11%".  For percentage values in [1, 10), a leading-zero
    variant is also tried (e.g., ``0.038`` → "03.8%", ``0.0936`` →
    "09.36%").

    Uses word-boundary matching to avoid false positives where a short
    numeric string (e.g., ``0.2``) appears inside a longer number.

    Each unique SCM probability value must appear at least once to get
    credit; duplicates in the CPT are collapsed (e.g., if the SCM has
    three rows with 0.5, the question only needs to mention it once).
    """
    if not scm_values:
        return 1.0

    unique_values = set(scm_values)

    # Pre-count occurrences in the question for each unique value's
    # possible text representations.
    budget: dict[str, int] = {}
    for val_str in unique_values:
        val = float(val_str)
        dec_places = len(val_str.split(".")[-1]) if "." in val_str else 0
        pct_val = val * 100

        # Sum occurrences across decimal and percentage forms.
        # Percentage forms require a '%' or 'percent' cue to avoid
        # false positives with bare numbers (counts, years, etc.).
        total = _count_numeric_occurrences(val_str, question)

        # Generate percentage candidates at precision levels that
        # preserve the exact value.  min_pct_places = max(1, dp - 2)
        # is the fewest decimal places that avoid rounding (e.g.,
        # dp=4 → min 2, so 0.1234 only tries "12.34" not "12.3").
        # Trailing-zero-stripped forms cover "75%" for 0.7500 etc.
        min_pct_places = max(1, dec_places - 2)
        seen_candidates: set[str] = set()
        for places in range(min_pct_places, dec_places + 3):
            candidate = f"{pct_val:.{places}f}"
            stripped = candidate.rstrip("0").rstrip(".")
            for c in (candidate, stripped):
                if c not in seen_candidates and c != val_str:
                    seen_candidates.add(c)
                    total += _count_pct_occurrences(c, question)
                # For single-digit integer percentage (1% ≤ val < 10%),
                # also try with a leading zero (e.g., "03.8%", "09.36%").
                if 1 <= pct_val < 10:
                    lz = "0" + c
                    if lz not in seen_candidates and lz != val_str:
                        seen_candidates.add(lz)
                        total += _count_pct_occurrences(lz, question)

        budget[val_str] = total

    # Each unique SCM value needs at least one occurrence
    found = sum(1 for val_str in unique_values if budget.get(val_str, 0) >= 1)

    return found / len(unique_values)


# -------------------------------------------------------------------
# Entity coverage
# -------------------------------------------------------------------

def _check_entity_coverage(
    question: str,
    entity_mapping: Dict[str, Any],
) -> float:
    """Return fraction of entity names that appear in *question*."""
    if not entity_mapping:
        return 1.0

    found = 0
    total = 0
    for node_key, node_info in entity_mapping.items():
        entity_name = node_info.get("entity", "")
        if entity_name:
            total += 1
            if entity_name.lower() in question.lower():
                found += 1

    return found / total if total > 0 else 1.0


# -------------------------------------------------------------------
# Condition-variant node computation
# -------------------------------------------------------------------

def _get_condition_nodes(
    metadata: Dict[str, Any],
    triplet: Dict[str, Any],
) -> List[int] | None:
    """Compute the nodes whose CPTs were rendered for this variant.

    Returns ``None`` for the ``full`` variant (all nodes).
    """
    variant = metadata.get("condition_variant", "full")
    if variant == "full":
        return None

    dag = build_dag(triplet["dag_edges"], triplet["num_nodes"])
    query_type = metadata.get("query_type", "ate")

    if is_v2_2_query_type(query_type):
        return get_v2_2_condition_nodes(triplet, query_type, variant)

    if query_type == "ate":
        return get_surgery_ancestor_prune_nodes_ate(
            dag, metadata["treatment"], metadata["outcome"],
        )
    else:
        # CATE and JATE
        treatments = metadata.get("treatments", [metadata.get("treatment")])
        evidence_raw = metadata.get("evidence", {})
        evidence = {int(k): v for k, v in evidence_raw.items()}
        return get_surgery_ancestor_prune_nodes_cate_jate(
            dag, treatments, metadata["outcome"], evidence,
        )


# -------------------------------------------------------------------
# Main validator
# -------------------------------------------------------------------

def validate_record(
    record: Dict[str, Any],
    triplet: Dict[str, Any],
    *,
    numeric_coverage_threshold: float = 1.0,
    enable_entity_coverage: bool = False,
    entity_coverage_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Validate a single NL question record.

    Returns a dict with check results.
    """
    question = record["question"]
    metadata = record["metadata"]
    issues: List[str] = []

    # 1. Symbolic token check (mandatory)
    sym_count = count_symbolic_tokens(question)
    sym_pass = sym_count < 3
    if not sym_pass:
        issues.append(f"symbolic_leakage ({sym_count} tokens)")

    # 2. Numeric value coverage (scoped to rendered condition variant)
    condition_nodes = _get_condition_nodes(metadata, triplet)
    decimal_places = triplet.get("decimal_places")
    scm_values = _extract_scm_values(triplet["cpts"], condition_nodes, decimal_places)
    numeric_coverage = _check_numeric_coverage(question, scm_values)
    numeric_pass = numeric_coverage >= numeric_coverage_threshold
    if not numeric_pass:
        issues.append(
            f"numeric_coverage ({numeric_coverage:.1%} < {numeric_coverage_threshold:.0%})"
        )

    # 3. Entity coverage (optional)
    entity_coverage = None
    entity_pass = True
    if enable_entity_coverage:
        entity_mapping = metadata.get("entity_mapping", {})
        entity_coverage = _check_entity_coverage(question, entity_mapping)
        entity_pass = entity_coverage >= entity_coverage_threshold
        if not entity_pass:
            issues.append(
                f"entity_coverage ({entity_coverage:.1%} < {entity_coverage_threshold:.0%})"
            )

    return {
        "idx": record["idx"],
        "symbolic_token_count": sym_count,
        "symbolic_pass": sym_pass,
        "numeric_coverage": round(numeric_coverage, 4),
        "numeric_pass": numeric_pass,
        "entity_coverage": round(entity_coverage, 4) if entity_coverage is not None else None,
        "entity_pass": entity_pass,
        "overall_pass": sym_pass and numeric_pass and entity_pass,
        "issues": issues,
    }


def generate_report(
    validation_results: List[Dict[str, Any]],
    input_path: str,
    numeric_threshold: float,
    enable_entity_coverage: bool,
    entity_threshold: float,
    expected_total: int | None = None,
) -> Dict[str, Any]:
    """Generate an aggregate quality report.

    When *expected_total* is provided, pass rates are computed against
    that denominator (successes + failures) rather than against only the
    validated records.  This prevents pass rates from being artificially
    1.0 when step-3 failures are excluded from the input JSONL.
    """
    num_validated = len(validation_results)
    total = expected_total if expected_total is not None else num_validated
    if total == 0:
        return {
            "input_path": input_path,
            "total_records": 0,
            "overall_pass_count": 0,
            "overall_pass_rate": 0,
            "symbolic_check": {
                "pass_count": 0,
                "pass_rate": 0,
                "avg_token_count": 0,
            },
            "numeric_coverage_check": {
                "threshold": numeric_threshold,
                "pass_count": 0,
                "pass_rate": 0,
                "avg_coverage": 0,
            },
            "entity_coverage_check": {
                "enabled": enable_entity_coverage,
                "threshold": entity_threshold if enable_entity_coverage else None,
                "pass_count": 0 if enable_entity_coverage else None,
                "pass_rate": 0 if enable_entity_coverage else None,
            },
            "flagged_count": 0,
            "flagged_indices": [],
        }

    sym_pass = sum(1 for r in validation_results if r["symbolic_pass"])
    num_pass = sum(1 for r in validation_results if r["numeric_pass"])
    ent_pass = sum(1 for r in validation_results if r["entity_pass"])
    overall_pass = sum(1 for r in validation_results if r["overall_pass"])

    # Averages are computed over validated records only (not failures)
    avg_numeric = sum(r["numeric_coverage"] for r in validation_results) / num_validated if num_validated else 0
    avg_sym_tokens = sum(r["symbolic_token_count"] for r in validation_results) / num_validated if num_validated else 0

    flagged = [r for r in validation_results if not r["overall_pass"]]
    failure_count = total - num_validated

    report = {
        "input_path": input_path,
        "total_records": total,
        "validated_records": num_validated,
        "failure_count": failure_count,
        "overall_pass_count": overall_pass,
        "overall_pass_rate": round(overall_pass / total, 4),
        "symbolic_check": {
            "pass_count": sym_pass,
            "pass_rate": round(sym_pass / total, 4),
            "avg_token_count": round(avg_sym_tokens, 2),
        },
        "numeric_coverage_check": {
            "threshold": numeric_threshold,
            "pass_count": num_pass,
            "pass_rate": round(num_pass / total, 4),
            "avg_coverage": round(avg_numeric, 4),
        },
        "entity_coverage_check": {
            "enabled": enable_entity_coverage,
            "threshold": entity_threshold if enable_entity_coverage else None,
            "pass_count": ent_pass if enable_entity_coverage else None,
            "pass_rate": round(ent_pass / total, 4) if enable_entity_coverage else None,
        },
        "flagged_count": len(flagged),
        "flagged_indices": [r["idx"] for r in flagged],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NL post-validator: quality checks on generated NL questions",
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Final NL question JSONL to validate")
    parser.add_argument("--triplet_corpus_path", type=str, required=True,
                        help="Source triplet JSONL (for numeric value extraction)")
    parser.add_argument("--report_path", type=str, required=True,
                        help="Output path for validation report JSON")
    parser.add_argument("--numeric_coverage_threshold", type=float, default=1.0)
    parser.add_argument("--enable_entity_coverage", action="store_true")
    parser.add_argument("--entity_coverage_threshold", type=float, default=0.5)
    parser.add_argument("--failure_count", type=int, default=0,
                        help="Number of step-3 failures (added to denominator for pass rates)")
    parser.add_argument("--expected_records", type=int, default=0,
                        help="Expected total records for this variant block "
                             "(accounts for all upstream dropouts; overrides --failure_count)")
    args = parser.parse_args()

    print("=== NL Post-Validator ===")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # Load NL questions
    nl_records = []
    with open(args.input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                nl_records.append(json.loads(line))
    print(f"\nLoaded {len(nl_records)} NL question records")

    # Load triplets and build index
    triplet_map: Dict[int, Dict[str, Any]] = {}
    with open(args.triplet_corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                triplet_map[rec["idx"]] = rec
    print(f"Loaded {len(triplet_map)} triplet records")

    # Validate each record
    results = []
    for nl_rec in nl_records:
        idx = nl_rec["idx"]
        if idx not in triplet_map:
            print(f"  Warning: idx {idx} not in triplet corpus, skipping")
            continue
        result = validate_record(
            nl_rec, triplet_map[idx],
            numeric_coverage_threshold=args.numeric_coverage_threshold,
            enable_entity_coverage=args.enable_entity_coverage,
            entity_coverage_threshold=args.entity_coverage_threshold,
        )
        results.append(result)

    # Generate and write report
    if args.expected_records > 0:
        expected_total = args.expected_records
    elif args.failure_count > 0:
        expected_total = len(results) + args.failure_count
    else:
        expected_total = None
    report = generate_report(
        results, args.input_path,
        args.numeric_coverage_threshold,
        args.enable_entity_coverage,
        args.entity_coverage_threshold,
        expected_total=expected_total,
    )

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\n=== Report ===")
    print(f"  Total: {report['total_records']}")
    print(f"  Overall pass: {report['overall_pass_count']} "
          f"({report['overall_pass_rate']:.1%})")
    print(f"  Symbolic: {report['symbolic_check']['pass_count']} pass")
    print(f"  Numeric: {report['numeric_coverage_check']['pass_count']} pass "
          f"(avg coverage: {report['numeric_coverage_check']['avg_coverage']:.1%})")
    if args.enable_entity_coverage:
        print(f"  Entity: {report['entity_coverage_check']['pass_count']} pass")
    print(f"  Flagged: {report['flagged_count']}")
    print(f"  Report written to: {args.report_path}")


if __name__ == "__main__":
    main()
