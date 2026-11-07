"""
Shared code-domain assembly helpers for numerical association queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from data_generation.dataset_utils.association_assembly_shared import ancestor_closure
from data_generation.dataset_utils.code_domain_renderer import (
    compose_question_text,
    get_observational_entry_function_name,
    render_observational_code,
    sample_axes,
)
from data_generation.scm_mechanism_sampling import SCMBase


ASSOCIATION_CODE_VARIANT_NAMES = ["full", "query_ancestor_prune"]


@dataclass
class AssembledAssociationCodeQuestion:
    query_type: str
    variant_name: str
    answer: str
    code_structure: str
    cf_variant: str
    si_variant: str
    vn_variant: str
    qv_variant: int
    full_question_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


def _query_components(query_type: str, result) -> Dict:
    if query_type == "marginal_prob":
        target = result.target
        return {
            "return_nodes": [target],
            "evidence": {},
            "core_nodes": {target},
            "answer": result.probability,
            "result_fields": {"target": target, "probability": result.probability},
        }
    if query_type == "conditional_prob":
        target = result.target
        evidence = dict(result.evidence)
        return {
            "return_nodes": [target],
            "evidence": evidence,
            "core_nodes": {target, *evidence.keys()},
            "answer": result.probability,
            "result_fields": {"target": target, "evidence": evidence, "probability": result.probability},
        }
    if query_type == "joint_prob":
        target_nodes = list(result.target_nodes)
        return {
            "return_nodes": target_nodes,
            "evidence": {},
            "core_nodes": set(target_nodes),
            "answer": result.probability,
            "result_fields": {"target_nodes": target_nodes, "probability": result.probability},
        }
    if query_type == "observed_difference":
        return {
            "return_nodes": [result.source, result.target],
            "evidence": {},
            "core_nodes": {result.source, result.target},
            "answer": result.observed_difference,
            "result_fields": {
                "source": result.source,
                "target": result.target,
                "observed_difference": result.observed_difference,
            },
        }
    raise ValueError(f"Unsupported association code query_type={query_type}")


def _render_qv_text(query_type: str, result, qv_variant: int, func_name: str) -> str:
    if query_type == "marginal_prob":
        variants = {
            1: (
                f"What is the expected return value of `{func_name}`? In other "
                f"words, what is the probability that `{func_name}` returns 1?"
            ),
            2: (
                f"Suppose you call `{func_name}` a large number of times. What "
                f"fraction of calls return 1?"
            ),
            3: (
                f"Let p denote the probability that `{func_name}` returns 1. "
                f"Compute p."
            ),
        }
    elif query_type == "conditional_prob":
        variants = {
            1: (
                f"Considering only the calls where `{func_name}` does not return "
                f"None, what is the expected return value? In other words, what "
                f"is the probability that `{func_name}` returns 1, among "
                f"non-None results?"
            ),
            2: (
                f"Run `{func_name}` many times, and discard every call that "
                f"returns None. Among the kept results, what is the rate of "
                f"returning 1?"
            ),
            3: (
                f"Focus on the non-None outputs of `{func_name}`. Define p as "
                f"the probability that `{func_name}` returns 1, excluding calls "
                f"where it returns None. What is p?"
            ),
        }
    elif query_type == "joint_prob":
        variants = {
            1: (
                f"What is the probability that `{func_name}` returns all 1s?"
            ),
            2: (
                f"Suppose you call `{func_name}` a large number of times. What "
                f"fraction of calls return all 1s?"
            ),
            3: (
                f"Let p denote the probability that every value returned by "
                f"`{func_name}` equals 1. Compute p."
            ),
        }
    elif query_type == "observed_difference":
        variants = {
            1: (
                f"Each call to `{func_name}` returns a tuple `(s, y)`. The first "
                f"returned value `s` is the observed source value, and the "
                f"second returned value `y` is the target value. What is the "
                f"signed difference in the probability that `y` equals 1 when "
                f"`s` equals 1 compared to when `s` equals 0?"
            ),
            2: (
                f"Suppose you call `{func_name}` a large number of times. Each "
                f"call returns `(s, y)`, where `s` is the first returned value "
                f"and `y` is the second returned value. Among calls where "
                f"`s = 1`, compute the fraction with `y = 1`. Separately, do "
                f"the same among calls where `s = 0`. What is the first "
                f"fraction minus the second?"
            ),
            3: (
                f"Let `(s, y)` be the tuple returned by `{func_name}`. Define "
                f"p(x) as the probability that `y = 1` among calls where "
                f"`s = x`. Compute p(1) - p(0). Keep the sign; do not take "
                f"an absolute value."
            ),
        }
    else:
        raise ValueError(f"Unsupported association code query_type={query_type}")
    return variants[qv_variant]


def assemble_association_code_questions(
    scm: SCMBase,
    result,
    *,
    query_type: str,
    decimal_places: int,
    example_seed: int,
    example_idx: int | None = None,
    is_abnormal: bool | None = None,
    treatment_outcome_depth: float = 0.0,
    effective_prob_conds: int = 0,
    has_naive_conditioning: bool = False,
) -> List[AssembledAssociationCodeQuestion]:
    del example_idx
    cf_variant, si_variant, vn_variant, qv_variant = sample_axes(example_seed)
    components = _query_components(query_type, result)
    all_nodes = sorted(scm.nodes)
    ancestor_nodes = sorted(ancestor_closure(scm.dag, components["core_nodes"]))
    variants = [
        ("full", "single", all_nodes, []),
        ("query_ancestor_prune", "multi", ancestor_nodes, sorted(set(all_nodes) - set(ancestor_nodes))),
    ]
    questions = []
    for variant_name, code_structure, nodes_kept, nodes_pruned in variants:
        code = render_observational_code(
            scm.dag,
            scm.cpts,
            query_type=query_type,
            return_nodes=components["return_nodes"],
            evidence=components["evidence"],
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            structure=code_structure,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            decimal_places=decimal_places,
            example_seed=example_seed,
        )
        func_name = get_observational_entry_function_name(
            components["return_nodes"],
            vn_variant,
            query_type=query_type,
        )
        metadata = {
            "num_nodes": len(scm.nodes),
            "num_edges": scm.dag.number_of_edges(),
            "dag_edges": sorted(list(scm.dag.edges())),
            "query_type": query_type,
            "domain": "code",
            "variant_name": variant_name,
            "code_structure": code_structure,
            "cf_variant": cf_variant,
            "si_variant": si_variant,
            "vn_variant": vn_variant,
            "qv_variant": qv_variant,
            "is_abnormal": bool(is_abnormal),
            "verification_ok": result.verification_ok,
            "nodes_kept": nodes_kept,
            "nodes_pruned": nodes_pruned,
            "treatment_outcome_depth": treatment_outcome_depth,
            "effective_prob_conds": effective_prob_conds,
            "has_naive_conditioning": has_naive_conditioning,
        }
        metadata.update(components["result_fields"])
        questions.append(AssembledAssociationCodeQuestion(
            query_type=query_type,
            variant_name=variant_name,
            answer=f"{components['answer']:.4f}",
            code_structure=code_structure,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            qv_variant=qv_variant,
            full_question_text=compose_question_text(code, _render_qv_text(query_type, result, qv_variant, func_name)),
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            metadata=metadata,
        ))
    return questions
