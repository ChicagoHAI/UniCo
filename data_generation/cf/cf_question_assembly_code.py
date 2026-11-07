"""
cf_question_assembly_code.py

Counterfactual code-domain question assembly.

Translates symbolic counterfactual queries into code by making the shared
selector semantics explicit in code: one selector draw per node/parent
configuration, then a factual world and a counterfactual world built from the
same selectors.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
from typing import Dict, List, Optional, Tuple

from data_generation.cf.cf_query_sampling import CounterfactualQueryResult
from data_generation.cf.cf_question_assembly import (
    condition_variant_surgery_ancestor_prune as _symbolic_surgery_ancestor_prune,
)
from data_generation.dataset_utils.code_domain_renderer import (
    build_counterfactual_state_name,
    build_factual_state_name,
    build_selector_name_map,
    compose_question_text,
    compute_code_full_nodes,
    counterfactual_relevant_nodes,
    draw_latents_name,
    factual_relevant_nodes,
    latent_map_name,
    render_build_state_lines,
    render_sampling_line,
    render_selector_block,
    render_selector_lookup_declarations,
    sample_axes,
    selector_var_name,
)
from data_generation.scm_mechanism_sampling import SCMBase


CODE_VARIANT_NAMES = ["full", "surgery_ancestor_prune"]


@dataclass
class AssembledCodeQuestion:
    treatments: Dict[int, int]
    outcome: int
    evidence: Dict[int, int]
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


def _compute_nodes_surgery_ancestor_prune(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    if isinstance(evidence, int) and isinstance(outcome, dict):
        # Backward compatibility for older positional calls that used
        # (scm, treatments, evidence, outcome).
        outcome, evidence = evidence, outcome
    factual_nodes = factual_relevant_nodes(scm.dag, evidence, outcome)
    counterfactual_nodes = counterfactual_relevant_nodes(scm, treatments, outcome)
    kept = sorted(set(factual_nodes) | set(counterfactual_nodes))
    pruned = sorted(set(scm.nodes) - set(kept))
    return kept, pruned, factual_nodes, counterfactual_nodes


def _render_qv_text(qv_variant: int, func_name: str) -> str:
    variants = {
        1: (
            f"Considering only the calls where `{func_name}` does not return None, "
            f"what is the probability that it returns counterfactual outcome 1? "
            f"In other words, among the calls whose factual (baseline) world "
            f"matches the evidence, what fraction return 1 in the counterfactual "
            f"world built from the same latent draw?"
        ),
        2: (
            f"Run `{func_name}` many times, and discard every call that returns "
            f"None. Among the kept calls, what is the rate at which the returned "
            f"counterfactual outcome equals 1?"
        ),
        3: (
            f"Focus on the non-None outputs of `{func_name}`. Define p as the "
            f"probability that `{func_name}` returns counterfactual outcome 1 "
            f"(excluding calls where it returns None). What is p?"
        ),
    }
    return variants[qv_variant]


def _format_answer(cf_result: CounterfactualQueryResult) -> str:
    return f"{cf_result.prob_all_ones():.4f}"


def _render_code(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    factual_nodes: List[int],
    counterfactual_nodes: List[int],
    *,
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
) -> Tuple[str, str]:
    func_name = f"x{outcome}" if vn_variant == "A" else "func_main"
    draw_name = draw_latents_name(vn_variant)
    factual_build_name = build_factual_state_name(vn_variant)
    counterfactual_build_name = build_counterfactual_state_name(vn_variant)
    latent_name = latent_map_name(vn_variant)
    selector_set = set(factual_nodes) | set(counterfactual_nodes)
    topo_order = list(nx.topological_sort(scm.dag))
    selector_nodes = [node for node in topo_order if node in selector_set]
    selector_name_map = build_selector_name_map(scm, selector_nodes, vn_variant)
    lines = ["import random", "from typing import Optional", ""]
    lines.extend(render_selector_lookup_declarations(
        scm,
        selector_nodes,
        cf_variant=cf_variant,
        vn_variant=vn_variant,
        selector_name_map=selector_name_map,
    ))
    lines.append(f"def {draw_name}(rng: random.Random) -> dict[str, int]:")
    lines.extend(render_selector_block(scm, selector_nodes, si_variant, vn_variant, selector_name_map))
    selector_items = [
        f'"{selector_var_name(selector_name_map, node, config)}": {selector_var_name(selector_name_map, node, config)}'
        for node in selector_nodes
        for config in sorted(scm.cpts[node]['table'])
    ]
    lines.append(f"    return {{{', '.join(selector_items)}}}")
    lines.append("")
    lines.append(f"def {factual_build_name}({latent_name}: dict[str, int], overrides: dict[str, int]) -> dict[str, int]:")
    lines.append("    state = {}")
    lines.extend(render_build_state_lines(
        scm,
        factual_nodes,
        {},
        cf_variant=cf_variant,
        vn_variant=vn_variant,
        selector_name_map=selector_name_map,
    ))
    lines.append("")
    lines.append(f"def {counterfactual_build_name}({latent_name}: dict[str, int], overrides: dict[str, int]) -> dict[str, int]:")
    lines.append("    state = {}")
    lines.extend(render_build_state_lines(
        scm,
        counterfactual_nodes,
        treatments,
        cf_variant=cf_variant,
        vn_variant=vn_variant,
        selector_name_map=selector_name_map,
    ))
    lines.append("")
    lines.append(f"def {func_name}(rng: random.Random) -> Optional[int]:")
    lines.append(f"    {latent_name} = {draw_name}(rng)")
    lines.append(f"    baseline_state = {factual_build_name}({latent_name}, {{}})")
    if evidence:
        cond = ' and '.join(
            f'baseline_state["x{node}"] == {value}' for node, value in sorted(evidence.items())
        )
        lines.append(f"    accepted = {cond}")
    else:
        lines.append("    accepted = True")
    lines.append("    if not accepted:")
    lines.append("        return None")
    override_literal = '{' + ', '.join(
        f'"x{node}": {value}' for node, value in sorted(treatments.items())
    ) + '}'
    lines.append(f"    shifted_state = {counterfactual_build_name}({latent_name}, {override_literal})")
    lines.append(f"    return shifted_state[\"x{outcome}\"]")
    return "\n".join(lines), func_name


def assemble_all_questions_code(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    cf_result: CounterfactualQueryResult,
    *,
    decimal_places: int = 2,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledCodeQuestion]:
    cf_variant, si_variant, vn_variant, qv_variant = sample_axes(example_seed)
    full_kept, full_pruned, full_factual_nodes, full_counterfactual_nodes = compute_code_full_nodes(scm)
    symbolic_sap_variant = _symbolic_surgery_ancestor_prune(
        scm,
        treatments,
        outcome,
        evidence,
        decimal_places=decimal_places,
    )
    sap_kept = symbolic_sap_variant.nodes_kept
    sap_pruned = symbolic_sap_variant.nodes_pruned
    _, _, sap_factual_nodes, sap_counterfactual_nodes = _compute_nodes_surgery_ancestor_prune(
        scm,
        treatments,
        outcome,
        evidence,
    )

    results = []
    for variant_name, structure, nodes_kept, nodes_pruned, factual_nodes, counterfactual_nodes in [
        ('full', 'single', full_kept, full_pruned, full_factual_nodes, full_counterfactual_nodes),
        ('surgery_ancestor_prune', 'multi', sap_kept, sap_pruned, sap_factual_nodes, sap_counterfactual_nodes),
    ]:
        code, func_name = _render_code(
            scm,
            treatments,
            outcome,
            evidence,
            factual_nodes,
            counterfactual_nodes,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
        )

        metadata = {
            'num_nodes': len(scm.nodes),
            'num_edges': scm.dag.number_of_edges(),
            'dag_edges': sorted(list(scm.dag.edges())),
            'treatments': {str(k): v for k, v in sorted(treatments.items())},
            'outcome': outcome,
            'evidence': {str(k): v for k, v in sorted(evidence.items())},
            'query_type': 'cf',
            'domain': 'code',
            'variant_name': variant_name,
            'code_structure': structure,
            'cf_variant': cf_variant,
            'si_variant': si_variant,
            'vn_variant': vn_variant,
            'qv_variant': qv_variant,
            'is_abnormal': not cf_result.normalization_ok or False,
            'verification_ok': cf_result.normalization_ok,
            'cf_result': cf_result.to_json_dict(),
            'nodes_kept': nodes_kept,
            'nodes_pruned': nodes_pruned,
            'treatment_outcome_depth': treatment_outcome_depth,
            'effective_prob_conds': effective_prob_conds,
            'has_naive_conditioning': has_naive_conditioning,
        }
        results.append(
            AssembledCodeQuestion(
                treatments=treatments,
                outcome=outcome,
                evidence=evidence,
                variant_name=variant_name,
                answer=_format_answer(cf_result),
                code_structure=structure,
                cf_variant=cf_variant,
                si_variant=si_variant,
                vn_variant=vn_variant,
                qv_variant=qv_variant,
                full_question_text=compose_question_text(code, _render_qv_text(qv_variant, func_name)),
                nodes_kept=nodes_kept,
                nodes_pruned=nodes_pruned,
                metadata=metadata,
            )
        )

    return results
