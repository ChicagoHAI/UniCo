"""
att_question_assembly_code.py

ATT code-domain translation built on the same shared-latent worldview as the
counterfactual code-domain translator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from data_generation.att.att_query_sampling import ATTResult
from data_generation.att.att_question_assembly import (
    condition_variant_surgery_ancestor_prune as _symbolic_surgery_ancestor_prune,
)
from data_generation.dataset_utils.code_domain_renderer import (
    build_counterfactual_state_name,
    build_factual_state_name,
    build_selector_name_map,
    compose_question_text,
    compute_code_full_nodes,
    draw_latents_name,
    latent_map_name,
    render_build_state_lines,
    render_sampling_line,
    render_selector_lookup_declarations,
    sample_axes,
    selector_var_name,
)
from data_generation.scm_mechanism_sampling import SCMBase


CODE_VARIANT_NAMES = ["full", "surgery_ancestor_prune"]


@dataclass
class AssembledCodeQuestion:
    treatment: int
    outcome: int
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


def _render_att_qv_text(qv_variant: int, func_name: str) -> str:
    variants = {
        1: (
            f"Considering only the calls where `{func_name}` does not return "
            f"None, how does the input affect the expected return value? In "
            f"other words, among the runs whose factual (baseline) world is "
            f"treated, what is the change in the probability that "
            f"`{func_name}` returns counterfactual outcome 1 as the input "
            f"changes from 0 to 1?"
        ),
        2: (
            f"Run `{func_name}` many times with the input set to 1, and "
            f"discard every call that returns None. Separately, do the same "
            f"with the input set to 0. Among the kept calls in each group, "
            f"what is the difference in the rate of returning counterfactual "
            f"outcome 1?"
        ),
        3: (
            f"Focus on the non-None outputs of `{func_name}`. Define p(x) as "
            f"the probability that `{func_name}` returns counterfactual "
            f"outcome 1 (excluding calls where it returns None) when the input "
            f"is fixed to x. What is p(1) - p(0)?"
        ),
    }
    return variants[qv_variant]


def _render_code(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    nodes_kept: List[int],
    *,
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
) -> str:
    draw_name = draw_latents_name(vn_variant)
    factual_build_name = build_factual_state_name(vn_variant)
    counterfactual_build_name = build_counterfactual_state_name(vn_variant)
    latent_name = latent_map_name(vn_variant)
    selector_nodes = list(nodes_kept)
    selector_name_map = build_selector_name_map(scm, selector_nodes, vn_variant)
    func_name = f"x{outcome}" if vn_variant == 'A' else 'func_main'

    lines = ["import random", "from typing import Optional", ""]
    lines.extend(render_selector_lookup_declarations(scm, selector_nodes, cf_variant, vn_variant, selector_name_map))
    lines.append(f"def {draw_name}(rng: random.Random) -> dict[str, int]:")
    for node in selector_nodes:
        for config, prob in sorted(scm.cpts[node]['table'].items()):
            selector_name = selector_var_name(selector_name_map, node, config)
            lines.append(render_sampling_line(selector_name, prob, si_variant, indent='    '))
    selector_items = [
        f'"{selector_var_name(selector_name_map, node, config)}": {selector_var_name(selector_name_map, node, config)}'
        for node in selector_nodes
        for config in sorted(scm.cpts[node]['table'])
    ]
    lines.append(f"    return {{{', '.join(selector_items)}}}")
    lines.append("")

    lines.append(f"def {factual_build_name}({latent_name}: dict[str, int], overrides: dict[str, int]) -> dict[str, int]:")
    lines.append("    state = {}")
    lines.extend(render_build_state_lines(scm, nodes_kept, {}, cf_variant=cf_variant, vn_variant=vn_variant, selector_name_map=selector_name_map))
    lines.append("")

    lines.append(f"def {counterfactual_build_name}({latent_name}: dict[str, int], overrides: dict[str, int]) -> dict[str, int]:")
    lines.append("    state = {}")
    lines.extend(render_build_state_lines(scm, nodes_kept, {treatment: 0}, cf_variant=cf_variant, vn_variant=vn_variant, selector_name_map=selector_name_map))
    lines.append("")

    lines.append(f"def {func_name}(probe_value: int, rng: random.Random) -> Optional[int]:")
    lines.append("    if probe_value not in (0, 1):")
    lines.append("        raise ValueError(\"probe_value must be 0 or 1\")")
    lines.append(f"    {latent_name} = {draw_name}(rng)")
    lines.append(f"    baseline_state = {factual_build_name}({latent_name}, {{}})")
    lines.append(f"    accepted = baseline_state[\"x{treatment}\"] == 1")
    lines.append("    if not accepted:")
    lines.append("        return None")
    lines.append(f"    shifted_state = {counterfactual_build_name}({latent_name}, {{\"x{treatment}\": probe_value}})")
    lines.append(f"    return shifted_state[\"x{outcome}\"]")
    return "\n".join(lines)


def assemble_all_questions_code(
    scm: SCMBase,
    att_result: ATTResult,
    *,
    decimal_places: int = 2,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledCodeQuestion]:
    cf_variant, si_variant, vn_variant, qv_variant = sample_axes(example_seed)
    full_kept, full_pruned, _, _ = compute_code_full_nodes(scm)
    symbolic_sap_variant = _symbolic_surgery_ancestor_prune(
        scm,
        att_result.treatment,
        att_result.outcome,
        decimal_places=decimal_places,
    )
    sap_kept = symbolic_sap_variant.nodes_kept
    sap_pruned = symbolic_sap_variant.nodes_pruned
    answer = f"{att_result.att_value:.4f}"
    results = []
    for variant_name, structure, nodes_kept, nodes_pruned in [
        ('full', 'single', full_kept, full_pruned),
        ('surgery_ancestor_prune', 'multi', sap_kept, sap_pruned),
    ]:
        code = _render_code(
            scm,
            att_result.treatment,
            att_result.outcome,
            nodes_kept,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
        )
        metadata = {
            'num_nodes': len(scm.nodes),
            'num_edges': scm.dag.number_of_edges(),
            'dag_edges': sorted(list(scm.dag.edges())),
            'treatment': att_result.treatment,
            'outcome': att_result.outcome,
            'query_type': 'att',
            'domain': 'code',
            'variant_name': variant_name,
            'code_structure': structure,
            'cf_variant': cf_variant,
            'si_variant': si_variant,
            'vn_variant': vn_variant,
            'qv_variant': qv_variant,
            'is_abnormal': att_result.is_abnormal,
            'verification_ok': att_result.cf_result_do1.normalization_ok and att_result.cf_result_do0.normalization_ok,
            'att_value': att_result.att_value,
            'cf_result_do1': att_result.cf_result_do1.to_json_dict(),
            'cf_result_do0': att_result.cf_result_do0.to_json_dict(),
            'nodes_kept': nodes_kept,
            'nodes_pruned': nodes_pruned,
            'treatment_outcome_depth': treatment_outcome_depth,
            'effective_prob_conds': effective_prob_conds,
            'has_naive_conditioning': has_naive_conditioning,
        }
        results.append(AssembledCodeQuestion(
            treatment=att_result.treatment,
            outcome=att_result.outcome,
            variant_name=variant_name,
            answer=answer,
            code_structure=structure,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            qv_variant=qv_variant,
            full_question_text=compose_question_text(code, _render_att_qv_text(qv_variant, f"x{att_result.outcome}" if vn_variant == 'A' else 'func_main')),
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            metadata=metadata,
        ))
    return results
