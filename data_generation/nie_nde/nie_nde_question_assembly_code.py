"""
nie_nde_question_assembly_code.py

NIE/NDE code-domain translation.

Implements mode-based binary-output functions so the final answer is recovered
as a difference in average outputs between two modes, analogous to the ATE
code-domain style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
from data_generation.nie_nde.nie_nde_question_assembly import (
    condition_variant_surgery_ancestor_prune as _symbolic_surgery_ancestor_prune,
)
from data_generation.nie_nde.nie_nde_query_sampling import NIENDEResult
from data_generation.scm_mechanism_sampling import SCMBase


CODE_VARIANT_NAMES = ['full', 'surgery_ancestor_prune']


@dataclass
class AssembledCodeQuestion:
    effect_type: str
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


def _render_qv_text(effect_type: str, qv_variant: int, func_name: str) -> str:
    if effect_type == 'nie':
        variants = {
            1: (
                f"Run `{func_name}` many times with mode set to 1 and, "
                f"separately, many times with mode set to 0. What is the "
                f"difference in the probability that `{func_name}` returns "
                f"outcome 1 between the two modes?"
            ),
            2: (
                f"Let p(m) be the probability that `{func_name}` returns "
                f"outcome 1 when mode equals m. Compute "
                f"p(1) - p(0)."
            ),
            3: (
                f"Interpret mode 1 and mode 0 as the two mediator-related "
                f"counterfactual regimes encoded by the code. What is the "
                f"difference in the probability of outcome 1 "
                f"between these two regimes?"
            ),
        }
    else:
        variants = {
            1: (
                f"Run `{func_name}` many times with mode set to 1 and, "
                f"separately, many times with mode set to 0. What is the "
                f"difference in the probability that `{func_name}` returns "
                f"outcome 1 between the two modes?"
            ),
            2: (
                f"Let p(m) be the probability that `{func_name}` returns "
                f"outcome 1 when mode equals m. Compute "
                f"p(1) - p(0)."
            ),
            3: (
                f"Interpret mode 1 and mode 0 as the two direct-effect "
                f"counterfactual regimes encoded by the code. What is the "
                f"difference in the probability of outcome 1 "
                f"between these two regimes?"
            ),
        }
    return variants[qv_variant]


def _render_code(
    scm: SCMBase,
    result: NIENDEResult,
    effect_type: str,
    nodes_kept: List[int],
    *,
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
) -> str:
    draw_name = draw_latents_name(vn_variant)
    build_name = build_factual_state_name(vn_variant)
    latent_name = latent_map_name(vn_variant)
    selector_nodes = list(nodes_kept)
    selector_name_map = build_selector_name_map(scm, selector_nodes, vn_variant)
    func_name = f"x{result.outcome}" if vn_variant == 'A' else 'func_main'

    lines = ['import random', '']
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
    lines.append('')

    lines.append(f"def {build_name}({latent_name}: dict[str, int], overrides: dict[str, int]) -> dict[str, int]:")
    lines.append('    state = {}')
    lines.extend(render_build_state_lines(scm, nodes_kept, {}, cf_variant=cf_variant, vn_variant=vn_variant, selector_name_map=selector_name_map))
    lines.append('')

    t = result.treatment
    m = result.mediator
    y = result.outcome
    lines.append(f"def {func_name}(mode: int, rng: random.Random) -> int:")
    lines.append('    if mode not in (0, 1):')
    lines.append('        raise ValueError("mode must be 0 or 1")')
    lines.append(f"    {latent_name} = {draw_name}(rng)")
    lines.append(f"    mediator_do0 = {build_name}({latent_name}, {{\"x{t}\": 0}})[\"x{m}\"]")
    lines.append(f"    mediator_do1 = {build_name}({latent_name}, {{\"x{t}\": 1}})[\"x{m}\"]")
    if effect_type == 'nie':
        lines.append('    if mode == 1:')
        lines.append(f"        outcome_state = {build_name}({latent_name}, {{\"x{t}\": 1, \"x{m}\": mediator_do1}})")
        lines.append('    else:')
        lines.append(f"        outcome_state = {build_name}({latent_name}, {{\"x{t}\": 1, \"x{m}\": mediator_do0}})")
    else:
        lines.append('    if mode == 1:')
        lines.append(f"        outcome_state = {build_name}({latent_name}, {{\"x{t}\": 1, \"x{m}\": mediator_do0}})")
        lines.append('    else:')
        lines.append(f"        outcome_state = {build_name}({latent_name}, {{\"x{t}\": 0, \"x{m}\": mediator_do0}})")
    lines.append(f"    return outcome_state[\"x{y}\"]")
    return '\n'.join(lines)


def assemble_all_questions_code(
    scm: SCMBase,
    result: NIENDEResult,
    *,
    effect_type: str,
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
        result.treatment,
        result.mediator,
        result.outcome,
        decimal_places=decimal_places,
    )
    # Symbolic surgery_ancestor_prune can omit the treatment node because the
    # CPT text still conditions on it explicitly. The code-domain renderer
    # cannot reuse that node set directly: it builds executable state updates,
    # and mediator/outcome mechanisms may still read the treatment variable
    # even when treatment is supplied through overrides. Add treatment back for
    # code generation so the emitted program remains self-consistent.
    sap_kept = sorted(set(symbolic_sap_variant.nodes_kept) | {result.treatment})
    sap_pruned = sorted(set(scm.nodes) - set(sap_kept))
    answer = f"{result.nie_value:.4f}" if effect_type == 'nie' else f"{result.nde_value:.4f}"
    questions = []
    for variant_name, structure, nodes_kept, nodes_pruned in [
        ('full', 'single', full_kept, full_pruned),
        ('surgery_ancestor_prune', 'multi', sap_kept, sap_pruned),
    ]:
        code = _render_code(scm, result, effect_type, nodes_kept, cf_variant=cf_variant, si_variant=si_variant, vn_variant=vn_variant)
        metadata = {
            'num_nodes': len(scm.nodes),
            'num_edges': scm.dag.number_of_edges(),
            'dag_edges': sorted(list(scm.dag.edges())),
            'treatment': result.treatment,
            'mediator': result.mediator,
            'outcome': result.outcome,
            'effect_type': effect_type,
            'query_type': effect_type,
            'domain': 'code',
            'variant_name': variant_name,
            'code_structure': structure,
            'cf_variant': cf_variant,
            'si_variant': si_variant,
            'vn_variant': vn_variant,
            'qv_variant': qv_variant,
            'is_abnormal': result.is_abnormal,
            'verification_ok': True,
            'nie_value': result.nie_value,
            'nde_value': result.nde_value,
            'nodes_kept': nodes_kept,
            'nodes_pruned': nodes_pruned,
            'treatment_outcome_depth': treatment_outcome_depth,
            'effective_prob_conds': effective_prob_conds,
            'has_naive_conditioning': has_naive_conditioning,
        }
        questions.append(AssembledCodeQuestion(
            effect_type=effect_type,
            variant_name=variant_name,
            answer=answer,
            code_structure=structure,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            qv_variant=qv_variant,
            full_question_text=compose_question_text(code, _render_qv_text(effect_type, qv_variant, f"x{result.outcome}" if vn_variant == 'A' else 'func_main')),
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            metadata=metadata,
        ))
    return questions
