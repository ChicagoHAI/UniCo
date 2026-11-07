"""
pn_ps_question_assembly_code.py

PN/PS code-domain question assembly.

Translates PN/PS triplets into shared-latent code questions. The entry
function returns ``None`` when the factual evidence is not satisfied and
returns the intervened outcome value otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from data_generation.dataset_utils.code_domain_renderer import (
    build_counterfactual_state_name,
    build_factual_state_name,
    build_selector_name_map,
    compose_question_text,
    compute_code_full_nodes,
    draw_latents_name,
    latent_map_name,
    render_build_state_lines,
    render_selector_block,
    render_selector_lookup_declarations,
    sample_axes,
    selector_var_name,
)
from data_generation.pn_ps.pn_ps_query_sampling import PNPSResult
from data_generation.scm_mechanism_sampling import SCMBase


CODE_VARIANT_NAMES = ["full", "surgery_ancestor_prune"]


@dataclass
class AssembledCodeQuestion:
    effect_type: str
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


def _effect_spec(result: PNPSResult, effect_type: str) -> Tuple[Dict[int, int], Dict[int, int], int, float, bool]:
    """Return factual evidence, intervention, queried return value, answer, and normalization flag."""
    if effect_type == "pn":
        return (
            {result.treatment: 1, result.outcome: 1},
            {result.treatment: 0},
            0,
            result.pn_value,
            result.cf_result_pn.normalization_ok,
        )
    if effect_type == "ps":
        return (
            {result.treatment: 0, result.outcome: 0},
            {result.treatment: 1},
            1,
            result.ps_value,
            result.cf_result_ps.normalization_ok,
        )
    raise ValueError(f"Unknown PN/PS effect_type: {effect_type}")


def _ancestor_closure(dag: nx.DiGraph, nodes: set[int]) -> List[int]:
    closed = set(nodes)
    for node in list(nodes):
        closed.update(nx.ancestors(dag, node))
    return sorted(closed)


def _compute_nodes_surgery_ancestor_prune(
    scm: SCMBase,
    factual_evidence: Dict[int, int],
    intervention: Dict[int, int],
    outcome: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    # Factual state must materialize every evidence node that is read in the
    # acceptance check, plus all parents/ancestors needed to build those nodes.
    factual_nodes = _ancestor_closure(scm.dag, set(factual_evidence))

    # Counterfactual state must materialize the returned outcome and every
    # intervention node used as an override, plus all ancestors needed for the
    # returned outcome after the override is applied. Including intervention
    # nodes is harmless and keeps override handling explicit in the generated code.
    counterfactual_nodes = _ancestor_closure(scm.dag, set(intervention) | {outcome})
    kept = sorted(set(factual_nodes) | set(counterfactual_nodes))
    pruned = sorted(set(scm.nodes) - set(kept))
    return kept, pruned, factual_nodes, counterfactual_nodes


def _render_qv_text(
    effect_type: str,
    qv_variant: int,
    func_name: str,
    treatment: int,
    outcome: int,
    queried_return_value: int,
) -> str:
    del treatment, outcome
    if effect_type == "pn":
        variants = {
            1: (
                f"Considering only the calls where `{func_name}` does not return None, "
                f"what is the probability that it returns outcome {queried_return_value}? "
                f"These kept calls are exactly the runs accepted by the factual PN "
                f"filter in the code; within those runs, the code changes the "
                f"treatment setting using the same latent draw."
            ),
            2: (
                f"Run `{func_name}` many times and discard every call that returns None. "
                f"Among the kept calls, what is the rate at which the returned outcome "
                f"equals {queried_return_value}? The kept calls are exactly those "
                f"that satisfy the factual PN acceptance check before the treatment "
                f"setting is changed."
            ),
            3: (
                f"Focus on the non-None outputs of `{func_name}`. Define p as the "
                f"probability that `{func_name}` returns outcome {queried_return_value}, "
                f"excluding calls rejected by the factual PN filter. What is p?"
            ),
        }
    elif effect_type == "ps":
        variants = {
            1: (
                f"Considering only the calls where `{func_name}` does not return None, "
                f"what is the probability that it returns outcome {queried_return_value}? "
                f"These kept calls are exactly the runs accepted by the factual PS "
                f"filter in the code; within those runs, the code changes the "
                f"treatment setting using the same latent draw."
            ),
            2: (
                f"Run `{func_name}` many times and discard every call that returns None. "
                f"Among the kept calls, what is the rate at which the returned outcome "
                f"equals {queried_return_value}? The kept calls are exactly those "
                f"that satisfy the factual PS acceptance check before the treatment "
                f"setting is changed."
            ),
            3: (
                f"Focus on the non-None outputs of `{func_name}`. Define p as the "
                f"probability that `{func_name}` returns outcome {queried_return_value}, "
                f"excluding calls rejected by the factual PS filter. What is p?"
            ),
        }
    else:
        raise ValueError(f"Unknown PN/PS effect_type: {effect_type}")
    return variants[qv_variant]


def _render_code(
    scm: SCMBase,
    intervention: Dict[int, int],
    outcome: int,
    factual_evidence: Dict[int, int],
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
        for config in sorted(scm.cpts[node]["table"])
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
        intervention,
        cf_variant=cf_variant,
        vn_variant=vn_variant,
        selector_name_map=selector_name_map,
    ))
    lines.append("")

    lines.append(f"def {func_name}(rng: random.Random) -> Optional[int]:")
    lines.append(f"    {latent_name} = {draw_name}(rng)")
    lines.append(f"    baseline_state = {factual_build_name}({latent_name}, {{}})")
    cond = " and ".join(
        f'baseline_state["x{node}"] == {value}'
        for node, value in sorted(factual_evidence.items())
    )
    lines.append(f"    accepted = {cond}")
    lines.append("    if not accepted:")
    lines.append("        return None")
    override_literal = "{" + ", ".join(
        f'"x{node}": {value}' for node, value in sorted(intervention.items())
    ) + "}"
    lines.append(f"    shifted_state = {counterfactual_build_name}({latent_name}, {override_literal})")
    lines.append(f'    return shifted_state["x{outcome}"]')
    return "\n".join(lines), func_name


def assemble_all_questions_code(
    scm: SCMBase,
    pn_ps_result: PNPSResult,
    *,
    effect_type: str,
    decimal_places: int = 2,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledCodeQuestion]:
    del decimal_places  # Code-domain answers use fixed four-decimal formatting.
    cf_variant, si_variant, vn_variant, qv_variant = sample_axes(example_seed)
    factual_evidence, intervention, queried_return_value, answer_value, verification_ok = _effect_spec(
        pn_ps_result,
        effect_type,
    )
    full_kept, full_pruned, full_factual_nodes, full_counterfactual_nodes = compute_code_full_nodes(scm)
    sap_kept, sap_pruned, sap_factual_nodes, sap_counterfactual_nodes = _compute_nodes_surgery_ancestor_prune(
        scm,
        factual_evidence,
        intervention,
        pn_ps_result.outcome,
    )

    questions = []
    for variant_name, structure, nodes_kept, nodes_pruned, factual_nodes, counterfactual_nodes in [
        ("full", "single", full_kept, full_pruned, full_factual_nodes, full_counterfactual_nodes),
        ("surgery_ancestor_prune", "multi", sap_kept, sap_pruned, sap_factual_nodes, sap_counterfactual_nodes),
    ]:
        code, func_name = _render_code(
            scm,
            intervention,
            pn_ps_result.outcome,
            factual_evidence,
            factual_nodes,
            counterfactual_nodes,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
        )
        metadata = {
            "num_nodes": len(scm.nodes),
            "num_edges": scm.dag.number_of_edges(),
            "dag_edges": sorted(list(scm.dag.edges())),
            "treatment": pn_ps_result.treatment,
            "outcome": pn_ps_result.outcome,
            "effect_type": effect_type,
            "query_type": effect_type,
            "domain": "code",
            "variant_name": variant_name,
            "code_structure": structure,
            "cf_variant": cf_variant,
            "si_variant": si_variant,
            "vn_variant": vn_variant,
            "qv_variant": qv_variant,
            "is_abnormal": pn_ps_result.is_abnormal,
            "verification_ok": verification_ok,
            "pn_value": pn_ps_result.pn_value,
            "ps_value": pn_ps_result.ps_value,
            "p_y0_eq0_given_x1_y1": pn_ps_result.p_y0_eq0_given_x1_y1,
            "p_y1_eq1_given_x0_y0": pn_ps_result.p_y1_eq1_given_x0_y0,
            "cf_result_pn": pn_ps_result.cf_result_pn.to_json_dict(),
            "cf_result_ps": pn_ps_result.cf_result_ps.to_json_dict(),
            "factual_evidence": {str(k): v for k, v in sorted(factual_evidence.items())},
            "intervention": {str(k): v for k, v in sorted(intervention.items())},
            "queried_return_value": queried_return_value,
            "nodes_kept": nodes_kept,
            "nodes_pruned": nodes_pruned,
            "treatment_outcome_depth": treatment_outcome_depth,
            "effective_prob_conds": effective_prob_conds,
            "has_naive_conditioning": has_naive_conditioning,
        }
        questions.append(AssembledCodeQuestion(
            effect_type=effect_type,
            treatment=pn_ps_result.treatment,
            outcome=pn_ps_result.outcome,
            variant_name=variant_name,
            answer=f"{answer_value:.4f}",
            code_structure=structure,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            qv_variant=qv_variant,
            full_question_text=compose_question_text(
                code,
                _render_qv_text(
                    effect_type,
                    qv_variant,
                    func_name,
                    pn_ps_result.treatment,
                    pn_ps_result.outcome,
                    queried_return_value,
                ),
            ),
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            metadata=metadata,
        ))
    return questions


if __name__ == "__main__":
    import random

    from data_generation.pn_ps.pn_ps_query_sampling import compute_pn_ps
    from data_generation.scm_mechanism_sampling import RLCausalSCM

    print("=== Testing PN/PS Code Question Assembly ===\n")

    dag = nx.DiGraph([(0, 1)])
    cpts = {
        0: {"parents": [], "table": {(): 0.5}},
        1: {"parents": [0], "table": {(0,): 0.0, (1,): 1.0}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=4)
    result = compute_pn_ps(scm, treatment=0, outcome=1)

    print("Test 1: PN code uses PN answer and asks for returned outcome 0")
    pn_questions = assemble_all_questions_code(
        scm,
        result,
        effect_type="pn",
        decimal_places=4,
        example_seed=17,
        treatment_outcome_depth=1.0,
        effective_prob_conds=1,
        has_naive_conditioning=False,
    )
    assert {q.variant_name for q in pn_questions} == set(CODE_VARIANT_NAMES)
    assert all(q.answer == "1.0000" for q in pn_questions)
    assert all("outcome 0" in q.full_question_text for q in pn_questions)
    assert all(q.metadata["domain"] == "code" for q in pn_questions)
    print("   -> PASSED\n")

    print("Test 2: PS code uses PS answer and asks for returned outcome 1")
    ps_questions = assemble_all_questions_code(
        scm,
        result,
        effect_type="ps",
        decimal_places=4,
        example_seed=18,
        treatment_outcome_depth=1.0,
        effective_prob_conds=1,
        has_naive_conditioning=False,
    )
    assert {q.variant_name for q in ps_questions} == set(CODE_VARIANT_NAMES)
    assert all(q.answer == "1.0000" for q in ps_questions)
    assert all("outcome 1" in q.full_question_text for q in ps_questions)
    print("   -> PASSED\n")

    print("Test 3: Generated function returns None on rejected factual runs and 0/1 otherwise")
    namespace = {}
    code = pn_questions[0].full_question_text.split("Question:", 1)[0].strip()
    exec(code, namespace)
    func = namespace["x1"] if "x1" in namespace else namespace["func_main"]
    outputs = [func(random.Random(seed)) for seed in range(20)]
    assert any(value is None for value in outputs)
    assert all(value in (None, 0, 1) for value in outputs)
    assert "-> Optional[int]" in code
    assert "return None" in code
    assert "return True" not in code
    assert "return False" not in code
    print("   -> PASSED\n")

    print("Test 4: Metadata includes effect and code axes")
    for question in pn_questions + ps_questions:
        metadata = question.metadata
        assert metadata["effect_type"] in {"pn", "ps"}
        assert metadata["query_type"] == metadata["effect_type"]
        assert metadata["variant_name"] in CODE_VARIANT_NAMES
        assert metadata["code_structure"] in {"single", "multi"}
        assert metadata["cf_variant"] in {"A", "B", "C"}
        assert metadata["si_variant"] in {"A", "B", "C"}
        assert metadata["vn_variant"] in {"A", "B"}
        assert metadata["qv_variant"] in {1, 2, 3}
        assert "nodes_kept" in metadata
        assert "nodes_pruned" in metadata
        assert "has_naive_conditioning" in metadata
    print("   -> PASSED\n")

    print("=== All PN/PS Code Question Assembly Tests Passed ===")
