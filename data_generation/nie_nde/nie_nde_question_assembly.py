"""
nie_nde_question_assembly.py

Question assembly for Natural Indirect Effect (NIE) and Natural Direct
Effect (NDE), following the local ATE prompt/layout style.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from data_generation.ate.ate_question_assembly import (
    _render_graph_text,
    _topo_sorted_subset,
    condition_variant_full as _ate_condition_variant_full,
    render_node_cpt,
)
from data_generation.nie_nde.nie_nde_query_sampling import NIENDEResult, compute_nie_nde
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class ConditionVariant:
    variant_name: str
    conditions_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]


@dataclass
class AssembledQuestion:
    treatment: int
    mediator: int
    outcome: int
    effect_type: str
    answer_value: float
    nie_value: float
    nde_value: float
    variant_name: str
    graph_variant: int
    query_variant: int
    graph_text: str
    conditions_text: str
    query_text: str
    full_question_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


def condition_variant_full(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    return _ate_condition_variant_full(
        scm,
        treatment,
        outcome,
        decimal_places=decimal_places,
    )


def condition_variant_surgery_only(
    scm: SCMBase,
    treatment: int,
    mediator: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    all_nodes = set(scm.nodes)
    cpd_nodes = all_nodes - {treatment}
    topo_nodes = _topo_sorted_subset(scm.dag, cpd_nodes)
    cpt_blocks = [
        render_node_cpt(scm, node, decimal_places=decimal_places)
        for node in topo_nodes
    ]
    return ConditionVariant(
        variant_name="surgery_only",
        conditions_text="\n".join(cpt_blocks),
        nodes_kept=sorted(cpd_nodes),
        nodes_pruned=sorted(all_nodes - cpd_nodes),
    )


def condition_variant_surgery_ancestor_prune(
    scm: SCMBase,
    treatment: int,
    mediator: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    # After intervening on treatment, keep only the mediator/outcome side
    # of the mutilated graph. This matches the local surgery-ancestor-prune
    # style while preserving the mechanisms needed to evaluate NIE/NDE.
    mutilated = scm.pgmpy_model.do([treatment])
    anc_graph = mutilated.get_ancestral_graph(nodes=[mediator, outcome])
    cpd_nodes = set(anc_graph.nodes()) - {treatment}
    topo_nodes = _topo_sorted_subset(scm.dag, cpd_nodes)
    cpt_blocks = [
        render_node_cpt(scm, node, decimal_places=decimal_places)
        for node in topo_nodes
    ]
    return ConditionVariant(
        variant_name="surgery_ancestor_prune",
        conditions_text="\n".join(cpt_blocks),
        nodes_kept=sorted(cpd_nodes),
        nodes_pruned=sorted(set(scm.nodes) - cpd_nodes),
    )


def generate_condition_variants(
    scm: SCMBase,
    treatment: int,
    mediator: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> List[ConditionVariant]:
    return [
        condition_variant_full(
            scm,
            treatment,
            outcome,
            decimal_places=decimal_places,
        ),
        condition_variant_surgery_only(
            scm,
            treatment,
            mediator,
            outcome,
            decimal_places=decimal_places,
        ),
        condition_variant_surgery_ancestor_prune(
            scm,
            treatment,
            mediator,
            outcome,
            decimal_places=decimal_places,
        ),
    ]


def _render_query_text(
    treatment: int,
    mediator: int,
    outcome: int,
    *,
    effect_type: str,
    query_variant: int,
) -> str:
    treatment_label = f"X{treatment}"
    mediator_label = f"X{mediator}"
    outcome_label = f"X{outcome}"

    if effect_type == "nie":
        formal = (
            "What is the Natural Indirect Effect (NIE) of "
            f"{treatment_label} on {outcome_label} through {mediator_label}?"
        )
        if query_variant == 1:
            variation = (
                f"That is, how much does the expected value of {outcome_label} "
                f"change when the mediator moves from the value it would take under "
                f"{treatment_label} = 0 to the value it would take under "
                f"{treatment_label} = 1, while treatment itself is fixed at "
                f"{treatment_label} = 1?"
            )
        elif query_variant == 2:
            variation = (
                f"In other words, how much of the effect of {treatment_label} on "
                f"{outcome_label} is transmitted through changes in "
                f"{mediator_label}?"
            )
        elif query_variant == 3:
            variation = (
                f"Equivalently, what is the expected difference in {outcome_label} when "
                f"{treatment_label} is set to 1 and the mediator takes the value it would "
                f"have under treatment, versus when {treatment_label} is still 1 but the "
                f"mediator instead takes the value it would have under no treatment?"
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
    elif effect_type == "nde":
        formal = (
            "What is the Natural Direct Effect (NDE) of "
            f"{treatment_label} on {outcome_label} relative to {mediator_label}?"
        )
        if query_variant == 1:
            variation = (
                f"That is, how much does the expected value of {outcome_label} "
                f"change when {treatment_label} changes from 0 to 1 while "
                f"{mediator_label} is kept at the value it would naturally take "
                f"under {treatment_label} = 0?"
            )
        elif query_variant == 2:
            variation = (
                f"In other words, how much of the effect of {treatment_label} on "
                f"{outcome_label} operates through pathways other than "
                f"{mediator_label}?"
            )
        elif query_variant == 3:
            variation = (
                f"Equivalently, what is the expected difference in {outcome_label} when "
                f"{treatment_label} changes from 0 to 1 while the mediator is fixed at the "
                f"value it would have under no treatment?"
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    return "\n".join([formal, variation])


ANSWER_FORMAT_TEMPLATE = (
    'Think step-by-step to solve this causal reasoning question. '
    'Your final answer should be rounded to four decimal places '
    'and included inside \\boxed{}, as below: \\boxed{<your final answer>}.'
)


def assemble_question(
    scm: SCMBase,
    nie_nde_result: NIENDEResult,
    variant: ConditionVariant,
    *,
    effect_type: str,
    graph_variant: int,
    query_variant: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> AssembledQuestion:
    graph_text = _render_graph_text(scm, graph_variant)
    query_text = _render_query_text(
        nie_nde_result.treatment,
        nie_nde_result.mediator,
        nie_nde_result.outcome,
        effect_type=effect_type,
        query_variant=query_variant,
    )
    answer_value = (
        nie_nde_result.nie_value if effect_type == "nie"
        else nie_nde_result.nde_value
    )

    full_question_text = (
        f"{graph_text}\n\n"
        f"The following probability relationships hold:\n"
        f"{variant.conditions_text}\n\n"
        f"{query_text}\n\n"
        f"{ANSWER_FORMAT_TEMPLATE}"
    )

    metadata = {
        "num_nodes": len(scm.nodes),
        "num_edges": scm.dag.number_of_edges(),
        "dag_edges": sorted(list(scm.dag.edges())),
        "treatment": nie_nde_result.treatment,
        "mediator": nie_nde_result.mediator,
        "outcome": nie_nde_result.outcome,
        "variant_name": variant.variant_name,
        "effect_type": effect_type,
        "nie_value": nie_nde_result.nie_value,
        "nde_value": nie_nde_result.nde_value,
        "pm_do0": nie_nde_result.pm_do0,
        "pm_do1": nie_nde_result.pm_do1,
        "py_values": nie_nde_result.py_values,
        "is_abnormal": nie_nde_result.is_abnormal,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "treatment_outcome_depth": treatment_outcome_depth,
        "effective_prob_conds": effective_prob_conds,
        "has_naive_conditioning": has_naive_conditioning,
    }

    return AssembledQuestion(
        treatment=nie_nde_result.treatment,
        mediator=nie_nde_result.mediator,
        outcome=nie_nde_result.outcome,
        effect_type=effect_type,
        answer_value=answer_value,
        nie_value=nie_nde_result.nie_value,
        nde_value=nie_nde_result.nde_value,
        variant_name=variant.variant_name,
        graph_variant=graph_variant,
        query_variant=query_variant,
        graph_text=graph_text,
        conditions_text=variant.conditions_text,
        query_text=query_text,
        full_question_text=full_question_text,
        nodes_kept=variant.nodes_kept,
        nodes_pruned=variant.nodes_pruned,
        metadata=metadata,
    )


def assemble_all_questions(
    scm: SCMBase,
    nie_nde_result: NIENDEResult,
    *,
    effect_type: str,
    decimal_places: int = 2,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledQuestion]:
    rng = random.Random(example_seed)
    graph_variant = rng.randint(1, 3)
    query_variant = rng.randint(1, 3)

    variants = generate_condition_variants(
        scm,
        nie_nde_result.treatment,
        nie_nde_result.mediator,
        nie_nde_result.outcome,
        decimal_places=decimal_places,
    )

    return [
        assemble_question(
            scm,
            nie_nde_result,
            variant,
            effect_type=effect_type,
            graph_variant=graph_variant,
            query_variant=query_variant,
            treatment_outcome_depth=treatment_outcome_depth,
            effective_prob_conds=effective_prob_conds,
            has_naive_conditioning=has_naive_conditioning,
        )
        for variant in variants
    ]


if __name__ == "__main__":
    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing NIE/NDE Question Assembly ===\n")

    dag = nx.DiGraph([(0, 1), (1, 2), (0, 2), (3, 0)])
    cpts = {
        0: {"parents": [3], "table": {(0,): 0.2, (1,): 0.8}},
        1: {"parents": [0], "table": {(0,): 0.1, (1,): 0.9}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.3, (1, 0): 0.6, (1, 1): 0.9}},
        3: {"parents": [], "table": {(): 0.5}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=4)
    result = compute_nie_nde(scm, 0, 1, 2)

    print("Test 1: Condition variants")
    variants = generate_condition_variants(scm, 0, 1, 2, decimal_places=2)
    assert [variant.variant_name for variant in variants] == [
        "full",
        "surgery_only",
        "surgery_ancestor_prune",
    ]
    assert variants[0].nodes_kept == [0, 1, 2, 3]
    assert 0 not in variants[1].nodes_kept
    assert variants[2].nodes_kept == [1, 2]
    print("   -> PASSED\n")

    print("Test 2: Assemble NIE question")
    nie_questions = assemble_all_questions(
        scm,
        result,
        effect_type="nie",
        decimal_places=2,
        example_seed=7,
        treatment_outcome_depth=2.0,
        effective_prob_conds=6,
    )
    assert len(nie_questions) == 3
    assert all(q.effect_type == "nie" for q in nie_questions)
    assert_close(nie_questions[0].answer_value, result.nie_value)
    assert all(q.graph_variant == nie_questions[0].graph_variant for q in nie_questions)
    assert all(q.query_variant == nie_questions[0].query_variant for q in nie_questions)
    print("   -> PASSED\n")

    print("Test 3: Assemble NDE question")
    nde_questions = assemble_all_questions(
        scm,
        result,
        effect_type="nde",
        decimal_places=2,
        example_seed=7,
        treatment_outcome_depth=2.0,
        effective_prob_conds=6,
    )
    assert len(nde_questions) == 3
    assert all(q.effect_type == "nde" for q in nde_questions)
    assert_close(nde_questions[0].answer_value, result.nde_value)
    assert "Natural Direct Effect" in nde_questions[0].query_text
    print("   -> PASSED\n")

    print("=== All NIE/NDE question-assembly tests passed ===")
