"""
pn_ps_question_assembly.py

PN/PS question assembly in the local ATE house style.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from data_generation.ate.ate_question_assembly import (
    _render_graph_text,
)
from data_generation.cf.cf_question_assembly import (
    _render_assumption_text,
    condition_variant_full as _cf_condition_variant_full,
    condition_variant_surgery_ancestor_prune as _cf_condition_variant_surgery_ancestor_prune,
)
from data_generation.pn_ps.pn_ps_query_sampling import PNPSResult
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
    outcome: int
    effect_type: str
    variant_name: str
    answer_value: float
    pn_value: float
    ps_value: float
    is_abnormal: bool
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
    del treatment, outcome
    variant = _cf_condition_variant_full(
        scm,
        decimal_places=decimal_places,
    )
    return ConditionVariant(
        variant_name=variant.variant_name,
        conditions_text=variant.conditions_text,
        nodes_kept=variant.nodes_kept,
        nodes_pruned=variant.nodes_pruned,
    )


def condition_variant_surgery_ancestor_prune(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    variant = _cf_condition_variant_surgery_ancestor_prune(
        scm,
        treatments={treatment: 1},
        outcome=outcome,
        evidence={treatment: 1, outcome: 1},
        decimal_places=decimal_places,
    )
    return ConditionVariant(
        variant_name=variant.variant_name,
        conditions_text=variant.conditions_text,
        nodes_kept=variant.nodes_kept,
        nodes_pruned=variant.nodes_pruned,
    )


def generate_condition_variants(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> List[ConditionVariant]:
    return [
        condition_variant_full(scm, treatment, outcome, decimal_places=decimal_places),
        condition_variant_surgery_ancestor_prune(
            scm,
            treatment,
            outcome,
            decimal_places=decimal_places,
        ),
    ]


def _render_query_text(
    treatment: int,
    outcome: int,
    *,
    effect_type: str,
    query_variant: int,
) -> str:
    t_label = f"X{treatment}"
    o_label = f"X{outcome}"

    if effect_type == "pn":
        formal = (
            f"What is the Probability of Necessity (PN) of {t_label} for {o_label}?"
        )
        if query_variant == 1:
            variation = (
                f"That is, among units with factual {t_label} = 1 and {o_label} = 1, "
                f"what is the probability that {o_label} would have been 0 if "
                f"{t_label} had instead been set to 0?"
            )
        elif query_variant == 2:
            variation = (
                f"In other words, conditioned on observing {t_label} = 1 and "
                f"{o_label} = 1, how likely is it that changing {t_label} to 0 "
                f"would switch {o_label} to 0?"
            )
        elif query_variant == 3:
            variation = (
                f"Equivalently, compute P({o_label}_0 = 0 | {t_label} = 1, "
                f"{o_label} = 1)."
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
    elif effect_type == "ps":
        formal = (
            f"What is the Probability of Sufficiency (PS) of {t_label} for {o_label}?"
        )
        if query_variant == 1:
            variation = (
                f"That is, among units with factual {t_label} = 0 and {o_label} = 0, "
                f"what is the probability that {o_label} would become 1 if "
                f"{t_label} were set to 1?"
            )
        elif query_variant == 2:
            variation = (
                f"In other words, conditioned on observing {t_label} = 0 and "
                f"{o_label} = 0, how likely is it that changing {t_label} to 1 "
                f"would make {o_label} equal 1?"
            )
        elif query_variant == 3:
            variation = (
                f"Equivalently, compute P({o_label}_1 = 1 | {t_label} = 0, "
                f"{o_label} = 0)."
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
    else:
        raise ValueError(f"Unknown effect_type: {effect_type}")

    return "\n".join([formal, variation])


ANSWER_FORMAT_TEMPLATE = (
    "Think step-by-step to solve this causal reasoning question. "
    "Your final answer should be rounded to four decimal places "
    "and included inside \\boxed{}, as below: \\boxed{<your final answer>}."
)


def assemble_question(
    scm: SCMBase,
    pn_ps_result: PNPSResult,
    variant: ConditionVariant,
    *,
    effect_type: str,
    graph_variant: int,
    query_variant: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> AssembledQuestion:
    treatment = pn_ps_result.treatment
    outcome = pn_ps_result.outcome
    graph_text = _render_graph_text(scm, graph_variant)
    query_text = _render_query_text(
        treatment,
        outcome,
        effect_type=effect_type,
        query_variant=query_variant,
    )
    answer_value = (
        pn_ps_result.pn_value if effect_type == "pn"
        else pn_ps_result.ps_value
    )
    evidence = (
        {treatment: 1, outcome: 1} if effect_type == "pn"
        else {treatment: 0, outcome: 0}
    )

    assumption_text = _render_assumption_text(graph_variant)

    full_question_text = (
        f"{graph_text}\n\n"
        f"The following probability relationships hold:\n"
        f"{variant.conditions_text}\n\n"
        f"{assumption_text}\n\n"
        f"{query_text}\n\n"
        f"{ANSWER_FORMAT_TEMPLATE}"
    )

    metadata = {
        "num_nodes": len(scm.nodes),
        "num_edges": scm.dag.number_of_edges(),
        "dag_edges": sorted(list(scm.dag.edges())),
        "treatment": treatment,
        "outcome": outcome,
        "effect_type": effect_type,
        "variant_name": variant.variant_name,
        "pn_value": pn_ps_result.pn_value,
        "ps_value": pn_ps_result.ps_value,
        "p_y0_eq0_given_x1_y1": pn_ps_result.p_y0_eq0_given_x1_y1,
        "p_y1_eq1_given_x0_y0": pn_ps_result.p_y1_eq1_given_x0_y0,
        "is_abnormal": pn_ps_result.is_abnormal,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "cf_result_pn": pn_ps_result.cf_result_pn.to_json_dict(),
        "cf_result_ps": pn_ps_result.cf_result_ps.to_json_dict(),
        "conditioning_event": evidence,
        "treatment_outcome_depth": treatment_outcome_depth,
        "effective_prob_conds": effective_prob_conds,
        "has_naive_conditioning": has_naive_conditioning,
    }

    return AssembledQuestion(
        treatment=treatment,
        outcome=outcome,
        effect_type=effect_type,
        variant_name=variant.variant_name,
        answer_value=answer_value,
        pn_value=pn_ps_result.pn_value,
        ps_value=pn_ps_result.ps_value,
        is_abnormal=pn_ps_result.is_abnormal,
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
    pn_ps_result: PNPSResult,
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
        pn_ps_result.treatment,
        pn_ps_result.outcome,
        decimal_places=decimal_places,
    )

    return [
        assemble_question(
            scm,
            pn_ps_result,
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
    from data_generation.pn_ps.pn_ps_query_sampling import compute_pn_ps

    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing PN/PS Question Assembly ===\n")

    chain_dag = nx.DiGraph([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)
    pn_ps_result = compute_pn_ps(chain_scm, treatment=0, outcome=2)

    print("Test 1: PN/PS condition variants")
    variants = generate_condition_variants(chain_scm, treatment=0, outcome=2)
    assert len(variants) == 2
    assert variants[0].variant_name == "full"
    assert variants[1].variant_name == "surgery_ancestor_prune"
    print("   -> PASSED\n")

    print("Test 2: Assemble PN question")
    pn_questions = assemble_all_questions(
        chain_scm,
        pn_ps_result,
        effect_type="pn",
        decimal_places=2,
        example_seed=7,
        treatment_outcome_depth=2.0,
        effective_prob_conds=5,
        has_naive_conditioning=False,
    )
    assert len(pn_questions) == 2
    assert all(q.effect_type == "pn" for q in pn_questions)
    assert "Probability of Necessity" in pn_questions[0].query_text
    assert_close(pn_questions[0].answer_value, pn_ps_result.pn_value)
    print("   -> PASSED\n")

    print("Test 3: Assemble PS question")
    ps_questions = assemble_all_questions(
        chain_scm,
        pn_ps_result,
        effect_type="ps",
        decimal_places=2,
        example_seed=7,
        treatment_outcome_depth=2.0,
        effective_prob_conds=5,
        has_naive_conditioning=False,
    )
    assert len(ps_questions) == 2
    assert all(q.effect_type == "ps" for q in ps_questions)
    assert "Probability of Sufficiency" in ps_questions[0].query_text
    assert_close(ps_questions[0].answer_value, pn_ps_result.ps_value)
    print("   -> PASSED\n")

    print("Test 4: Shared graph/query variants per example")
    assert len({q.graph_variant for q in ps_questions}) == 1
    assert len({q.query_variant for q in ps_questions}) == 1
    print("   -> PASSED\n")

    print("=== All PN/PS Question Assembly Tests Passed ===")
