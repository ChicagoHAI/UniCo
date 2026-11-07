"""marginal_prob_question_assembly.py"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from data_generation.ate.ate_question_assembly import ANSWER_FORMAT_TEMPLATE, _node_label
from data_generation.dataset_utils.association_assembly_shared import (
    ConditionVariant,
    numerical_condition_variants,
    render_graph_text,
)
from data_generation.marginal_prob.marginal_prob_query_sampling import (
    MarginalProbResult,
    compute_marginal_prob_query,
)
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class AssembledQuestion:
    query_type: str
    variant_name: str
    answer: str
    graph_variant: int
    query_variant: int
    graph_text: str
    conditions_text: str
    query_text: str
    full_question_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


def generate_condition_variants(
    scm: SCMBase,
    result: MarginalProbResult,
    *,
    decimal_places: int = 2,
) -> List[ConditionVariant]:
    return numerical_condition_variants(scm, {result.target}, decimal_places=decimal_places)


def _render_query_text(result: MarginalProbResult, *, query_variant: int) -> str:
    target_label = _node_label(result.target)
    return {
        1: f"What is the marginal probability that {target_label} = 1?\nCompute P({target_label} = 1).",
        2: f"Ignoring any interventions, what is the observational probability of {target_label} = 1?",
        3: f"Under the given SCM, how likely is {target_label} to take value 1?",
    }[query_variant]


def assemble_question(
    scm: SCMBase,
    result: MarginalProbResult,
    variant: ConditionVariant,
    *,
    graph_variant: int,
    query_variant: int,
    decimal_places: int,
) -> AssembledQuestion:
    graph_text = render_graph_text(scm, graph_variant)
    query_text = _render_query_text(result, query_variant=query_variant)
    answer = f"{result.probability:.4f}"
    full_question_text = f"{graph_text}\n\nThe following conditional probability tables are available:\n{variant.conditions_text}\n\n{query_text}\n\n{ANSWER_FORMAT_TEMPLATE}"
    metadata = {
        "query_type": "marginal_prob",
        "target": result.target,
        "probability": result.probability,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "variant_name": variant.variant_name,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "decimal_places": decimal_places,
    }
    return AssembledQuestion(
        "marginal_prob",
        variant.variant_name,
        answer,
        graph_variant,
        query_variant,
        graph_text,
        variant.conditions_text,
        query_text,
        full_question_text,
        variant.nodes_kept,
        variant.nodes_pruned,
        metadata,
    )


def assemble_all_questions(
    scm: SCMBase,
    result: MarginalProbResult,
    *,
    decimal_places: int,
    example_seed: int,
    example_idx: int | None = None,
) -> List[AssembledQuestion]:
    if example_idx is None:
        rng = random.Random(example_seed)
        graph_variant = rng.randint(1, 3)
        query_variant = rng.randint(1, 3)
    else:
        graph_variant = (example_idx % 3) + 1
        query_variant = ((example_idx // 3) % 3) + 1
    return [
        assemble_question(
            scm,
            result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
            decimal_places=decimal_places,
        )
        for variant in generate_condition_variants(scm, result, decimal_places=decimal_places)
    ]


if __name__ == "__main__":
    print("Test 1: assemble_all_questions returns the two expected condition variants")
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    cpts = {
        0: {"parents": [], "table": {(): 0.3}},
        1: {"parents": [], "table": {(): 0.6}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.7, (1, 1): 0.9}},
        3: {"parents": [2], "table": {(0,): 0.2, (1,): 0.8}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=1)
    result = compute_marginal_prob_query(scm, target=2)
    questions = assemble_all_questions(scm, result, decimal_places=2, example_seed=123)
    assert [question.variant_name for question in questions] == ["full", "query_ancestor_prune"]

    print("Test 2: assembled answer matches the exact marginal probability")
    assert all(question.answer == "0.3580" for question in questions)
    assert all(question.metadata["probability"] == result.probability for question in questions)

    print("Test 3: rendered prompt contains the target node, CPT context, and boxed-answer instruction")
    assert all("X2 = 1" in question.query_text for question in questions)
    assert all("conditional probability tables" in question.full_question_text for question in questions)
    assert all("\\boxed{" in question.full_question_text for question in questions)
    print("Marginal Prob question assembly checks passed.")
