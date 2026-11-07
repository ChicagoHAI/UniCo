"""conditional_prob_question_assembly.py"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from data_generation.ate.ate_question_assembly import ANSWER_FORMAT_TEMPLATE, _node_label
from data_generation.dataset_utils.association_assembly_shared import (
    ConditionVariant,
    numerical_condition_variants,
    render_assignment,
    render_graph_text,
)
from data_generation.conditional_prob.conditional_prob_query_sampling import (
    ConditionalProbResult,
    compute_conditional_prob_query,
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


def generate_condition_variants(scm: SCMBase, result: ConditionalProbResult, *, decimal_places: int = 2) -> List[ConditionVariant]:
    return numerical_condition_variants(scm, {result.target, *result.evidence.keys()}, decimal_places=decimal_places)


def _render_query_text(result: ConditionalProbResult, *, query_variant: int) -> str:
    target_label = _node_label(result.target)
    evidence_text = render_assignment(result.evidence)
    return {
        1: f"What is the conditional probability that {target_label} = 1 given {evidence_text}?\nCompute P({target_label} = 1 | {evidence_text}).",
        2: f"Conditioning on {evidence_text}, what is the observational probability that {target_label} = 1?",
        3: f"After learning that {evidence_text}, how likely is {target_label} to equal 1?",
    }[query_variant]


def assemble_question(scm: SCMBase, result: ConditionalProbResult, variant: ConditionVariant, *, graph_variant: int, query_variant: int, decimal_places: int) -> AssembledQuestion:
    graph_text = render_graph_text(scm, graph_variant)
    query_text = _render_query_text(result, query_variant=query_variant)
    answer = f"{result.probability:.4f}"
    full_question_text = f"{graph_text}\n\nThe following conditional probability tables are available:\n{variant.conditions_text}\n\n{query_text}\n\n{ANSWER_FORMAT_TEMPLATE}"
    metadata = {
        "query_type": "conditional_prob",
        "target": result.target,
        "evidence": result.evidence,
        "probability": result.probability,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "variant_name": variant.variant_name,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "decimal_places": decimal_places,
    }
    return AssembledQuestion(
        "conditional_prob",
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


def assemble_all_questions(scm: SCMBase, result: ConditionalProbResult, *, decimal_places: int, example_seed: int, example_idx: int | None = None) -> List[AssembledQuestion]:
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
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    scm = RLCausalSCM(dag, seed=42)
    result = compute_conditional_prob_query(scm, target=2, evidence={0: 1})
    questions = assemble_all_questions(scm, result, decimal_places=2, example_seed=123)
    assert len(questions) == 2
    assert all("\\boxed{" in question.full_question_text for question in questions)
    print("Conditional Prob question assembly checks passed.")
