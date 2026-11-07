"""observed_difference_question_assembly.py"""

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
from data_generation.observed_difference.observed_difference_query_sampling import (
    ObservedDifferenceResult,
    compute_observed_difference_query,
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


def generate_condition_variants(scm: SCMBase, result: ObservedDifferenceResult, *, decimal_places: int = 2) -> List[ConditionVariant]:
    return numerical_condition_variants(scm, {result.source, result.target}, decimal_places=decimal_places)


def _render_query_text(result: ObservedDifferenceResult, *, query_variant: int) -> str:
    source_label = _node_label(result.source)
    target_label = _node_label(result.target)
    return {
        1: f"What is the observed difference between {target_label} given {source_label} = 1 and {target_label} given {source_label} = 0? The answer is positive when {target_label} = 1 is more likely when {source_label} = 1 than when {source_label} = 0, and negative when {target_label} = 1 is less likely when {source_label} = 1 than when {source_label} = 0.",
        2: f"How much does the observational probability of {target_label} = 1 change when comparing {source_label} = 1 versus {source_label} = 0? Report P({target_label} = 1 | {source_label} = 1) - P({target_label} = 1 | {source_label} = 0).",
        3: f"What is the observed difference in the probability of {target_label} = 1 across the two values of {source_label}? Use the signed quantity P({target_label} = 1 | {source_label} = 1) - P({target_label} = 1 | {source_label} = 0), not the absolute difference.",
    }[query_variant]


def assemble_question(scm: SCMBase, result: ObservedDifferenceResult, variant: ConditionVariant, *, graph_variant: int, query_variant: int, decimal_places: int) -> AssembledQuestion:
    graph_text = render_graph_text(scm, graph_variant)
    query_text = _render_query_text(result, query_variant=query_variant)
    answer = f"{result.observed_difference:.4f}"
    full_question_text = f"{graph_text}\n\nThe following conditional probability tables are available:\n{variant.conditions_text}\n\n{query_text}\n\n{ANSWER_FORMAT_TEMPLATE}"
    metadata = {
        "query_type": "observed_difference",
        "source": result.source,
        "target": result.target,
        "observed_difference": result.observed_difference,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "variant_name": variant.variant_name,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "decimal_places": decimal_places,
    }
    return AssembledQuestion(
        "observed_difference",
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


def assemble_all_questions(scm: SCMBase, result: ObservedDifferenceResult, *, decimal_places: int, example_seed: int, example_idx: int | None = None) -> List[AssembledQuestion]:
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
    result = compute_observed_difference_query(scm, source=0, target=2)
    questions = assemble_all_questions(scm, result, decimal_places=2, example_seed=123)
    assert len(questions) == 2 and "\\boxed{" in questions[0].full_question_text
    print("Observed Difference question assembly checks passed.")
