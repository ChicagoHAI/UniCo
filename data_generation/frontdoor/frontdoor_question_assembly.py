"""
frontdoor_question_assembly.py

Question assembly for the frontdoor query type.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from data_generation.ate.ate_question_assembly import (
    _node_label,
    render_dag_edges,
)
from data_generation.frontdoor.frontdoor_query_sampling import (
    FrontdoorResult,
    OptionFrontdoorResult,
    OPTION_LABELS,
)


VARIANT_NAME = "full"


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
    variant_name: str
    correct_options: str
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


def format_answer_for_export(query_format: str, correct_options: str):
    if query_format == "binary":
        return correct_options
    if correct_options == "":
        return []
    return [label.strip() for label in correct_options.split(",") if label.strip()]


def _render_frontdoor_set(frontdoor_variables: List[int]) -> str:
    if not frontdoor_variables:
        return "{}"
    rendered = ", ".join(_node_label(node) for node in frontdoor_variables)
    return "{" + rendered + "}"


def _render_graph_text_from_dag(dag: nx.DiGraph, graph_variant: int) -> str:
    """Render the full original causal graph section from a raw DAG."""
    node_labels = ", ".join(_node_label(n) for n in dag.nodes)
    edges = sorted(dag.edges())

    if graph_variant == 1:
        edges_str = render_dag_edges(dag)
        return (
            "Imagine a self-contained, hypothetical world with only the "
            "following conditions, and without any unmentioned factors or "
            "causal relationships:\n"
            f"Given the following causal graph among variables {node_labels}:\n"
            f"{edges_str}"
        )
    if graph_variant == 2:
        edge_sentences = "\n".join(
            f"{_node_label(u)} directly causes {_node_label(v)}."
            for u, v in edges
        )
        return (
            f"Given the following causal relationships among variables {node_labels}:\n"
            f"{edge_sentences}\n"
            "These are the only direct causal relationships in this world. "
            "No other causal relationships or unobserved factors exist."
        )
    if graph_variant == 3:
        parent_lines = []
        for n in dag.nodes:
            parents = sorted(dag.predecessors(n))
            if not parents:
                parent_lines.append(f"{_node_label(n)} does not have any parent;")
            else:
                parent_set = ", ".join(_node_label(p) for p in parents)
                parent_lines.append(f"{_node_label(n)} has parents: {{{parent_set}}}")
        parent_block = "\n".join(parent_lines)
        return (
            f"Consider variables {node_labels}, where each variable is a function "
            "only of its direct causes (parents), listed below:\n"
            f"{parent_block}\n"
            "The parent sets listed above are exhaustive--each variable is "
            "affected only by its specified parents, and no unobserved or "
            "latent variables exist in this system."
        )
    raise ValueError(f"Unknown graph_variant: {graph_variant}")


def _render_query_text(frontdoor_result: FrontdoorResult, *, query_variant: int) -> str:
    treatment_label = _node_label(frontdoor_result.treatment)
    outcome_label = _node_label(frontdoor_result.outcome)

    if frontdoor_result.query_format == "binary":
        candidate_set = _render_frontdoor_set(
            frontdoor_result.option_results[0].frontdoor_variables
        )
        if query_variant == 1:
            return (
                f"Question: Is {candidate_set} a valid frontdoor set for estimating P({outcome_label} | do({treatment_label}))?"
            )
        if query_variant == 2:
            return (
                f"Question: Can we use {candidate_set} as a valid frontdoor set to estimate P({outcome_label} | do({treatment_label}))?"
            )
        if query_variant == 3:
            return (
                f"Question: Does {candidate_set} satisfy the frontdoor criterion for estimating P({outcome_label} | do({treatment_label}))?"
            )
        raise ValueError(f"Unknown query_variant: {query_variant}")

    lines = []
    if frontdoor_result.query_format == "multi_choice":
        if query_variant == 1:
            lines.append(
                f"Question: Which answer choice correctly describes whether candidate set 1 and candidate set 2 are valid frontdoor sets for estimating P({outcome_label} | do({treatment_label}))?"
            )
        elif query_variant == 2:
            lines.append(
                f"Question: Please select the one answer choice that correctly states whether candidate set 1 and candidate set 2 satisfy the frontdoor criterion for P({outcome_label} | do({treatment_label}))."
            )
        elif query_variant == 3:
            lines.append(
                f"Question: Choose the one answer choice that correctly summarizes whether candidate set 1 and candidate set 2 can be used as valid frontdoor sets for estimating P({outcome_label} | do({treatment_label}))."
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
        for idx, option in enumerate(frontdoor_result.option_results, start=1):
            lines.append(
                f"Candidate set {idx}: {_render_frontdoor_set(option.frontdoor_variables)}"
            )
        lines.append("Option (A): Neither candidate set 1 nor candidate set 2 is a valid frontdoor set.")
        lines.append("Option (B): Candidate set 1 is a valid frontdoor set, but candidate set 2 is not.")
        lines.append("Option (C): Candidate set 2 is a valid frontdoor set, but candidate set 1 is not.")
        lines.append("Option (D): Both candidate set 1 and candidate set 2 are valid frontdoor sets.")
        return "\n".join(lines)

    if query_variant == 1:
        lines.append(
            f"Question: Given the causal graph, which of the following are valid frontdoor sets for estimating P({outcome_label} | do({treatment_label}))?"
        )
    elif query_variant == 2:
        lines.append(
            f"Question: Which candidate sets satisfy the frontdoor criterion for estimating P({outcome_label} | do({treatment_label}))?"
        )
    elif query_variant == 3:
        lines.append(
            f"Question: Choose all candidate sets that can be used as valid frontdoor sets for estimating P({outcome_label} | do({treatment_label}))."
        )
    else:
        raise ValueError(f"Unknown query_variant: {query_variant}")

    for label, option in zip(OPTION_LABELS, frontdoor_result.option_results):
        lines.append(
            f"Option ({label}): {_render_frontdoor_set(option.frontdoor_variables)}"
        )
    return "\n".join(lines)


def _answer_format_template(query_format: str) -> str:
    if query_format == "binary":
        return (
            "Think step-by-step to solve this causal reasoning question. "
            "Your final answer should only include a single Yes/No formatted inside \\boxed{}, as below: \\boxed{Yes|No}"
        )
    if query_format == "multi_choice":
        return (
            "Think step-by-step to solve this causal reasoning question. "
            "Your final answer should only include the option letter formatted inside \\boxed{}, as below: \\boxed{A|B|C|D}."
        )
    return (
        "Think step-by-step to solve this multiple-choice causal reasoning question. "
        "Choose all the correct options. If no options are correct, output \\boxed{}. "
        "Otherwise, your final answer should only include option letters formatted inside \\boxed{}, delimited by commas, as below: \\boxed{A,B,C,D}."
    )


def generate_condition_variants(
    dag: nx.DiGraph,
    frontdoor_result: FrontdoorResult,
) -> List[ConditionVariant]:
    all_nodes = set(dag.nodes())
    candidate_nodes = {
        node
        for option in frontdoor_result.option_results
        for node in option.frontdoor_variables
    }
    candidate_nodes.update({frontdoor_result.treatment, frontdoor_result.outcome})
    return [
        ConditionVariant(
            variant_name=VARIANT_NAME,
            conditions_text="",
            nodes_kept=sorted(candidate_nodes),
            nodes_pruned=sorted(all_nodes - candidate_nodes),
        )
    ]


def assemble_question(
    dag: nx.DiGraph,
    frontdoor_result: FrontdoorResult,
    variant: ConditionVariant,
    *,
    graph_variant: int,
    query_variant: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> AssembledQuestion:
    graph_text = _render_graph_text_from_dag(dag, graph_variant)
    conditions_text = ""
    query_text = _render_query_text(frontdoor_result, query_variant=query_variant)
    full_question_text = (
        f"{graph_text}\n\n"
        f"{query_text}\n\n"
        f"{_answer_format_template(frontdoor_result.query_format)}"
    )

    metadata = {
        "query_type": "frontdoor",
        "query_format": frontdoor_result.query_format,
        "num_nodes": dag.number_of_nodes(),
        "num_edges": dag.number_of_edges(),
        "dag_edges": sorted(list(dag.edges())),
        "treatment": frontdoor_result.treatment,
        "outcome": frontdoor_result.outcome,
        "option_frontdoor_sets": [
            option.frontdoor_variables for option in frontdoor_result.option_results
        ],
        "option_valid_flags": [
            option.valid_frontdoor for option in frontdoor_result.option_results
        ],
        "option_choice_semantics": (
            {
                "A": "neither candidate set 1 nor candidate set 2 is a valid frontdoor set",
                "B": "candidate set 1 is a valid frontdoor set, but candidate set 2 is not",
                "C": "candidate set 2 is a valid frontdoor set, but candidate set 1 is not",
                "D": "both candidate set 1 and candidate set 2 are valid frontdoor sets",
            }
            if frontdoor_result.query_format == "multi_choice"
            else None
        ),
        "option_methods": [option.method for option in frontdoor_result.option_results],
        "option_witnesses": [option.witness for option in frontdoor_result.option_results],
        "correct_options": frontdoor_result.correct_options,
        "num_correct_options": sum(
            option.valid_frontdoor for option in frontdoor_result.option_results
        ),
        "variant_name": variant.variant_name,
        "is_abnormal": frontdoor_result.is_abnormal,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "treatment_outcome_depth": treatment_outcome_depth,
        "effective_prob_conds": effective_prob_conds,
        "has_naive_conditioning": has_naive_conditioning,
    }

    return AssembledQuestion(
        treatment=frontdoor_result.treatment,
        outcome=frontdoor_result.outcome,
        variant_name=variant.variant_name,
        correct_options=frontdoor_result.correct_options,
        is_abnormal=frontdoor_result.is_abnormal,
        graph_variant=graph_variant,
        query_variant=query_variant,
        graph_text=graph_text,
        conditions_text=conditions_text,
        query_text=query_text,
        full_question_text=full_question_text,
        nodes_kept=variant.nodes_kept,
        nodes_pruned=variant.nodes_pruned,
        metadata=metadata,
    )


def assemble_all_questions(
    dag: nx.DiGraph,
    frontdoor_result: FrontdoorResult,
    *,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledQuestion]:
    rng = random.Random(example_seed)
    graph_variant = rng.randint(1, 3)
    query_variant = rng.randint(1, 3)
    variants = generate_condition_variants(dag, frontdoor_result)
    return [
        assemble_question(
            dag,
            frontdoor_result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
            treatment_outcome_depth=treatment_outcome_depth,
            effective_prob_conds=effective_prob_conds,
            has_naive_conditioning=has_naive_conditioning,
        )
        for variant in variants
    ]

if __name__ == "__main__":
    print("=== Testing Frontdoor Question Assembly ===\n")

    print("Test 1: format_answer_for_export handles binary and multi-answer outputs")
    assert format_answer_for_export("binary", "No") == "No"
    assert format_answer_for_export("multi_answer", "") == []
    assert format_answer_for_export("multi_answer", "B,D") == ["B", "D"]
    print("   -> PASSED\n")

    print("Test 2: assemble_all_questions returns one full variant with frontdoor metadata")
    dag = nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 2), (0, 4), (4, 2)])
    frontdoor_result = FrontdoorResult(
        query_format="multi_choice",
        treatment=0,
        outcome=2,
        option_results=[
            OptionFrontdoorResult([1, 4], True, "frontdoor_validate", ["X1", "X4"]),
            OptionFrontdoorResult([3], False, "frontdoor_validate", []),
        ],
        correct_options="B",
        is_abnormal=False,
    )
    questions = assemble_all_questions(
        dag,
        frontdoor_result,
        example_seed=7,
        treatment_outcome_depth=2.0,
        effective_prob_conds=5,
        has_naive_conditioning=False,
    )
    assert [question.variant_name for question in questions] == ["full"]
    question = questions[0]
    assert question.metadata["query_type"] == "frontdoor"
    assert question.metadata["option_frontdoor_sets"] == [[1, 4], [3]]
    assert "frontdoor" in question.full_question_text.lower()
    assert "Option (A)" in question.full_question_text
    assert "\\boxed{}" in question.full_question_text
    print("   -> PASSED\n")

    print("=== All Frontdoor Question Assembly Tests Passed ===")
