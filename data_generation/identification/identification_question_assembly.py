"""
identification_question_assembly.py

Question assembly for the identification query type.
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
from data_generation.identification.identification_query_sampling import (
    IdentificationResult,
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


def _render_distribution(observed_variables: List[int]) -> str:
    rendered = ", ".join(_node_label(node) for node in observed_variables)
    return f"P({rendered})"


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
    elif graph_variant == 2:
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
    elif graph_variant == 3:
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


def _render_query_text(
    identification_result: IdentificationResult,
    *,
    query_variant: int,
) -> str:
    treatment_label = _node_label(identification_result.treatment)
    outcome_label = _node_label(identification_result.outcome)

    if identification_result.query_format == "binary":
        observed = _render_distribution(
            identification_result.option_results[0].observed_variables
        )
        if query_variant == 1:
            question = (
                f"Can we deduce the following: P({outcome_label} | do({treatment_label})) is identifiable from the available observed distribution {observed}?"
            )
        elif query_variant == 2:
            question = (
                f"Can we identify the causal effect P({outcome_label} | do({treatment_label})) from the available observed distribution {observed}?"
            )
        elif query_variant == 3:
            question = (
                f"Does the available observed distribution {observed} contain enough information to identify P({outcome_label} | do({treatment_label}))?"
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
        return question

    lines = []
    if identification_result.query_format == "multi_choice":
        if query_variant == 1:
            lines.append(
                f"Question: Which answer choice correctly describes whether observed set 1 and observed set 2 identify P({outcome_label} | do({treatment_label}))?"
            )
        elif query_variant == 2:
            lines.append(
                f"Question: Please select the one answer choice that correctly states the identifiability of P({outcome_label} | do({treatment_label})) under observed set 1 and observed set 2."
            )
        elif query_variant == 3:
            lines.append(
                f"Question: Choose the one answer choice that correctly summarizes whether observed set 1 and observed set 2 contain enough information to identify P({outcome_label} | do({treatment_label}))."
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")
        for idx, option in enumerate(identification_result.option_results, start=1):
            lines.append(
                f"Observed set {idx}: {_render_distribution(option.observed_variables)}"
            )
        lines.append("Option (A): The target causal effect cannot be identified based on either observed set 1 or observed set 2.")
        lines.append("Option (B): The target causal effect can be identified based on observed set 1, but not based on observed set 2.")
        lines.append("Option (C): The target causal effect can be identified based on observed set 2, but not based on observed set 1.")
        lines.append("Option (D): The target causal effect can be identified based on both observed set 1 and observed set 2.")
        return "\n".join(lines)

    if query_variant == 1:
        lines.append(
            f"Question: Choose all options from which P({outcome_label} | do({treatment_label})) is identifiable."
        )
    elif query_variant == 2:
        lines.append(
            f"Question: Which observed distribution options contain enough information to identify P({outcome_label} | do({treatment_label}))?"
        )
    elif query_variant == 3:
        lines.append(
            f"Question: Under which options can the causal effect P({outcome_label} | do({treatment_label})) be computed from observational data?"
        )
    else:
        raise ValueError(f"Unknown query_variant: {query_variant}")

    for label, option in zip(OPTION_LABELS, identification_result.option_results):
        lines.append(f"Option ({label}): {_render_distribution(option.observed_variables)}")
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
    identification_result: IdentificationResult,
) -> List[ConditionVariant]:
    all_nodes = set(dag.nodes())
    observed_nodes = {
        node
        for option in identification_result.option_results
        for node in option.observed_variables
    }
    return [
        ConditionVariant(
            variant_name=VARIANT_NAME,
            conditions_text="",
            nodes_kept=sorted(observed_nodes),
            nodes_pruned=sorted(all_nodes - observed_nodes),
        )
    ]


def assemble_question(
    dag: nx.DiGraph,
    identification_result: IdentificationResult,
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
    query_text = _render_query_text(
        identification_result,
        query_variant=query_variant,
    )
    full_question_text = (
        f"{graph_text}\n\n"
        f"{query_text}\n\n"
        f"{_answer_format_template(identification_result.query_format)}"
    )

    metadata = {
        "query_type": "identification",
        "query_format": identification_result.query_format,
        "num_nodes": dag.number_of_nodes(),
        "num_edges": dag.number_of_edges(),
        "dag_edges": sorted(list(dag.edges())),
        "treatment": identification_result.treatment,
        "outcome": identification_result.outcome,
        "option_observed_sets": [
            option.observed_variables for option in identification_result.option_results
        ],
        "option_identifiable_flags": [
            option.identifiable for option in identification_result.option_results
        ],
        "option_choice_semantics": (
            {
                "A": "the target causal effect cannot be identified based on either observed set 1 or observed set 2",
                "B": "the target causal effect can be identified based on observed set 1, but not based on observed set 2",
                "C": "the target causal effect can be identified based on observed set 2, but not based on observed set 1",
                "D": "the target causal effect can be identified based on both observed set 1 and observed set 2",
            }
            if identification_result.query_format == "multi_choice"
            else None
        ),
        "option_methods": [option.method for option in identification_result.option_results],
        "option_witnesses": [option.witness for option in identification_result.option_results],
        "correct_options": identification_result.correct_options,
        "num_correct_options": sum(
            option.identifiable for option in identification_result.option_results
        ),
        "variant_name": variant.variant_name,
        "is_abnormal": identification_result.is_abnormal,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "treatment_outcome_depth": treatment_outcome_depth,
        "effective_prob_conds": effective_prob_conds,
        "has_naive_conditioning": has_naive_conditioning,
    }

    return AssembledQuestion(
        treatment=identification_result.treatment,
        outcome=identification_result.outcome,
        variant_name=variant.variant_name,
        correct_options=identification_result.correct_options,
        is_abnormal=identification_result.is_abnormal,
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
    identification_result: IdentificationResult,
    *,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledQuestion]:
    rng = random.Random(example_seed)
    graph_variant = rng.randint(1, 3)
    query_variant = rng.randint(1, 3)
    variants = generate_condition_variants(dag, identification_result)
    return [
        assemble_question(
            dag,
            identification_result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
            treatment_outcome_depth=treatment_outcome_depth,
            effective_prob_conds=effective_prob_conds,
            has_naive_conditioning=has_naive_conditioning,
        )
        for variant in variants
    ]
