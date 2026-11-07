"""markov_blanket_question_assembly.py"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from data_generation.backdoor.backdoor_question_assembly import _render_graph_text_from_dag
from data_generation.identification.identification_question_assembly import _answer_format_template
from data_generation.dataset_utils.association_assembly_shared import ConditionVariant, structural_condition_variants
from data_generation.markov_blanket.markov_blanket_query_sampling import MarkovBlanketResult, OPTION_LABELS, build_markov_blanket_query


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


def format_answer_for_export(query_format: str, correct_options: str):
    if query_format == "binary":
        return correct_options
    if correct_options == "":
        return []
    return [label.strip() for label in correct_options.split(",") if label.strip()]


def _statement_text(option, *, statement_variant: int) -> str:
    candidate = f"X{option.candidate}"
    target = f"X{option.target}"
    return {
        1: f"{candidate} is in the Markov blanket of {target}",
        2: f"{candidate} belongs to the Markov blanket of {target}",
        3: f"{candidate} is a Markov blanket member of {target}",
    }[statement_variant]


def generate_condition_variants(dag: nx.DiGraph, result: MarkovBlanketResult) -> List[ConditionVariant]:
    dummy = type("Dummy", (), {"nodes": list(dag.nodes()), "dag": dag})()
    focus_nodes: Set[int] = {option.target for option in result.option_results} | {option.candidate for option in result.option_results}
    return [
        variant
        for variant in structural_condition_variants(
            dummy,
            focus_nodes,
            prune_text="No probability tables are needed for this structural query. Focus on the queried target/candidate nodes and their ancestors.",
        )
        if variant.variant_name == "full"
    ]


def _render_query_text(result: MarkovBlanketResult, *, query_variant: int, statement_variant: int) -> str:
    if result.query_format == "binary":
        option = result.option_results[0]
        candidate = f"X{option.candidate}"
        target = f"X{option.target}"
        return {
            1: f"Question: Is {candidate} in the Markov blanket of {target}?",
            2: f"Question: Using the DAG only, should {candidate} be treated as a Markov blanket variable for {target}?",
            3: f"Question: Does {candidate} belong to {target}'s Markov blanket in this graph?",
        }[query_variant]
    if result.query_format == "multi_choice":
        prompt = {
            1: "Question: Which answer choice correctly describes whether candidate statement 1 and candidate statement 2 are true?",
            2: "Question: Please select the one answer choice that correctly states whether candidate statement 1 and candidate statement 2 hold.",
            3: "Question: Choose the one answer choice that correctly summarizes which candidate Markov blanket membership statements are true.",
        }[query_variant]
        lines = [prompt]
        for idx, option in enumerate(result.option_results, start=1):
            lines.append(f"Candidate statement {idx}: {_statement_text(option, statement_variant=statement_variant)}")
        lines.extend([
            "Option (A): Neither candidate statement 1 nor candidate statement 2 is true.",
            "Option (B): Candidate statement 1 is true, but candidate statement 2 is not.",
            "Option (C): Candidate statement 2 is true, but candidate statement 1 is not.",
            "Option (D): Both candidate statement 1 and candidate statement 2 are true.",
        ])
        return "\n".join(lines)
    prompt = {
        1: "Question: Which of the following Markov blanket membership statements are true?",
        2: "Question: Choose all options where the candidate node is in the target node's Markov blanket.",
        3: "Question: Which candidate membership statements hold from the DAG structure?",
    }[query_variant]
    lines = [prompt]
    for label, option in zip(OPTION_LABELS, result.option_results):
        lines.append(f"Option ({label}): {_statement_text(option, statement_variant=statement_variant)}")
    return "\n".join(lines)


def assemble_question(dag: nx.DiGraph, result: MarkovBlanketResult, variant: ConditionVariant, *, graph_variant: int, query_variant: int, statement_variant: int) -> AssembledQuestion:
    graph_text = _render_graph_text_from_dag(dag, graph_variant)
    query_text = _render_query_text(result, query_variant=query_variant, statement_variant=statement_variant)
    full_question_text = f"{graph_text}\n\n{variant.conditions_text}\n\n{query_text}\n\n{_answer_format_template(result.query_format)}"
    metadata = {
        "query_type": "markov_blanket",
        "query_format": result.query_format,
        "correct_options": result.correct_options,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "statement_variant": statement_variant,
        "variant_name": variant.variant_name,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "is_abnormal": result.is_abnormal,
    }
    return AssembledQuestion("markov_blanket", variant.variant_name, result.correct_options, graph_variant, query_variant, graph_text, variant.conditions_text, query_text, full_question_text, variant.nodes_kept, variant.nodes_pruned, metadata)


def assemble_all_questions(dag: nx.DiGraph, result: MarkovBlanketResult, *, example_seed: int, example_idx: int | None = None) -> List[AssembledQuestion]:
    if example_idx is None:
        rng = random.Random(example_seed)
        graph_variant = rng.randint(1, 3)
        query_variant = rng.randint(1, 3)
        statement_variant = rng.randint(1, 3)
    else:
        graph_variant = (example_idx % 3) + 1
        query_variant = ((example_idx // 3) % 3) + 1
        statement_variant = ((example_idx // 9) % 3) + 1
    return [
        assemble_question(
            dag,
            result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
            statement_variant=statement_variant,
        )
        for variant in generate_condition_variants(dag, result)
    ]


if __name__ == "__main__":
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    result = build_markov_blanket_query(dag, query_format="multi_answer", option_order_rng=random.Random(0))
    questions = assemble_all_questions(dag, result, example_seed=123)
    assert len(questions) == 1
    assert "Option (A):" in questions[0].full_question_text
    assert questions[0].metadata["statement_variant"] in {1, 2, 3}
    print("Markov Blanket question assembly checks passed.")
