"""independence_test_question_assembly.py"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from data_generation.identification.identification_question_assembly import _answer_format_template
from data_generation.dataset_utils.association_assembly_shared import ConditionVariant, numerical_condition_variants, render_graph_text
from data_generation.independence_test.independence_test_query_sampling import IndependenceTestResult, OPTION_LABELS, build_independence_test_query
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


def format_answer_for_export(query_format: str, correct_options: str):
    if query_format == "binary":
        return correct_options
    if correct_options == "":
        return []
    return [label.strip() for label in correct_options.split(",") if label.strip()]


def _statement_text(option, *, statement_variant: int) -> str:
    left = f"X{option.source}"
    right = f"X{option.target}"
    if option.conditioning_nodes:
        conditioning_nodes = [f"X{node}" for node in option.conditioning_nodes]
        conditioning_braced = ", ".join(conditioning_nodes)
        conditioning_plain = ", ".join(conditioning_nodes)
        return {
            1: f"{left} ⟂ {right} | {{{conditioning_braced}}}",
            2: f"{left} is independent of {right} given {{{conditioning_braced}}}",
            3: f"{left} and {right} are conditionally independent given {conditioning_plain}",
        }[statement_variant]
    return {
        1: f"{left} ⟂ {right}",
        2: f"{left} is independent of {right}",
        3: f"{left} and {right} are independent",
    }[statement_variant]


def generate_condition_variants(scm: SCMBase, result: IndependenceTestResult, *, decimal_places: int = 2) -> List[ConditionVariant]:
    core_nodes: Set[int] = set()
    for option in result.option_results:
        core_nodes.update({option.source, option.target, *option.conditioning_nodes})
    return [
        variant
        for variant in numerical_condition_variants(scm, core_nodes, decimal_places=decimal_places)
        if variant.variant_name == "full"
    ]


def _render_query_text(result: IndependenceTestResult, *, query_variant: int, statement_variant: int) -> str:
    if result.query_format == "binary":
        statement = _statement_text(result.option_results[0], statement_variant=statement_variant)
        return {
            1: f"Question: Does the observational distribution satisfy {statement}?",
            2: f"Question: Is the following conditional independence statement true in the observational distribution: {statement}?",
            3: f"Question: Using the exact SCM distribution, should {statement} be judged true?",
        }[query_variant]
    if result.query_format == "multi_choice":
        prompt = {
            1: "Question: Which answer choice correctly describes whether candidate statement 1 and candidate statement 2 hold in the observational distribution?",
            2: "Question: Please select the one answer choice that correctly states whether candidate statement 1 and candidate statement 2 are true conditional-independence statements.",
            3: "Question: Choose the one answer choice that correctly summarizes whether candidate statement 1 and candidate statement 2 hold under the exact SCM distribution.",
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
        1: "Question: Which of the following conditional-independence statements hold in the observational distribution?",
        2: "Question: Choose all options that are true independence statements under the exact SCM distribution.",
        3: "Question: Which candidate independence statements should be judged true from the observational distribution?",
    }[query_variant]
    lines = [prompt]
    for label, option in zip(OPTION_LABELS, result.option_results):
        lines.append(f"Option ({label}): {_statement_text(option, statement_variant=statement_variant)}")
    return "\n".join(lines)


def assemble_question(scm: SCMBase, result: IndependenceTestResult, variant: ConditionVariant, *, graph_variant: int, query_variant: int, statement_variant: int, decimal_places: int) -> AssembledQuestion:
    graph_text = render_graph_text(scm, graph_variant)
    query_text = _render_query_text(result, query_variant=query_variant, statement_variant=statement_variant)
    full_question_text = f"{graph_text}\n\nThe following conditional probability tables are available:\n{variant.conditions_text}\n\n{query_text}\n\n{_answer_format_template(result.query_format)}"
    metadata = {
        "query_type": "independence_test",
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
    return AssembledQuestion("independence_test", variant.variant_name, result.correct_options, graph_variant, query_variant, graph_text, variant.conditions_text, query_text, full_question_text, variant.nodes_kept, variant.nodes_pruned, metadata)


def assemble_all_questions(scm: SCMBase, result: IndependenceTestResult, *, decimal_places: int, example_seed: int, example_idx: int | None = None) -> List[AssembledQuestion]:
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
            scm,
            result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
            statement_variant=statement_variant,
            decimal_places=decimal_places,
        )
        for variant in generate_condition_variants(scm, result, decimal_places=decimal_places)
    ]


if __name__ == "__main__":
    dag = nx.DiGraph([(0, 1), (1, 2)])
    scm = RLCausalSCM(dag, seed=42)
    result = build_independence_test_query(scm, query_format="multi_answer", option_order_rng=random.Random(0))
    questions = assemble_all_questions(scm, result, decimal_places=2, example_seed=123)
    assert len(questions) == 1
    assert "Option (A):" in questions[0].full_question_text
    assert questions[0].metadata["statement_variant"] in {1, 2, 3}
    print("Independence Test question assembly checks passed.")
