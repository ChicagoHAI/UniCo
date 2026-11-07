"""
cf_question_assembly.py

Counterfactual question assembly mirroring the ATE layout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from data_generation.ate.ate_question_assembly import (
    _node_label,
    _render_graph_text,
    _topo_sorted_subset,
    condition_variant_full as _ate_condition_variant_full,
    render_node_cpt,
)
from data_generation.cf.cf_query_sampling import CounterfactualQueryResult
from data_generation.scm_mechanism_sampling import SCMBase


@dataclass
class ConditionVariant:
    """One of three condition variants for a question."""

    variant_name: str
    conditions_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]


@dataclass
class AssembledQuestion:
    """A complete counterfactual question for one variant."""

    treatments: Dict[int, int]
    outcome: int
    evidence: Dict[int, int]
    variant_name: str
    cf_ground_truth: float
    graph_variant: int
    query_variant: int
    graph_text: str
    conditions_text: str
    query_text: str
    full_question_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


def _render_assumption_text(graph_variant: int) -> str:
    if graph_variant == 1:
        return (
            'Furthermore, each variable v is assumed to depend deterministically on its '
            'parents pa(v) and a collection of independent exogenous selector '
            'variables, one for each possible joint assignment to pa(v), whose '
            'marginal distribution is defined to be p(v | pa(v)). Given a particular '
            'assignment to pa(v), v takes on the value of the selector variable '
            'corresponding to that particular assignment pa(v).'
        )
    if graph_variant == 2:
        return (
            'Assume a Markovian SCM with surgical interventions evaluated via '
            'abduction--action--prediction. Specifically, each variable v is '
            'generated from a family of independent exogenous selector variables '
            'indexed by the possible joint assignments of its parents pa(v): for '
            'each assignment a, the selector U_{v,a} satisfies P(U_{v,a}=1)=P(v=1 | '
            'pa(v)=a), and when pa(v)=a, the variable v takes the value U_{v,a}.'
        )
    if graph_variant == 3:
        return (
            'For every node v, we assume a separate exogenous variable U_{v,a} for '
            'each possible parent configuration a in Val(pa(v)), with all such '
            'variables mutually independent and satisfying P(U_{v,a}=1)=P(v=1 | '
            'pa(v)=a). The structural equation for v then selects the exogenous '
            'variable indexed by the realized parent values, so that if pa(v)=a, we '
            'set v=U_{v,a}.'
        )
    raise ValueError(f'Unknown graph_variant: {graph_variant}')


def condition_variant_full(
    scm: SCMBase,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    return _ate_condition_variant_full(
        scm,
        treatment=0,
        outcome=0,
        decimal_places=decimal_places,
    )


def condition_variant_surgery_ancestor_prune(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    # Counterfactual ancestor pruning keeps only the nodes needed for the two
    # inference stages of a counterfactual query:
    # 1. Abduction in the factual world: keep evidence nodes and their ancestors
    #    in the original DAG, since those mechanisms can matter for explaining
    #    the observed factual evidence.
    # 2. Prediction in the counterfactual world: keep the outcome node and its
    #    ancestors in the mutilated graph after do(treatments), since those
    #    mechanisms can affect the post-intervention counterfactual outcome.
    # Any node outside the union of those two relevance sets is pruned. 
    do_nodes = sorted(treatments.keys())
    evidence_nodes = sorted(evidence.keys())

    factual_relevant_nodes = {outcome}
    if evidence_nodes:
        factual_relevant_nodes = set(evidence_nodes)
        for evidence_node in evidence_nodes:
            factual_relevant_nodes.update(nx.ancestors(scm.dag, evidence_node))

    mutilated = scm.pgmpy_model.do(do_nodes)
    outcome_anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])
    counterfactual_relevant_nodes = set(outcome_anc_graph.nodes())

    cpd_nodes = factual_relevant_nodes | counterfactual_relevant_nodes
    topo_nodes = _topo_sorted_subset(scm.dag, cpd_nodes)
    cpt_blocks = [
        render_node_cpt(scm, node, decimal_places=decimal_places)
        for node in topo_nodes
    ]
    return ConditionVariant(
        variant_name='surgery_ancestor_prune',
        conditions_text='\n'.join(cpt_blocks),
        nodes_kept=sorted(cpd_nodes),
        nodes_pruned=sorted(set(scm.nodes) - cpd_nodes),
    )


def generate_condition_variants(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    decimal_places: int = 2,
) -> List[ConditionVariant]:
    return [
        condition_variant_full(
            scm,
            decimal_places=decimal_places,
        ),
        condition_variant_surgery_ancestor_prune(
            scm,
            treatments,
            outcome,
            evidence,
            decimal_places=decimal_places,
        ),
    ]


def _format_intervention_subscript(treatments: Dict[int, int]) -> str:
    return ', '.join(
        f'{_node_label(node)}={value}'
        for node, value in sorted(treatments.items())
    )


def _format_assignment_plain(assignments: Dict[int, int]) -> str:
    terms = [
        f'{_node_label(node)} = {value}'
        for node, value in sorted(assignments.items())
    ]
    if not terms:
        return '(none)'
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f'{terms[0]} and {terms[1]}'
    return ', '.join(terms[:-1]) + f', and {terms[-1]}'


def _format_assignment_factual(assignments: Dict[int, int]) -> str:
    terms = [
        f'{_node_label(node)}^F = {value}'
        for node, value in sorted(assignments.items())
    ]
    if not terms:
        return '(none)'
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f'{terms[0]} and {terms[1]}'
    return ', '.join(terms[:-1]) + f', and {terms[-1]}'


def _format_outcome_phrase(outcome: int) -> str:
    terms = [_node_label(node) for node in [outcome]]
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f'{terms[0]} and {terms[1]}'
    return ', '.join(terms[:-1]) + f', and {terms[-1]}'


def _render_query_text(
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    query_variant: int,
) -> str:
    intervention_subscript = _format_intervention_subscript(treatments)
    formal_targets = ', '.join(
        f'{_node_label(node)}_{{{intervention_subscript}}} = 1'
        for node in [outcome]
    )
    factual_plain = _format_assignment_plain(evidence)
    force_phrase = _format_assignment_plain(treatments)
    outcome_phrase = _format_outcome_phrase(outcome)

    formal_question = (
        'What is the expected value of '
        f'P({formal_targets} \\mid {factual_plain})?'
    )

    if query_variant == 1:
        variation = (
            f'That is, if we observe that in the actual world {factual_plain}, '
            f'what is the probability that {outcome_phrase} would be 1 if we were '
            f'to force {force_phrase}?'
        )
    elif query_variant == 2:
        variation = (
            f'In other words, for units with factual evidence {factual_plain}, '
            f'how likely is {outcome_phrase} to equal 1 in the world where '
            f'{force_phrase} is forced?'
        )
    elif query_variant == 3:
        variation = (
            f'Equivalently, for units observed with {factual_plain} in the actual '
            f'world, what is the expected counterfactual value of {outcome_phrase} '
            f'under an intervention that fixes {force_phrase}?'
        )

    return '\n'.join(
        [
            formal_question,
            variation,
        ]
    )



ANSWER_FORMAT_TEMPLATE = (
    "Think step-by-step to solve this causal reasoning question. "
    "Your final answer should be rounded to four decimal places "
    "and included inside \\boxed{}, as below: \\boxed{<your final answer>}."
)


def assemble_question(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    cf_result: CounterfactualQueryResult,
    variant: ConditionVariant,
    *,
    graph_variant: int,
    query_variant: int,
) -> AssembledQuestion:
    graph_text = _render_graph_text(scm, graph_variant)
    assumption_text = _render_assumption_text(graph_variant)
    query_text = _render_query_text(
        treatments,
        outcome,
        evidence,
        query_variant=query_variant,
    )
    answer_prob = cf_result.prob_all_ones()

    full_question_text = (
        f'{graph_text}\n\n'
        f'The following probability relationships hold:\n'
        f'{variant.conditions_text}\n\n'
        f'{assumption_text}\n\n'
        f'{query_text}\n\n'
        f'{ANSWER_FORMAT_TEMPLATE}'
    )

    metadata = {
        'num_nodes': len(scm.nodes),
        'num_edges': scm.dag.number_of_edges(),
        'dag_edges': sorted(list(scm.dag.edges())),
        'treatments': treatments,
        'outcome': outcome,
        'evidence': evidence,
        'variant_name': variant.variant_name,
        'graph_variant': graph_variant,
        'query_variant': query_variant,
        'nodes_kept': variant.nodes_kept,
        'nodes_pruned': variant.nodes_pruned,
        'cf_result': cf_result.to_json_dict(),
        'cf_answer_assignment': 1,
        'cf_answer_probability': answer_prob,
    }

    return AssembledQuestion(
        treatments=treatments,
        outcome=outcome,
        evidence=evidence,
        variant_name=variant.variant_name,
        cf_ground_truth=answer_prob,
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
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    cf_result: CounterfactualQueryResult,
    *,
    decimal_places: int = 2,
    example_seed: int,
) -> List[AssembledQuestion]:
    rng = random.Random(example_seed)
    graph_variant = rng.randint(1, 3)
    query_variant = rng.randint(1, 3)

    variants = generate_condition_variants(
        scm,
        treatments,
        outcome,
        evidence,
        decimal_places=decimal_places,
    )

    return [
        assemble_question(
            scm,
            treatments,
            outcome,
            evidence,
            cf_result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
        )
        for variant in variants
    ]



# ============================================================
# TEST SUITE
# ============================================================
if __name__ == "__main__":
    import math

    from data_generation.cf.cf_query_sampling import compute_counterfactual_result
    from data_generation.scm_mechanism_sampling import RLCausalSCM

    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing CF Question Assembly ===\n")

    # ==================================================================
    # Group 1: Condition variants (base cases)
    # ==================================================================
    print("--- Group 1: Condition Variants (Base Cases) ---\n")

    print("Test 1: Full and ancestor-prune variants for chain 0->1->2")
    chain_dag = nx.DiGraph()
    chain_dag.add_edges_from([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)

    chain_variants = generate_condition_variants(
        chain_scm,
        treatments={0: 1},
        outcome=2,
        evidence={1: 1},
    )
    assert len(chain_variants) == 2

    # Full: all nodes.
    assert chain_variants[0].variant_name == "full"
    assert chain_variants[0].nodes_kept == [0, 1, 2]
    assert chain_variants[0].nodes_pruned == []

    # Ancestor-prune CF rule: keep ancestors of evidence in original DAG ({0,1})
    # union ancestors of outcome in mutilated graph after do(0) ({1,2}), yielding all nodes.
    assert chain_variants[1].variant_name == "surgery_ancestor_prune"
    assert chain_variants[1].nodes_kept == [0, 1, 2]
    assert chain_variants[1].nodes_pruned == []
    print("   -> PASSED\n")

    print("Test 2: Ancestor-prune removes irrelevant node in fork-plus-isolated graph")
    prune_dag = nx.DiGraph()
    prune_dag.add_edges_from([(0, 1), (1, 2), (3, 4)])
    prune_scm = RLCausalSCM(prune_dag, seed=7)

    prune_variants = generate_condition_variants(
        prune_scm,
        treatments={0: 1},
        outcome=2,
        evidence={1: 0},
    )
    assert len(prune_variants) == 2
    assert prune_variants[1].nodes_kept == [0, 1, 2]
    assert prune_variants[1].nodes_pruned == [3, 4]
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: Full question assembly
    # ==================================================================
    print("--- Group 2: Full Question Assembly ---\n")

    print("Test 3: assemble_question stores CF metadata and answer correctly")
    cf_result = compute_counterfactual_result(
        chain_scm,
        treatments={0: 1},
        outcome=2,
        evidence={1: 1},
    )
    variant = chain_variants[1]
    question = assemble_question(
        chain_scm,
        treatments={0: 1},
        outcome=2,
        evidence={1: 1},
        cf_result=cf_result,
        variant=variant,
        graph_variant=2,
        query_variant=3,
    )
    print(question.full_question_text)
    assert question.variant_name == "surgery_ancestor_prune"
    assert question.metadata["treatments"] == {0: 1}
    assert question.metadata["outcome"] == 2
    assert question.metadata["evidence"] == {1: 1}
    assert question.metadata["graph_variant"] == 2
    assert question.metadata["query_variant"] == 3
    assert question.metadata["nodes_kept"] == [0, 1, 2]
    assert question.metadata["nodes_pruned"] == []
    assert "expected value of P(" in question.query_text
    assert "\\boxed{" in question.full_question_text
    assert_close(question.cf_ground_truth, cf_result.prob_all_ones())
    assert_close(question.metadata["cf_answer_probability"], cf_result.prob_all_ones())
    print("   -> PASSED\n")

    # ==================================================================
    # Group 3: assemble_all_questions determinism and shared variants
    # ==================================================================
    print("--- Group 3: Deterministic Variant Sampling ---\n")

    print("Test 4: assemble_all_questions uses one graph/query variant pair per example")
    assembled = assemble_all_questions(
        chain_scm,
        treatments={0: 1},
        outcome=2,
        evidence={1: 1},
        cf_result=cf_result,
        decimal_places=2,
        example_seed=123,
    )
    assert len(assembled) == 2
    assert {q.variant_name for q in assembled} == {"full", "surgery_ancestor_prune"}
    assert len({q.graph_variant for q in assembled}) == 1
    assert len({q.query_variant for q in assembled}) == 1
    for q in assembled:
        assert_close(q.cf_ground_truth, cf_result.prob_all_ones())
        assert q.metadata["cf_result"]["outcome"] == 2
        assert q.metadata["cf_result"]["evidence"] == {"1": 1}
    print("   -> PASSED\n")

    print("=== All CF Question Assembly Tests Passed ===")
