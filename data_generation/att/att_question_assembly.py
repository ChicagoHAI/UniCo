"""
att_question_assembly.py

ATT question assembly in the local ATE house style.
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
from data_generation.att.att_query_sampling import ATTResult
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
    variant_name: str
    att_ground_truth: float
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
    return _ate_condition_variant_full(
        scm,
        treatment,
        outcome,
        decimal_places=decimal_places,
    )


def condition_variant_surgery_ancestor_prune(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    # ATT uses the same counterfactual-style pruning rule as CF. Since ATT
    # conditions on factual T=1, pruning must preserve both sides of the
    # abduction-action-prediction computation:
    # 1. Factual abduction: keep the factual evidence node T and its ancestors
    #    in the original DAG, because those mechanisms can matter for the
    #    treated-only conditioning event.
    # 2. Counterfactual prediction: keep the outcome node and its ancestors in
    #    the mutilated graph after do(T), because those mechanisms can affect
    #    the post-intervention outcome.
    # Keep the union of those sets and do not automatically remove treatment.
    evidence = {treatment: 1}
    factual_nodes = set(evidence.keys())
    for node in evidence:
        factual_nodes.update(nx.ancestors(scm.dag, node))

    mutilated = scm.pgmpy_model.do([treatment])
    outcome_anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])
    counterfactual_nodes = set(outcome_anc_graph.nodes())

    cpd_nodes = factual_nodes | counterfactual_nodes
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


def _render_query_text(treatment: int, outcome: int, *, query_variant: int) -> str:
    t_label = f'X{treatment}'
    o_label = f'X{outcome}'
    formal = (
        f'What is the Average Treatment Effect on the Treated (ATT) of {t_label} '
        f'on {o_label}?'
    )

    if query_variant == 1:
        variation = (
            f'That is, among units that actually received treatment '
            f'({t_label} = 1), what is the expected difference between the '
            f'probability that {o_label} would be 1 under forcing {t_label} = 1 '
            f'and under forcing {t_label} = 0?'
        )
    elif query_variant == 2:
        variation = (
            f'In other words, for the treated units with factual {t_label} = 1, '
            f'how much larger or smaller would the probability of {o_label} = 1 '
            f'be under intervention {t_label} = 1 than under the counterfactual '
            f'intervention {t_label} = 0?'
        )
    elif query_variant == 3:
        variation = (
            f'Equivalently, restricting attention to units that were actually '
            f'treated ({t_label} = 1), what is the average causal effect of '
            f'changing {t_label} from 0 to 1 on the probability that {o_label} = 1?'
        )
    else:
        raise ValueError(f'Unknown query_variant: {query_variant}')

    return '\n'.join([formal, variation])


ANSWER_FORMAT_TEMPLATE = (
    'Think step-by-step to solve this causal reasoning question. '
    'Your final answer should be rounded to four decimal places '
    'and included inside \\boxed{}, as below: \\boxed{<your final answer>}.'
)


def assemble_question(
    scm: SCMBase,
    att_result: ATTResult,
    variant: ConditionVariant,
    *,
    graph_variant: int,
    query_variant: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> AssembledQuestion:
    treatment = att_result.treatment
    outcome = att_result.outcome
    graph_text = _render_graph_text(scm, graph_variant)
    query_text = _render_query_text(
        treatment,
        outcome,
        query_variant=query_variant,
    )

    full_question_text = (
        f'{graph_text}\n\n'
        f'The following probability relationships hold:\n'
        f'{variant.conditions_text}\n\n'
        f'{query_text}\n\n'
        f'{ANSWER_FORMAT_TEMPLATE}'
    )

    metadata = {
        'num_nodes': len(scm.nodes),
        'num_edges': scm.dag.number_of_edges(),
        'dag_edges': sorted(list(scm.dag.edges())),
        'treatment': treatment,
        'outcome': outcome,
        'variant_name': variant.variant_name,
        'is_abnormal': att_result.is_abnormal,
        'graph_variant': graph_variant,
        'query_variant': query_variant,
        'nodes_kept': variant.nodes_kept,
        'nodes_pruned': variant.nodes_pruned,
        'p_y1_do1_given_treated': att_result.p_y1_do1_given_treated,
        'p_y1_do0_given_treated': att_result.p_y1_do0_given_treated,
        'att_value': att_result.att_value,
        'cf_result_do1': att_result.cf_result_do1.to_json_dict(),
        'cf_result_do0': att_result.cf_result_do0.to_json_dict(),
        'treatment_outcome_depth': treatment_outcome_depth,
        'effective_prob_conds': effective_prob_conds,
        'has_naive_conditioning': has_naive_conditioning,
    }

    return AssembledQuestion(
        treatment=treatment,
        outcome=outcome,
        variant_name=variant.variant_name,
        att_ground_truth=att_result.att_value,
        is_abnormal=att_result.is_abnormal,
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
    att_result: ATTResult,
    *,
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
        att_result.treatment,
        att_result.outcome,
        decimal_places=decimal_places,
    )

    return [
        assemble_question(
            scm,
            att_result,
            variant,
            graph_variant=graph_variant,
            query_variant=query_variant,
            treatment_outcome_depth=treatment_outcome_depth,
            effective_prob_conds=effective_prob_conds,
            has_naive_conditioning=has_naive_conditioning,
        )
        for variant in variants
    ]


if __name__ == '__main__':
    from data_generation.att.att_query_sampling import compute_att

    def assert_close(a, b, *, atol=1e-5, msg=''):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f'Not close: {a} vs {b} (atol={atol})')

    print('=== Testing ATT Question Assembly ===\n')

    print('Test 1: ATT condition variants for chain 0->1->2')
    chain_dag = nx.DiGraph()
    chain_dag.add_edges_from([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)
    variants = generate_condition_variants(chain_scm, treatment=0, outcome=2)
    assert len(variants) == 2
    assert variants[0].nodes_kept == [0, 1, 2]
    assert variants[1].nodes_kept == [0, 1, 2]
    print('   -> PASSED\n')

    print('Test 2: assemble_question stores ATT metadata correctly')
    att_result = compute_att(chain_scm, treatment=0, outcome=2)
    question = assemble_question(
        chain_scm,
        att_result,
        variants[1],
        graph_variant=2,
        query_variant=3,
        treatment_outcome_depth=2.0,
        effective_prob_conds=3,
        has_naive_conditioning=False,
    )
    assert question.variant_name == 'surgery_ancestor_prune'
    assert question.metadata['treatment'] == 0
    assert question.metadata['outcome'] == 2
    assert question.metadata['graph_variant'] == 2
    assert question.metadata['query_variant'] == 3
    assert 'Average Treatment Effect on the Treated' in question.query_text
    assert '\\boxed{' in question.full_question_text
    assert_close(question.att_ground_truth, att_result.att_value)
    print('   -> PASSED\n')

    print('Test 3: assemble_all_questions uses shared graph/query variants')
    assembled = assemble_all_questions(
        chain_scm,
        att_result,
        decimal_places=2,
        example_seed=123,
        treatment_outcome_depth=2.0,
        effective_prob_conds=3,
        has_naive_conditioning=False,
    )
    assert len(assembled) == 2
    assert {q.variant_name for q in assembled} == {
        'full',
        'surgery_ancestor_prune',
    }
    assert len({q.graph_variant for q in assembled}) == 1
    assert len({q.query_variant for q in assembled}) == 1
    print('   -> PASSED\n')

    print('=== All ATT Question Assembly Tests Passed ===')
