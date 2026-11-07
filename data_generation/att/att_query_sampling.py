"""
att_query_sampling.py

Average Treatment Effect on the Treated (ATT) helpers.

ATT = P(Y=1 | do(T=1), T^F=1) - P(Y=1 | do(T=0), T^F=1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import networkx as nx

from data_generation.ate.ate_query_sampling import is_abnormal_pair
from data_generation.cf.cf_query_sampling import (
    CounterfactualQueryResult,
    compute_counterfactual_result,
)
from data_generation.scm_mechanism_sampling import SCMBase


@dataclass
class ATTResult:
    """Stores ATT computation components for one (treatment, outcome) pair."""

    treatment: int
    outcome: int
    p_y1_do1_given_treated: float
    p_y1_do0_given_treated: float
    att_value: float
    is_abnormal: bool
    cf_result_do1: CounterfactualQueryResult
    cf_result_do0: CounterfactualQueryResult

    def to_json_dict(self) -> Dict:
        return {
            'treatment': self.treatment,
            'outcome': self.outcome,
            'p_y1_do1_given_treated': self.p_y1_do1_given_treated,
            'p_y1_do0_given_treated': self.p_y1_do0_given_treated,
            'att_value': self.att_value,
            'is_abnormal': self.is_abnormal,
            'cf_result_do1': self.cf_result_do1.to_json_dict(),
            'cf_result_do0': self.cf_result_do0.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> 'ATTResult':
        return cls(
            treatment=int(data['treatment']),
            outcome=int(data['outcome']),
            p_y1_do1_given_treated=float(data['p_y1_do1_given_treated']),
            p_y1_do0_given_treated=float(data['p_y1_do0_given_treated']),
            att_value=float(data['att_value']),
            is_abnormal=bool(data['is_abnormal']),
            cf_result_do1=CounterfactualQueryResult.from_json_dict(data['cf_result_do1']),
            cf_result_do0=CounterfactualQueryResult.from_json_dict(data['cf_result_do0']),
        )


def compute_att(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    atol: float = 1e-6,
) -> ATTResult:
    """Compute ATT via two counterfactual queries conditioned on factual T=1."""
    evidence = {treatment: 1}

    cf_result_do1 = compute_counterfactual_result(
        scm,
        treatments={treatment: 1},
        outcome=outcome,
        evidence=evidence,
        atol=atol,
    )
    cf_result_do0 = compute_counterfactual_result(
        scm,
        treatments={treatment: 0},
        outcome=outcome,
        evidence=evidence,
        atol=atol,
    )

    p_do1 = cf_result_do1.prob_all_ones()
    p_do0 = cf_result_do0.prob_all_ones()
    return ATTResult(
        treatment=treatment,
        outcome=outcome,
        p_y1_do1_given_treated=p_do1,
        p_y1_do0_given_treated=p_do0,
        att_value=p_do1 - p_do0,
        is_abnormal=is_abnormal_pair(scm.dag, treatment, outcome),
        cf_result_do1=cf_result_do1,
        cf_result_do0=cf_result_do0,
    )


if __name__ == '__main__':
    from data_generation.scm_mechanism_sampling import RLCausalSCM

    def assert_close(a, b, *, atol=1e-6, msg=''):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f'Not close: {a} vs {b} (atol={atol})')

    print('=== Testing ATT Query Sampling ===\n')

    print('Test 1: JSON round-trip for ATTResult')
    chain_dag = nx.DiGraph()
    chain_dag.add_edge(0, 1)
    chain_scm = RLCausalSCM(chain_dag, seed=42)
    att_result = compute_att(chain_scm, treatment=0, outcome=1)
    att_roundtrip = ATTResult.from_json_dict(att_result.to_json_dict())
    assert att_roundtrip.treatment == 0
    assert att_roundtrip.outcome == 1
    assert_close(att_roundtrip.att_value, att_result.att_value)
    assert_close(
        att_roundtrip.cf_result_do1.prob_all_ones(),
        att_result.cf_result_do1.prob_all_ones(),
    )
    assert_close(
        att_roundtrip.cf_result_do0.prob_all_ones(),
        att_result.cf_result_do0.prob_all_ones(),
    )
    print('   -> PASSED\n')

    print('Test 2: ATT decomposes as p_do1 - p_do0')
    assert_close(
        att_result.att_value,
        att_result.p_y1_do1_given_treated - att_result.p_y1_do0_given_treated,
    )
    print('   -> PASSED\n')

    print('Test 3: Abnormality matches treatment->outcome path check')
    disconnected_dag = nx.DiGraph()
    disconnected_dag.add_edges_from([(2, 0), (2, 1)])
    disconnected_scm = RLCausalSCM(disconnected_dag, seed=7)
    abnormal_result = compute_att(disconnected_scm, treatment=0, outcome=1)
    assert abnormal_result.is_abnormal is True
    normal_result = compute_att(chain_scm, treatment=0, outcome=1)
    assert normal_result.is_abnormal is False
    print('   -> PASSED\n')

    print('Test 4: Abnormal ATT is zero')
    assert_close(abnormal_result.att_value, 0.0)
    print('   -> PASSED\n')

    print('=== All ATT Query Sampling Tests Passed ===')
