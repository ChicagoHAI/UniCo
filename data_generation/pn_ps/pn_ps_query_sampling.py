"""
pn_ps_query_sampling.py

Probability of Necessity (PN) and Probability of Sufficiency (PS) helpers.

PN = P(Y_0 = 0 | X = 1, Y = 1)
PS = P(Y_1 = 1 | X = 0, Y = 0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from data_generation.cf.cf_query_sampling import (
    CounterfactualQueryResult,
    compute_counterfactual_result,
    is_abnormal_counterfactual,
)
from data_generation.scm_mechanism_sampling import SCMBase


@dataclass
class PNPSResult:
    """Stores PN/PS computation components for one (treatment, outcome) pair."""

    treatment: int
    outcome: int
    p_y0_eq0_given_x1_y1: float
    p_y1_eq1_given_x0_y0: float
    pn_value: float
    ps_value: float
    is_abnormal: bool
    cf_result_pn: CounterfactualQueryResult
    cf_result_ps: CounterfactualQueryResult

    def to_json_dict(self) -> Dict:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "p_y0_eq0_given_x1_y1": self.p_y0_eq0_given_x1_y1,
            "p_y1_eq1_given_x0_y0": self.p_y1_eq1_given_x0_y0,
            "pn_value": self.pn_value,
            "ps_value": self.ps_value,
            "is_abnormal": self.is_abnormal,
            "cf_result_pn": self.cf_result_pn.to_json_dict(),
            "cf_result_ps": self.cf_result_ps.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "PNPSResult":
        return cls(
            treatment=int(data["treatment"]),
            outcome=int(data["outcome"]),
            p_y0_eq0_given_x1_y1=float(data["p_y0_eq0_given_x1_y1"]),
            p_y1_eq1_given_x0_y0=float(data["p_y1_eq1_given_x0_y0"]),
            pn_value=float(data["pn_value"]),
            ps_value=float(data["ps_value"]),
            is_abnormal=bool(data["is_abnormal"]),
            cf_result_pn=CounterfactualQueryResult.from_json_dict(data["cf_result_pn"]),
            cf_result_ps=CounterfactualQueryResult.from_json_dict(data["cf_result_ps"]),
        )


def compute_pn_ps(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    atol: float = 1e-6,
) -> PNPSResult:
    """Compute PN and PS via counterfactual queries under factual evidence."""
    cf_result_pn = compute_counterfactual_result(
        scm,
        treatments={treatment: 0},
        outcome=outcome,
        evidence={treatment: 1, outcome: 1},
        atol=atol,
    )
    cf_result_ps = compute_counterfactual_result(
        scm,
        treatments={treatment: 1},
        outcome=outcome,
        evidence={treatment: 0, outcome: 0},
        atol=atol,
    )

    p_y0_eq0_given_x1_y1 = 1.0 - cf_result_pn.prob_all_ones()
    p_y1_eq1_given_x0_y0 = cf_result_ps.prob_all_ones()
    return PNPSResult(
        treatment=treatment,
        outcome=outcome,
        p_y0_eq0_given_x1_y1=p_y0_eq0_given_x1_y1,
        p_y1_eq1_given_x0_y0=p_y1_eq1_given_x0_y0,
        pn_value=p_y0_eq0_given_x1_y1,
        ps_value=p_y1_eq1_given_x0_y0,
        is_abnormal=is_abnormal_counterfactual(scm.dag, {treatment: 1}, outcome),
        cf_result_pn=cf_result_pn,
        cf_result_ps=cf_result_ps,
    )


if __name__ == "__main__":
    import networkx as nx

    from data_generation.scm_mechanism_sampling import RLCausalSCM

    def assert_close(a, b, *, atol=1e-6, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing PN/PS Query Sampling ===\n")

    print("Test 1: JSON round-trip for PNPSResult")
    chain_dag = nx.DiGraph()
    chain_dag.add_edge(0, 1)
    chain_scm = RLCausalSCM(chain_dag, seed=42)
    pn_ps_result = compute_pn_ps(chain_scm, treatment=0, outcome=1)
    pn_ps_roundtrip = PNPSResult.from_json_dict(pn_ps_result.to_json_dict())
    assert pn_ps_roundtrip.treatment == 0
    assert pn_ps_roundtrip.outcome == 1
    assert_close(pn_ps_roundtrip.pn_value, pn_ps_result.pn_value)
    assert_close(pn_ps_roundtrip.ps_value, pn_ps_result.ps_value)
    print("   -> PASSED\n")

    print("Test 2: Deterministic positive effect gives PN=PS=1")
    deterministic_dag = nx.DiGraph([(0, 1)])
    deterministic_cpts = {
        0: {"parents": [], "table": {(): 0.5}},
        1: {"parents": [0], "table": {(0,): 0.0, (1,): 1.0}},
    }
    deterministic_scm = RLCausalSCM.from_cpts(
        deterministic_dag,
        deterministic_cpts,
        decimal_places=4,
    )
    deterministic_result = compute_pn_ps(deterministic_scm, treatment=0, outcome=1)
    assert_close(deterministic_result.pn_value, 1.0)
    assert_close(deterministic_result.ps_value, 1.0)
    print("   -> PASSED\n")

    print("Test 3: Abnormality matches CF-style treatment->outcome reachability")
    disconnected_dag = nx.DiGraph()
    disconnected_dag.add_edges_from([(2, 0), (2, 1)])
    disconnected_scm = RLCausalSCM(disconnected_dag, seed=7)
    abnormal_result = compute_pn_ps(disconnected_scm, treatment=0, outcome=1)
    assert abnormal_result.is_abnormal is True
    assert_close(abnormal_result.pn_value, 0.0)
    assert_close(abnormal_result.ps_value, 0.0)
    print("   -> PASSED\n")

    print("Test 4: Counterfactual normalization flags propagate")
    assert pn_ps_result.cf_result_pn.normalization_ok is True
    assert pn_ps_result.cf_result_ps.normalization_ok is True
    print("   -> PASSED\n")

    print("=== All PN/PS Query Sampling Tests Passed ===")
