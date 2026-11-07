"""
ate_query_sampling.py

ATE Query Sampling: ground-truth ATE computation and cross-verification.

Given an RLCausalSCM (from Step 2), computes ground-truth ATE for (treatment,
outcome) pairs via two methods:
  1) Graph surgery: model.do([treatment]) + VariableElimination (primary, always correct)
  2) CausalInference.query (verification, may raise ValueError for abnormal cases)

Only RLCausalSCM is supported as the input SCM class.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, CausalInference

from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM
from data_generation.rlcausal_dag_sampling import CausalGraphSampler


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ATEResult:
    """Ground-truth ATE computation result for a single (treatment, outcome) pair."""
    treatment: int
    outcome: int
    ate_surgery: float
    ate_causal_query: Optional[float]  # None if CausalInference raised ValueError
    is_abnormal: bool  # True when no directed path from treatment to outcome in DAG
    verification_ok: Optional[bool]  # None if abnormal; True/False for normal cases


# ============================================================
# ATE Computation
# ============================================================

def _p_of_one(factor: DiscreteFactor, var) -> float:
    """
    Extract P(var=1) from a single-variable pgmpy DiscreteFactor.

    Args:
        factor: A DiscreteFactor returned by query([var], ...).
        var: The node ID (int or str).

    Returns:
        P(var=1) as a float.

    Raises:
        ValueError: If the factor's scope is not exactly [var] or not binary.
    """
    scope = list(factor.scope())
    if scope != [var]:
        raise ValueError(
            f"Expected a 1-variable factor over [{var}], got scope={scope}"
        )
    vals = factor.values.reshape(-1)
    if len(vals) < 2:
        raise ValueError(
            f"Expected binary factor for {var}, got values={vals}"
        )
    return float(vals[1])


def _build_string_pgmpy_model(scm: SCMBase) -> DiscreteBayesianNetwork:
    """
    Build a pgmpy DiscreteBayesianNetwork with string node names from an SCM.

    pgmpy's CausalInference requires string node names. This builds a new
    model from scm.cpts with str(node_id) as variable names, ensuring correct
    correspondence between evidence ordering and CPD values.
    """
    str_edges = [(str(u), str(v)) for u, v in scm.dag.edges()]
    str_model = DiscreteBayesianNetwork(str_edges)

    for node in scm.nodes:
        str_node = str(node)
        if str_node not in str_model.nodes():
            str_model.add_node(str_node)

    for node in scm.nodes:
        parents = scm.cpts[node]['parents']
        table = scm.cpts[node]['table']
        str_node = str(node)
        str_parents = [str(p) for p in parents]

        if not parents:
            p_1 = table[()]
            cpd = TabularCPD(
                variable=str_node,
                variable_card=2,
                values=[[1.0 - p_1], [p_1]],
            )
        else:
            configs = list(itertools.product([0, 1], repeat=len(parents)))
            p_1_list = [table[config] for config in configs]
            p_0_list = [1.0 - p for p in p_1_list]
            cpd = TabularCPD(
                variable=str_node,
                variable_card=2,
                values=[p_0_list, p_1_list],
                evidence=str_parents,
                evidence_card=[2] * len(str_parents),
            )

        str_model.add_cpds(cpd)

    if not str_model.check_model():
        raise ValueError("String pgmpy model validation failed")
    return str_model


def is_abnormal_pair(dag: nx.DiGraph, treatment: int, outcome: int) -> bool:
    """Check if (treatment, outcome) is abnormal: no directed path from treatment to outcome."""
    return not nx.has_path(dag, treatment, outcome)


def ate_via_graph_surgery(
    model: DiscreteBayesianNetwork,
    treatment: int,
    outcome: int,
) -> float:
    """
    Compute ATE via graph surgery: do(treatment) + VariableElimination.

    ATE = P(outcome=1 | do(treatment=1)) - P(outcome=1 | do(treatment=0))

    This is the PRIMARY method. It works for all valid (treatment, outcome)
    pairs, including abnormal cases where outcome is an ancestor of treatment
    (ATE will be ~0.0 in such cases).
    """
    mutilated = model.do([treatment])   # do-op modifies graph + CPDs for the intervened node(s)
    ve = VariableElimination(mutilated)

    phi1 = ve.query([outcome], evidence={treatment: 1}, show_progress=False)
    phi0 = ve.query([outcome], evidence={treatment: 0}, show_progress=False)

    return _p_of_one(phi1, outcome) - _p_of_one(phi0, outcome)


def ate_via_causal_query(
    scm: SCMBase,
    treatment: int,
    outcome: int,
) -> float:
    """
    Compute ATE via pgmpy CausalInference.query with adjustment.

    ATE = P(outcome=1 | do(treatment=1)) - P(outcome=1 | do(treatment=0))

    This is the VERIFICATION method. It uses CausalInference internally,
    which may raise ValueError for "abnormal" cases (e.g., outcome is an
    ancestor of treatment).

    Note: CausalInference requires string node names, so the model is
    rebuilt with string IDs from scm.cpts before querying.

    Raises:
        ValueError: If CausalInference cannot compute the interventional
            distribution.
    """
    str_model = _build_string_pgmpy_model(scm)
    str_treatment = str(treatment)
    str_outcome = str(outcome)

    ci = CausalInference(str_model)

    phi1 = ci.query([str_outcome], do={str_treatment: 1}, show_progress=False)
    phi0 = ci.query([str_outcome], do={str_treatment: 0}, show_progress=False)

    return _p_of_one(phi1, str_outcome) - _p_of_one(phi0, str_outcome)


def compute_ate(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    atol: float = 1e-5,
    raise_on_mismatch: bool = False,
) -> ATEResult:
    """
    Compute ground-truth ATE and cross-verify with two methods.

    Primary: graph surgery (always works).
    Verification: CausalInference.query (may raise ValueError for abnormal cases).

    For normal cases, verifies the two methods agree within atol.
    For abnormal cases (ValueError from CausalInference), flags is_abnormal=True
    and uses only the surgery result.

    Args:
        raise_on_mismatch: If True, raise RuntimeError when the two methods
            disagree beyond atol. If False (default), return an ATEResult with
            verification_ok=False instead.
    """
    ate_surgery = ate_via_graph_surgery(scm.pgmpy_model, treatment, outcome)
    # Structural check: abnormal iff no directed path from treatment to outcome.
    # Using graph structure (not numerical ATE) avoids false positives from CPT
    # rounding that can make a non-zero ATE collapse to exactly 0.
    is_abnormal = is_abnormal_pair(scm.dag, treatment, outcome)

    ate_cq = None
    verification_ok = None

    try:
        ate_cq = ate_via_causal_query(scm, treatment, outcome)
    except ValueError:
        is_abnormal = True  # when `ate_via_causal_query` raises ValueError, `is_abnormal_pair` must hold, so this branch can actually be `pass`. we set `is_abnormal=True` here just for clarity and explicitness.

    if ate_cq is not None and not math.isnan(ate_cq):
        diff = abs(ate_surgery - ate_cq)
        verification_ok = diff < atol
        if not verification_ok and raise_on_mismatch:
            raise RuntimeError(
                f"ATE mismatch for treatment={treatment}, outcome={outcome}: "
                f"surgery={ate_surgery:.12f}, causal_query={ate_cq:.12f}, "
                f"diff={diff:.2e}, atol={atol}"
            )

    return ATEResult(
        treatment=treatment,
        outcome=outcome,
        ate_surgery=ate_surgery,
        ate_causal_query=ate_cq,
        is_abnormal=is_abnormal,
        verification_ok=verification_ok,
    )


# ============================================================
# TEST SUITE
# ============================================================
if __name__ == "__main__":
    import math

    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing ATE Query Sampling ===\n")

    # ==================================================================
    # Group 1: ATE computation
    # ==================================================================
    print("--- Group 1: ATE Computation ---\n")

    # Test 1: Normal — chain 0→1, ATE 0→1
    print("Test 1: Normal chain 0→1, ATE 0→1")
    chain_dag = nx.DiGraph()
    chain_dag.add_edge(0, 1)
    chain_scm = RLCausalSCM(chain_dag, seed=42)
    chain_result = compute_ate(chain_scm, 0, 1)

    assert not chain_result.is_abnormal, "Chain 0→1 should be normal"
    assert chain_result.verification_ok, "Surgery and CausalInference should agree"

    # Manual check: ATE = P(1=1|do(0=1)) - P(1=1|do(0=0)) = cpt_1[(1,)] - cpt_1[(0,)]
    cpt_1 = chain_scm.get_node_cpt(1)
    expected_ate = cpt_1[(1,)] - cpt_1[(0,)]
    assert_close(chain_result.ate_surgery, expected_ate, atol=1e-5,
                 msg=f"Manual ATE check failed: {chain_result.ate_surgery} vs {expected_ate}")
    print(f"   ATE = {chain_result.ate_surgery:.10f} (manual = {expected_ate:.10f})")
    print("   -> PASSED\n")

    # Test 2: Normal — confounder 2→0, 2→1, 0→1, ATE 0→1
    print("Test 2: Normal confounder 2→0, 2→1, 0→1, ATE 0→1")
    conf_dag = nx.DiGraph()
    conf_dag.add_edges_from([(2, 0), (2, 1), (0, 1)])
    conf_scm = RLCausalSCM(conf_dag, seed=42)
    conf_result = compute_ate(conf_scm, 0, 1)

    assert not conf_result.is_abnormal, "Confounder ATE 0→1 should be normal"
    assert conf_result.verification_ok, "Surgery and CausalInference should agree"
    assert -1.0 <= conf_result.ate_surgery <= 1.0
    print(f"   ATE = {conf_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 3: Abnormal — independent node (0→1, node 2 isolated), ATE 0→2
    # ATE is 0 because no causal path from 0 to 2, so is_abnormal=True
    print("Test 3: Abnormal independent node, ATE 0→2 (should be ~0)")
    indep_dag = nx.DiGraph()
    indep_dag.add_nodes_from([0, 1, 2])
    indep_dag.add_edge(0, 1)
    indep_scm = RLCausalSCM(indep_dag, seed=42)
    indep_result = compute_ate(indep_scm, 0, 2)

    assert indep_result.is_abnormal, "Independent ATE 0→2 should be abnormal (ATE=0)"
    assert indep_result.verification_ok
    assert_close(indep_result.ate_surgery, 0.0, atol=1e-5,
                 msg=f"ATE of 0 on independent 2 should be 0, got {indep_result.ate_surgery}")
    print(f"   ATE = {indep_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 4: Abnormal — chain 0→1, ATE 1→0 (outcome is ancestor of treatment)
    print("Test 4: Abnormal chain 0→1, ATE 1→0")
    abnormal_chain_result = compute_ate(chain_scm, 1, 0)

    assert abnormal_chain_result.is_abnormal, "ATE 1→0 on chain 0→1 should be abnormal"
    assert abnormal_chain_result.ate_causal_query is None
    assert_close(abnormal_chain_result.ate_surgery, 0.0, atol=1e-5,
                 msg=f"Surgery ATE should be ~0, got {abnormal_chain_result.ate_surgery}")
    print(f"   is_abnormal = {abnormal_chain_result.is_abnormal}")
    print(f"   ate_surgery = {abnormal_chain_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 5: Abnormal — confounder 2→0, 2→1, 0→1, ATE 1→2
    print("Test 5: Abnormal confounder, ATE 1→2")
    abnormal_conf_result = compute_ate(conf_scm, 1, 2)

    assert abnormal_conf_result.is_abnormal, "ATE 1→2 on confounder should be abnormal"
    assert abnormal_conf_result.ate_causal_query is None
    assert_close(abnormal_conf_result.ate_surgery, 0.0, atol=1e-5,
                 msg=f"Surgery ATE should be ~0, got {abnormal_conf_result.ate_surgery}")
    print(f"   is_abnormal = {abnormal_conf_result.is_abnormal}")
    print(f"   ate_surgery = {abnormal_conf_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: _p_of_one helper
    # ==================================================================
    print("--- Group 2: _p_of_one Helper ---\n")

    # Test 6: _p_of_one helper
    print("Test 6: _p_of_one helper")
    test_factor = DiscreteFactor([0], [2], [0.3, 0.7])
    assert_close(_p_of_one(test_factor, 0), 0.7, atol=1e-10)

    # Wrong scope should raise
    try:
        _p_of_one(test_factor, 99)
        raise AssertionError("Expected ValueError for wrong scope")
    except ValueError:
        pass
    print("   -> PASSED\n")

    print("=== All Tests Passed Successfully ===")
