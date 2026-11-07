"""
cate_jate_query_sampling.py

CATE/JATE Query Sampling: ground-truth effect computation and cross-verification.

Given an RLCausalSCM (from Step 2), computes ground-truth CATE or JATE for
(treatments, outcome, evidence) queries via two methods:
  1) Graph surgery: model.do(treatments) + VariableElimination (primary, always correct)
  2) CausalInference.query (verification, may raise ValueError for abnormal cases;
     is a cross-check for CATE)

Only RLCausalSCM is supported as the input SCM class.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Union

import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, CausalInference

from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


# ============================================================
# Data Structures
# ============================================================

@dataclass
class CATEJATEResult:
    """Ground-truth effect computation result for a CATE/JATE query."""
    treatments: List[int]
    outcome: int
    evidence: Dict[int, int]            # {node: value} for CATE, {} for JATE
    query_type: str                     # "cate" | "jate"
    effect_surgery: float               # primary: always correct
    effect_causal_query: Optional[float]  # verification: may be None or wrong for CATE
    is_abnormal: bool
    verification_ok: Optional[bool]     # None if abnormal or CI failed; True/False otherwise


# ============================================================
# Local Utilities
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
    model from scm.cpts with str(node_id) as variable names.
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


def _determine_query_type(
    treatments: List[int],
    evidence: Dict[int, int],
) -> str:
    """
    Determine query type from treatments and evidence.

    Returns:
        "cate" if single treatment with evidence,
        "jate" if multiple treatments without evidence.

    Raises:
        ValueError: For unsupported shapes (multi-treatment + evidence,
            no treatments, etc.)
    """
    if len(treatments) == 0:
        raise ValueError("No treatments provided.")
    if len(treatments) >= 2 and len(evidence) > 0:
        raise ValueError(
            f"Unsupported shape: multiple treatments AND evidence. "
            f"treatments={treatments}, evidence={evidence}"
        )
    if len(treatments) == 1 and len(evidence) > 0:
        return "cate"
    if len(treatments) >= 2 and len(evidence) == 0:
        return "jate"
    # Single treatment, no evidence — treat as degenerate CATE (plain ATE)
    return "cate"


# ============================================================
# Abnormality Checking
# ============================================================

Node = Hashable


def _as_list(x: Union[Node, Sequence[Node], None]) -> list[Node]:
    if x is None:
        return []
    # Strings are Sequences; treat them as atomic node IDs.
    if isinstance(x, str) or not isinstance(x, Sequence):
        return [x]  # type: ignore[list-item]
    return list(x)


def _has_directed_path(dag: nx.DiGraph, a: Node, b: Node) -> bool:
    return nx.has_path(dag, a, b)


def is_abnormal_triplets(
    dag: nx.DiGraph,
    treatments: Union[Node, Sequence[Node]],
    outcome: Node,
    evidence: Optional[Union[Node, Sequence[Node]]] = None,
    *,
    max_treatments: int = 2,
    max_evidence: int = 2,
    jate_relaxed: Optional[bool] = None,
    cate_relaxed_mode: Optional[str] = None,
) -> bool:
    """
    Return True if the query is "abnormal" (weird / corner-ish) by
    graph-structural criteria.

    IMPORTANT: this function *raises* ValueError for sampling/process errors
    (nodes not in graph, overlaps/duplicates, unsupported query shapes).

    Supported shapes:
      - CATE: exactly 1 treatment AND at least 1 evidence variable
      - JATE: at least 2 treatments AND no evidence

    Abnormality constraints — labelled systematically.  Which ones are
    active depends on the mode parameters.

    Universal constraints (all query types):
      All-1  At least one treatment has a directed path to the outcome.
      All-2  Every treatment has a directed path to the outcome.
             (Strict JATE only; redundant for single-treatment queries.)

    CATE-only constraints:
      CATE-1  No post-treatment evidence: reject if T→E.  (Strict only.)
      CATE-2  Evidence must be ancestor of T or ancestor of Y.
              (Strict and ancestor_evidence modes.)
      CATE-3  No causal ordering among evidence: reject if E1→E2.
              (Strict only.)
      CATE-4  Evidence must be d-connected to Y in the mutilated graph
              G^{do(T)} given the remaining evidence variables.
              (non_dsep_evidence mode only.)

    JATE-only constraints:
      JATE-1  No causal relations among treatments: reject if T1→T2.
              (Strict only.)
      JATE-2  Outcome not ancestor of any treatment: reject if Y→T.
              (Always for JATE.)

    Mode → active constraints:
      Strict CATE  (cate_relaxed_mode=None)           : All-1, CATE-1, CATE-2, CATE-3
      Ancestor CATE ("ancestor_evidence")              : All-1, CATE-2
      D-sep CATE   ("non_dsep_evidence")               : All-1, CATE-4
      Strict JATE  (jate_relaxed=None)                 : All-1, All-2, JATE-1, JATE-2
      Relaxed JATE (jate_relaxed=True)                 : All-1, JATE-2

    Args:
        dag: The causal DAG (nx.DiGraph).
        treatments: Treatment node(s).
        outcome: Outcome node.
        evidence: Evidence node(s), or None.
        max_treatments: Max allowed number of treatments.
        max_evidence: Max allowed number of evidence variables.
        jate_relaxed: None = strict JATE, True = relaxed.
        cate_relaxed_mode: None = strict CATE, or a relaxed mode:
            "ancestor_evidence" or "non_dsep_evidence".
    """
    if cate_relaxed_mode is not None and cate_relaxed_mode not in (
        "ancestor_evidence", "non_dsep_evidence",
    ):
        raise ValueError(f"Unknown cate_relaxed_mode: {cate_relaxed_mode!r}")

    T = _as_list(treatments)
    E = _as_list(evidence)

    # -----------------------
    # Sanity checks => RAISE
    # -----------------------
    # membership
    all_vars = T + [outcome] + E
    missing = [v for v in all_vars if v not in dag.nodes]
    if missing:
        raise ValueError(f"Sampled node(s) not in dag: {missing}")

    # duplicates inside T or E
    if len(set(T)) != len(T):
        raise ValueError(f"Duplicate(s) in treatments: {T}")
    if len(set(E)) != len(E):
        raise ValueError(f"Duplicate(s) in evidence: {E}")

    # overlaps between groups
    if outcome in T:
        raise ValueError(f"Outcome appears in treatments: outcome={outcome}, treatments={T}")
    if outcome in E:
        raise ValueError(f"Outcome appears in evidence: outcome={outcome}, evidence={E}")
    overlap_TE = set(T) & set(E)
    if overlap_TE:
        raise ValueError(f"Treatments overlap with evidence: {sorted(overlap_TE)}")

    # shape constraints
    if len(T) == 0:
        raise ValueError("No treatments provided.")
    if len(T) > max_treatments:
        raise ValueError(f"Too many treatments (>{max_treatments}): {T}")
    if len(E) > max_evidence:
        raise ValueError(f"Too many evidence vars (>{max_evidence}): {E}")

    # no query contains both multiple treatments and evidence simultaneously
    if len(T) >= 2 and len(E) > 0:
        raise ValueError(
            f"Unsupported shape: multiple treatments AND evidence provided. treatments={T}, evidence={E}"
        )

    is_cate = (len(T) == 1) and (len(E) > 0)
    is_jate = (len(T) >= 2) and (len(E) == 0)

    # -----------------------
    # Abnormality checks => RETURN True/False
    # -----------------------

    # All-1: at least one treatment→outcome directed path
    if len(T) == 1:
        if not _has_directed_path(dag, T[0], outcome):
            return True
    else:
        reaches = [_has_directed_path(dag, t, outcome) for t in T]
        if not any(reaches):                                     # All-1
            return True
        # All-2: every treatment reaches outcome (strict JATE only)
        if is_jate and not jate_relaxed and not all(reaches):    # All-2
            return True

    # CATE filters — dispatch on cate_relaxed_mode
    if is_cate:
        t = T[0]

        if cate_relaxed_mode is None:
            # Strict mode: CATE-1, CATE-2, CATE-3
            anc_t = nx.ancestors(dag, t)
            anc_y = nx.ancestors(dag, outcome)
            for e in E:
                if _has_directed_path(dag, t, e):                # CATE-1
                    return True
                if e not in anc_t and e not in anc_y:            # CATE-2
                    return True
            for e1, e2 in itertools.permutations(E, 2):          # CATE-3
                if _has_directed_path(dag, e1, e2):
                    return True
            return False

        elif cate_relaxed_mode == "ancestor_evidence":
            # ancestor_evidence: only CATE-2
            anc_t = nx.ancestors(dag, t)
            anc_y = nx.ancestors(dag, outcome)
            for e in E:
                if e not in anc_t and e not in anc_y:            # CATE-2
                    return True
            return False

        else:  # "non_dsep_evidence"
            # non_dsep_evidence: only CATE-4
            mutilated = dag.copy()
            mutilated.remove_edges_from(list(dag.in_edges(t)))
            for e in E:
                other_evidence = set(E) - {e}
                # CATE-4: reject if e is d-separated from outcome
                if nx.is_d_separator(mutilated, {e}, {outcome}, other_evidence):
                    return True
            return False

    # JATE filters — conditioned on jate_relaxed
    if is_jate:
        if not jate_relaxed:
            # JATE-1: no causal relations among treatments (strict only)
            for t1, t2 in itertools.combinations(T, 2):
                if _has_directed_path(dag, t1, t2) or _has_directed_path(dag, t2, t1):
                    return True
        # JATE-2: outcome not ancestor of any treatment (always)
        for t in T:
            if _has_directed_path(dag, outcome, t):
                return True
        return False

    # Fallback for plain ATE-like usage (1 treatment, no evidence)
    return False


# ============================================================
# Effect Computation
# ============================================================

def cate_jate_via_graph_surgery(
    model: DiscreteBayesianNetwork,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
) -> float:
    """
    Compute CATE/JATE via graph surgery: do(treatments) + VariableElimination.

    For CATE: P(Y=1 | do(T=1), E=e) - P(Y=1 | do(T=0), E=e)
    For JATE: P(Y=1 | do(T1=1, T2=1)) - P(Y=1 | do(T1=0, T2=0))

    This is the PRIMARY method. It works for all valid queries, including
    abnormal cases (effect will be ~0.0).
    """
    mutilated = model.do(treatments)
    ve = VariableElimination(mutilated)

    # Build VE evidence: treatment values + conditioning evidence
    ve_evidence_1 = {t: 1 for t in treatments}
    ve_evidence_0 = {t: 0 for t in treatments}
    ve_evidence_1.update(evidence)
    ve_evidence_0.update(evidence)

    phi1 = ve.query([outcome], evidence=ve_evidence_1, show_progress=False)
    phi0 = ve.query([outcome], evidence=ve_evidence_0, show_progress=False)

    return _p_of_one(phi1, outcome) - _p_of_one(phi0, outcome)


def cate_jate_via_causal_query(
    scm: SCMBase,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
) -> float:
    """
    Compute CATE/JATE via pgmpy CausalInference.query with adjustment.

    This is the VERIFICATION method. CausalInference requires string node
    names, so the model is rebuilt with string IDs from scm.cpts.

    For CATE, this method is a cross-check of graph surgery. For JATE, the two
    results must match.

    Raises:
        ValueError: If CausalInference cannot compute the interventional
            distribution (e.g., abnormal cases).
    """
    str_model = _build_string_pgmpy_model(scm)
    str_outcome = str(outcome)

    do_1 = {str(t): 1 for t in treatments}
    do_0 = {str(t): 0 for t in treatments}
    str_evidence = {str(e): v for e, v in evidence.items()} if evidence else None

    ci = CausalInference(str_model)

    phi1 = ci.query([str_outcome], do=do_1, evidence=str_evidence, show_progress=False)
    phi0 = ci.query([str_outcome], do=do_0, evidence=str_evidence, show_progress=False)

    return _p_of_one(phi1, str_outcome) - _p_of_one(phi0, str_outcome)


def compute_cate_jate(
    scm: SCMBase,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    atol: float = 1e-5,
    raise_on_mismatch: bool = False,
    cate_relaxed_mode: Optional[str] = None,
    jate_relaxed: Optional[bool] = None,
) -> CATEJATEResult:
    """
    Compute ground-truth CATE/JATE and cross-verify with two methods.

    Primary: graph surgery (always works, always correct).
    Verification: CausalInference.query (may raise ValueError for abnormal
        cases; is a cross-check for CATE).

    Args:
        raise_on_mismatch: If True, raise RuntimeError when the two methods
            disagree beyond atol. Only applies to JATE queries (for CATE,
            CI may legitimately disagree for normal cases).
        cate_relaxed_mode: Forwarded to is_abnormal_triplets.
        jate_relaxed: Forwarded to is_abnormal_triplets.
    """
    query_type = _determine_query_type(treatments, evidence)
    evidence_nodes = sorted(evidence.keys())

    effect_surgery = cate_jate_via_graph_surgery(
        scm.pgmpy_model, treatments, outcome, evidence
    )

    is_abnormal = is_abnormal_triplets(
        scm.dag, treatments, outcome,
        evidence=evidence_nodes if evidence_nodes else None,
        cate_relaxed_mode=cate_relaxed_mode,
        jate_relaxed=jate_relaxed,
    )

    effect_cq = None
    verification_ok = None

    try:
        effect_cq = cate_jate_via_causal_query(scm, treatments, outcome, evidence)
    except ValueError:
        pass  # CI couldn't compute, so skip verification; `is_abnormal` should only be determined by `is_abnormal_triplets`

    if effect_cq is not None and not math.isnan(effect_cq) and not is_abnormal:
        diff = abs(effect_surgery - effect_cq)
        verification_ok = diff < atol
        if not verification_ok and raise_on_mismatch and query_type == "jate":
            # NOTE: for relaxed JATE queries that allow causal relationships among treatments, `effect_cq` can also be incorrect
            raise RuntimeError(
                f"JATE mismatch for treatments={treatments}, outcome={outcome}: "
                f"surgery={effect_surgery:.12f}, causal_query={effect_cq:.12f}, "
                f"diff={diff:.2e}, atol={atol}"
            )

    return CATEJATEResult(
        treatments=treatments,
        outcome=outcome,
        evidence=evidence,
        query_type=query_type,
        effect_surgery=effect_surgery,
        effect_causal_query=effect_cq,
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

    print("=== Testing CATE/JATE Query Sampling ===\n")

    # ==================================================================
    # Group 1: Effect Computation
    # ==================================================================
    print("--- Group 1: Effect Computation ---\n")

    # Test 1: CATE on chain 0→1→2, treatment=0, outcome=2, evidence={1: 1}
    # CATE = P(2=1|do(0=1), 1=1) - P(2=1|do(0=0), 1=1)
    # After do(0), node 1's CPD is unchanged. With evidence 1=1
    # With evidence={0: 1, 1: 1}: P(2=1|0=1, 1=1) = P(2=1|1=1) = cpt_2[(1,)]
    # With evidence={0: 0, 1: 1}: P(2=1|0=0, 1=1) = P(2=1|1=1) = cpt_2[(1,)]
    # So CATE = cpt_2[(1,)] - cpt_2[(1,)] = 0 when conditioning on the mediator!
    # This is correct — conditioning on the mediator blocks the causal effect.
    print("Test 1: CATE on chain 0→1→2, conditioning on mediator 1=1")
    chain_dag = nx.DiGraph()
    chain_dag.add_edges_from([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)

    chain_cate_result = compute_cate_jate(chain_scm, [0], 2, {1: 1})
    assert chain_cate_result.query_type == "cate"
    # Conditioning on mediator blocks effect → CATE should be 0
    assert_close(chain_cate_result.effect_surgery, 0.0, atol=1e-10,
                 msg=f"CATE should be 0 when conditioning on mediator, got {chain_cate_result.effect_surgery}")
    print(f"   CATE = {chain_cate_result.effect_surgery:.10f} (expected 0.0)")
    print("   -> PASSED\n")

    # Test 2: CATE with confounder — 2→0, 2→1, 0→1, treatment=0, outcome=1, evidence={2: 1}
    # CATE = P(1=1|do(0=1), 2=1) - P(1=1|do(0=0), 2=1)
    # After do(0): removes edges into 0 (i.e., 2→0 removed). 2→1 and 0→1 remain.
    # P(1=1|do(0=t), 2=1) = P(1=1|0=t, 2=1) in mutilated graph = cpt_1[(t, 1)]
    # (since parents of 1 are [0, 2], sorted)
    # CATE = cpt_1[(1, 1)] - cpt_1[(0, 1)]
    print("Test 2: CATE with confounder, conditioning on confounder value")
    conf_dag = nx.DiGraph()
    conf_dag.add_edges_from([(2, 0), (2, 1), (0, 1)])
    conf_scm = RLCausalSCM(conf_dag, seed=42)

    conf_cate_result = compute_cate_jate(conf_scm, [0], 1, {2: 1})
    assert conf_cate_result.query_type == "cate"
    assert not conf_cate_result.is_abnormal

    # Manual: parents of 1 are [0, 2] sorted. cpt_1[(1,1)] - cpt_1[(0,1)]
    cpt_1 = conf_scm.get_node_cpt(1)
    expected_cate = cpt_1[(1, 1)] - cpt_1[(0, 1)]
    assert_close(conf_cate_result.effect_surgery, expected_cate, atol=1e-8,
                 msg=f"CATE manual check: {conf_cate_result.effect_surgery} vs {expected_cate}")
    print(f"   CATE = {conf_cate_result.effect_surgery:.10f} (manual = {expected_cate:.10f})")
    print("   -> PASSED\n")

    # Test 3: JATE on fork — 0→2, 1→2, treatment=[0, 1], outcome=2
    # JATE = P(2=1|do(0=1, 1=1)) - P(2=1|do(0=0, 1=0))
    # After do(0, 1): remove edges into 0 and 1 (they're roots, no change).
    # P(2=1|0=t0, 1=t1) = cpt_2[(t0, t1)]  (parents of 2 are [0, 1] sorted)
    # JATE = cpt_2[(1, 1)] - cpt_2[(0, 0)]
    print("Test 3: JATE on fork 0→2, 1→2")
    fork_dag = nx.DiGraph()
    fork_dag.add_edges_from([(0, 2), (1, 2)])
    fork_scm = RLCausalSCM(fork_dag, seed=42)

    fork_jate_result = compute_cate_jate(fork_scm, [0, 1], 2, {})
    assert fork_jate_result.query_type == "jate"
    assert not fork_jate_result.is_abnormal

    cpt_2 = fork_scm.get_node_cpt(2)
    expected_jate = cpt_2[(1, 1)] - cpt_2[(0, 0)]
    assert_close(fork_jate_result.effect_surgery, expected_jate, atol=1e-8,
                 msg=f"JATE manual check: {fork_jate_result.effect_surgery} vs {expected_jate}")
    print(f"   JATE = {fork_jate_result.effect_surgery:.10f} (manual = {expected_jate:.10f})")
    print("   -> PASSED\n")

    # Test 4: JATE with downstream — 0→2, 1→2, 2→3, treatment=[0, 1], outcome=3
    # JATE through mediator 2
    print("Test 4: JATE with downstream 0→2, 1→2, 2→3")
    downstream_dag = nx.DiGraph()
    downstream_dag.add_edges_from([(0, 2), (1, 2), (2, 3)])
    downstream_scm = RLCausalSCM(downstream_dag, seed=42)

    downstream_jate_result = compute_cate_jate(downstream_scm, [0, 1], 3, {})
    assert downstream_jate_result.query_type == "jate"
    assert not downstream_jate_result.is_abnormal
    assert -1.0 <= downstream_jate_result.effect_surgery <= 1.0
    print(f"   JATE = {downstream_jate_result.effect_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 5: Abnormal CATE — no path from treatment to outcome
    # DAG: 0→1, node 2 isolated; treatment=0, outcome=2, evidence={1: 1}
    # is_abnormal_triplets: no path 0→2, so abnormal
    print("Test 5: Abnormal CATE (no path from treatment to outcome)")
    indep_dag = nx.DiGraph()
    indep_dag.add_nodes_from([0, 1, 2])
    indep_dag.add_edge(0, 1)
    indep_scm = RLCausalSCM(indep_dag, seed=42)

    abnormal_cate_result = compute_cate_jate(indep_scm, [0], 2, {1: 0})
    assert abnormal_cate_result.is_abnormal, "Should be abnormal: no path 0→2"
    assert_close(abnormal_cate_result.effect_surgery, 0.0, atol=1e-10,
                 msg=f"CATE should be 0 for abnormal case, got {abnormal_cate_result.effect_surgery}")
    print(f"   is_abnormal = {abnormal_cate_result.is_abnormal}")
    print(f"   CATE = {abnormal_cate_result.effect_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 6: Abnormal JATE — causal relation among treatments
    # DAG: 0→1, 0→2, 1→2; treatment=[0, 1], outcome=2
    # is_abnormal_triplets: path 0→1 among treatments, so abnormal
    print("Test 6: Abnormal JATE (causal relation among treatments)")
    t_chain_dag = nx.DiGraph()
    t_chain_dag.add_edges_from([(0, 1), (0, 2), (1, 2)])
    t_chain_scm = RLCausalSCM(t_chain_dag, seed=42)

    abnormal_jate_result = compute_cate_jate(t_chain_scm, [0, 1], 2, {})
    assert abnormal_jate_result.is_abnormal, "Should be abnormal: path 0→1 among treatments"
    print(f"   is_abnormal = {abnormal_jate_result.is_abnormal}")
    print(f"   JATE = {abnormal_jate_result.effect_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 7: JATE verification_ok should be True for normal fork
    print("Test 7: JATE verification_ok for normal fork")
    assert fork_jate_result.verification_ok is True, \
        f"Expected verification_ok=True for normal JATE, got {fork_jate_result.verification_ok}"
    print(f"   verification_ok = {fork_jate_result.verification_ok}")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: _p_of_one Helper
    # ==================================================================
    print("--- Group 2: _p_of_one Helper ---\n")

    # Test 8: _p_of_one helper
    print("Test 8: _p_of_one helper")
    test_factor = DiscreteFactor([0], [2], [0.3, 0.7])
    assert_close(_p_of_one(test_factor, 0), 0.7, atol=1e-10)
    try:
        _p_of_one(test_factor, 99)
        raise AssertionError("Expected ValueError for wrong scope")
    except ValueError:
        pass
    print("   -> PASSED\n")

    # ==================================================================
    # Group 3: Relaxed Abnormality Modes
    # ==================================================================
    print("--- Group 3: Relaxed Abnormality Modes ---\n")

    # Test 9: JATE relaxed — causal relation among treatments accepted
    # DAG: 0→1, 0→2, 1→2; treatment=[0, 1], outcome=2
    # Strict: abnormal (JATE-1: 0→1 among treatments)
    # Relaxed: normal (JATE-1 dropped)
    print("Test 9: JATE relaxed — treatment-treatment causal relation accepted")
    assert is_abnormal_triplets(
        t_chain_dag, [0, 1], 2, None,
    ), "Strict JATE should be abnormal: path 0→1 among treatments"
    assert not is_abnormal_triplets(
        t_chain_dag, [0, 1], 2, None,
        jate_relaxed=True,
    ), "Relaxed JATE should be normal: JATE-1 dropped"
    # Verify compute_cate_jate also respects jate_relaxed
    relaxed_jate_result = compute_cate_jate(t_chain_scm, [0, 1], 2, {}, jate_relaxed=True)
    assert not relaxed_jate_result.is_abnormal, \
        f"Expected is_abnormal=False for relaxed JATE, got {relaxed_jate_result.is_abnormal}"
    print(f"   strict is_abnormal = True, relaxed is_abnormal = {relaxed_jate_result.is_abnormal}")
    print("   -> PASSED\n")

    # Test 10: JATE relaxed — "at least one reaches" vs "every reaches"
    # DAG: 0→2, 1 (isolated root), node 2; treatment=[0, 1], outcome=2
    # 0 reaches 2 but 1 does not.
    # Strict: abnormal (All-2: not all treatments reach outcome)
    # Relaxed: normal (All-2 dropped, at least one reaches via All-1)
    print("Test 10: JATE relaxed — at least one treatment reaches outcome")
    partial_reach_dag = nx.DiGraph()
    partial_reach_dag.add_nodes_from([0, 1, 2])
    partial_reach_dag.add_edge(0, 2)
    assert is_abnormal_triplets(
        partial_reach_dag, [0, 1], 2, None,
    ), "Strict JATE should be abnormal: not all treatments reach outcome"
    assert not is_abnormal_triplets(
        partial_reach_dag, [0, 1], 2, None,
        jate_relaxed=True,
    ), "Relaxed JATE should be normal: at least one reaches"
    print("   -> PASSED\n")

    # Test 11: JATE relaxed — outcome→treatment still rejected (JATE-2 kept)
    # DAG: 2→0, 2→1; treatment=[0, 1], outcome=2
    # 2→0 means outcome is ancestor of treatment 0 — always abnormal
    print("Test 11: JATE relaxed — outcome→treatment still abnormal")
    outcome_ancestor_dag = nx.DiGraph()
    outcome_ancestor_dag.add_edges_from([(2, 0), (2, 1)])
    assert is_abnormal_triplets(
        outcome_ancestor_dag, [0, 1], 2, None,
        jate_relaxed=True,
    ), "Even relaxed JATE should reject outcome→treatment (JATE-2)"
    print("   -> PASSED\n")

    # Test 12: CATE ancestor_evidence — post-treatment mediator accepted
    # DAG: 0→1→2; treatment=0, outcome=2, evidence={1: 1}
    # Node 1 is post-treatment (0→1) but is ancestor of outcome (1→2).
    # Strict: abnormal (CATE-1: post-treatment evidence)
    # ancestor_evidence: normal (CATE-2 passes: 1 is ancestor of 2)
    print("Test 12: CATE ancestor_evidence — mediator evidence accepted")
    assert is_abnormal_triplets(
        chain_dag, [0], 2, [1],
    ), "Strict CATE should reject post-treatment evidence"
    assert not is_abnormal_triplets(
        chain_dag, [0], 2, [1],
        cate_relaxed_mode="ancestor_evidence",
    ), "ancestor_evidence mode should accept mediator (ancestor of Y)"
    print("   -> PASSED\n")

    # Test 13: CATE ancestor_evidence — non-ancestor evidence rejected
    # DAG: 0→1, 2 (isolated); treatment=0, outcome=1, evidence={2: 0}
    # Node 2 is not ancestor of T(0) or Y(1).
    # Both strict and ancestor_evidence should reject.
    print("Test 13: CATE ancestor_evidence — non-ancestor evidence still rejected")
    non_anc_dag = nx.DiGraph()
    non_anc_dag.add_nodes_from([0, 1, 2])
    non_anc_dag.add_edge(0, 1)
    assert is_abnormal_triplets(
        non_anc_dag, [0], 1, [2],
        cate_relaxed_mode="ancestor_evidence",
    ), "ancestor_evidence should reject evidence not ancestor of T or Y"
    print("   -> PASSED\n")

    # Test 14: CATE non_dsep_evidence — d-connected evidence accepted
    # DAG: 2→0→1; treatment=0, outcome=1, evidence={2: 1}
    # Mutilated graph (remove edges into 0): 0→1 remains, 2 isolated.
    # In mutilated graph, 2 is d-separated from 1 given {} → should reject.
    # Different DAG: 2→0, 2→1, 0→1; treatment=0, outcome=1, evidence={2: 0}
    # Mutilated (remove 2→0): edges are 2→1, 0→1. 2 d-connected to 1 via 2→1.
    print("Test 14: CATE non_dsep_evidence — d-connected evidence accepted")
    dsep_dag1 = nx.DiGraph()
    dsep_dag1.add_edges_from([(2, 0), (2, 1), (0, 1)])
    assert not is_abnormal_triplets(
        dsep_dag1, [0], 1, [2],
        cate_relaxed_mode="non_dsep_evidence",
    ), "non_dsep_evidence should accept d-connected evidence"
    print("   -> PASSED\n")

    # Test 15: CATE non_dsep_evidence — d-separated evidence rejected
    # DAG: 2→0→1; treatment=0, outcome=1, evidence={2: 0}
    # Mutilated graph (remove 2→0): 0→1 remains, 2 isolated.
    # 2 is d-separated from 1 in mutilated graph → abnormal
    print("Test 15: CATE non_dsep_evidence — d-separated evidence rejected")
    dsep_dag2 = nx.DiGraph()
    dsep_dag2.add_edges_from([(2, 0), (0, 1)])
    assert is_abnormal_triplets(
        dsep_dag2, [0], 1, [2],
        cate_relaxed_mode="non_dsep_evidence",
    ), "non_dsep_evidence should reject d-separated evidence"
    print("   -> PASSED\n")

    # Test 16: CausalInference fallback does NOT override is_abnormal
    # DAG: 0→1, node 2 isolated; treatment=0, outcome=2, evidence={1: 0}
    # No path 0→2 → is_abnormal = True from checker
    # is_abnormal comes only from the checker
    print("Test 16: CI fallback does not override is_abnormal")
    ci_dag = nx.DiGraph()
    ci_dag.add_nodes_from([0, 1, 2])
    ci_dag.add_edge(0, 1)
    ci_scm = RLCausalSCM(ci_dag, seed=42)
    ci_result = compute_cate_jate(ci_scm, [0], 2, {1: 0})
    assert ci_result.is_abnormal, "No path 0→2, checker says abnormal"
    # Now test a normal case where CI might fail but checker says normal:
    # DAG: 2→0, 2→1, 0→1; treatment=0, outcome=1, evidence={2: 1}
    # Path 0→1 exists, 2 is ancestor of both T and Y.
    # Strict: 2→0 means post-treatment? No: T=0, path is 2→0 not 0→2. 2 is pre-treatment.
    # Actually: T=0, evidence=2. Is there path T→E? 0→2? No. Path is 2→0. OK good.
    # But 2→0 means evidence (2) has a directed path to treatment (0).
    # is ancestor of T(0)? Yes (2→0). So CATE-2 passes. CATE-1: T→E = 0→2? No path. OK.
    # So this is normal under strict. CI should succeed.
    ci_normal_result = compute_cate_jate(conf_scm, [0], 1, {2: 1})
    assert not ci_normal_result.is_abnormal, "Normal CATE should not be abnormal"
    print(f"   abnormal case: is_abnormal={ci_result.is_abnormal}")
    print(f"   normal case: is_abnormal={ci_normal_result.is_abnormal}")
    print("   -> PASSED\n")

    print("=== All Tests Passed Successfully ===")
