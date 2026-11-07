"""
nie_nde_query_sampling.py

Core Natural Indirect Effect (NIE) and Natural Direct Effect (NDE)
computation for valid (treatment, mediator, outcome) triplets.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict

import networkx as nx

from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class NIENDEResult:
    treatment: int
    mediator: int
    outcome: int
    pm_do0: Dict[int, float]
    pm_do1: Dict[int, float]
    py_values: Dict[str, float]
    nie_value: float
    nde_value: float
    is_abnormal: bool

    def to_json_dict(self) -> Dict:
        return {
            "treatment": self.treatment,
            "mediator": self.mediator,
            "outcome": self.outcome,
            "pm_do0": self.pm_do0,
            "pm_do1": self.pm_do1,
            "py_values": self.py_values,
            "nie_value": self.nie_value,
            "nde_value": self.nde_value,
            "is_abnormal": self.is_abnormal,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "NIENDEResult":
        return cls(
            treatment=data["treatment"],
            mediator=data["mediator"],
            outcome=data["outcome"],
            pm_do0={int(k): v for k, v in data["pm_do0"].items()},
            pm_do1={int(k): v for k, v in data["pm_do1"].items()},
            py_values=data["py_values"],
            nie_value=data["nie_value"],
            nde_value=data["nde_value"],
            is_abnormal=data["is_abnormal"],
        )


def _enumerate_distribution(scm: SCMBase, node: int) -> Dict[int, float]:
    totals = {0: 0.0, 1: 0.0}
    nodes = list(scm.nodes)
    for assignment in itertools.product([0, 1], repeat=len(nodes)):
        assign_map = {n: v for n, v in zip(nodes, assignment)}
        prob = 1.0
        for cur in nodes:
            parents = scm.cpts[cur]["parents"]
            table = scm.cpts[cur]["table"]
            parent_values = tuple(assign_map[parent] for parent in parents)
            p1 = table[parent_values]
            prob *= p1 if assign_map[cur] == 1 else 1.0 - p1
        totals[assign_map[node]] += prob

    total_prob = totals[0] + totals[1]
    if total_prob == 0.0:
        return {0: 0.0, 1: 0.0}
    return {0: totals[0] / total_prob, 1: totals[1] / total_prob}


def _enumerate_outcome_prob(scm: SCMBase, outcome: int) -> float:
    return _enumerate_distribution(scm, outcome).get(1, 0.0)


def _clone_with_do(scm: SCMBase, assignments: Dict[int, int]) -> SCMBase:
    dag = scm.dag.copy()
    cpts = {}
    for node in scm.nodes:
        cpts[node] = {
            "parents": list(scm.cpts[node]["parents"]),
            "table": dict(scm.cpts[node]["table"]),
        }

    for node, value in assignments.items():
        dag.remove_edges_from([(parent, node) for parent in dag.predecessors(node)])
        cpts[node] = {"parents": [], "table": {(): float(value)}}

    decimal_places = getattr(scm, "decimal_places", 4)
    return RLCausalSCM.from_cpts(dag, cpts, decimal_places=decimal_places)


def _p_y_given_do(
    scm: SCMBase,
    treatment: int,
    mediator: int,
    outcome: int,
    treatment_value: int,
    mediator_value: int,
) -> float:
    intervened = _clone_with_do(
        scm,
        {treatment: treatment_value, mediator: mediator_value},
    )
    return _enumerate_outcome_prob(intervened, outcome)


def _p_m_given_do(
    scm: SCMBase,
    treatment: int,
    mediator: int,
    treatment_value: int,
) -> Dict[int, float]:
    intervened = _clone_with_do(scm, {treatment: treatment_value})
    return _enumerate_distribution(intervened, mediator)


def has_chain(dag: nx.DiGraph, treatment: int, mediator: int, outcome: int) -> bool:
    return nx.has_path(dag, treatment, mediator) and nx.has_path(dag, mediator, outcome)


def _has_path_excluding_node(
    dag: nx.DiGraph,
    source: int,
    target: int,
    excluded_node: int,
) -> bool:
    if source == excluded_node or target == excluded_node:
        return False
    if not dag.has_node(source) or not dag.has_node(target):
        return False

    pruned = dag.copy()
    if pruned.has_node(excluded_node):
        pruned.remove_node(excluded_node)
    return nx.has_path(pruned, source, target)


def has_treatment_induced_mediator_outcome_confounder(
    dag: nx.DiGraph,
    treatment: int,
    mediator: int,
    outcome: int,
) -> bool:
    for node in dag.nodes():
        if node in (treatment, mediator, outcome):
            continue
        if not nx.has_path(dag, treatment, node):
            continue
        if not nx.has_path(dag, node, mediator):
            continue
        if _has_path_excluding_node(dag, node, outcome, mediator):
            return True
    return False


def is_invalid_condition(
    dag: nx.DiGraph,
    treatment: int,
    mediator: int,
    outcome: int,
) -> bool:
    return not (
        len({treatment, mediator, outcome}) == 3
        and has_chain(dag, treatment, mediator, outcome)
        and not has_treatment_induced_mediator_outcome_confounder(
            dag, treatment, mediator, outcome
        )
    )


def is_abnormal(
    dag: nx.DiGraph,
    treatment: int,
    mediator: int,
    outcome: int,
) -> bool:
    return not _has_path_excluding_node(dag, treatment, outcome, mediator)


def compute_nie_nde(
    scm: SCMBase,
    treatment: int,
    mediator: int,
    outcome: int,
) -> NIENDEResult:
    pm_do0 = _p_m_given_do(scm, treatment, mediator, 0)
    pm_do1 = _p_m_given_do(scm, treatment, mediator, 1)

    py_values: Dict[str, float] = {}
    for treatment_value, mediator_value in itertools.product([0, 1], repeat=2):
        key = f"t{treatment_value}_m{mediator_value}"
        py_values[key] = _p_y_given_do(
            scm,
            treatment,
            mediator,
            outcome,
            treatment_value,
            mediator_value,
        )

    def _expected_y(treatment_value: int, mediator_distribution: Dict[int, float]) -> float:
        total = 0.0
        for mediator_value, prob in mediator_distribution.items():
            total += prob * py_values[f"t{treatment_value}_m{mediator_value}"]
        return total

    e_y1_m1 = _expected_y(1, pm_do1)
    e_y1_m0 = _expected_y(1, pm_do0)
    e_y0_m0 = _expected_y(0, pm_do0)

    nie_value = e_y1_m1 - e_y1_m0
    nde_value = e_y1_m0 - e_y0_m0

    return NIENDEResult(
        treatment=treatment,
        mediator=mediator,
        outcome=outcome,
        pm_do0=pm_do0,
        pm_do1=pm_do1,
        py_values=py_values,
        nie_value=nie_value,
        nde_value=nde_value,
        is_abnormal=is_abnormal(scm.dag, treatment, mediator, outcome),
    )


if __name__ == "__main__":
    def assert_close(a, b, *, atol=1e-9, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing NIE/NDE Query Sampling ===\n")

    print("Test 1: JSON round-trip")
    result = NIENDEResult(
        treatment=0,
        mediator=1,
        outcome=2,
        pm_do0={0: 0.3, 1: 0.7},
        pm_do1={0: 0.2, 1: 0.8},
        py_values={"t0_m0": 0.1, "t0_m1": 0.2, "t1_m0": 0.4, "t1_m1": 0.9},
        nie_value=0.15,
        nde_value=0.25,
        is_abnormal=False,
    )
    restored = NIENDEResult.from_json_dict(result.to_json_dict())
    assert restored == result
    print("   -> PASSED\n")

    print("Test 2: Simple chain with pure mediation")
    dag = nx.DiGraph([(0, 1), (1, 2)])
    cpts = {
        0: {"parents": [], "table": {(): 0.5}},
        1: {"parents": [0], "table": {(0,): 0.0, (1,): 1.0}},
        2: {"parents": [1], "table": {(0,): 0.0, (1,): 1.0}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=4)
    result = compute_nie_nde(scm, 0, 1, 2)
    assert_close(result.pm_do0[1], 0.0)
    assert_close(result.pm_do1[1], 1.0)
    assert_close(result.py_values["t1_m1"], 1.0)
    assert_close(result.py_values["t1_m0"], 0.0)
    assert_close(result.py_values["t0_m0"], 0.0)
    assert_close(result.nie_value, 1.0)
    assert_close(result.nde_value, 0.0)
    assert result.is_abnormal is True
    assert is_invalid_condition(dag, 0, 1, 2) is False
    assert is_abnormal(dag, 0, 1, 2) is True
    print("   -> PASSED\n")

    print("Test 3: Direct-only effect gives zero NIE on a valid triplet")
    dag = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    cpts = {
        0: {"parents": [], "table": {(): 0.5}},
        1: {"parents": [0], "table": {(0,): 0.4, (1,): 0.4}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.1, (1, 0): 0.9, (1, 1): 0.9}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=4)
    result = compute_nie_nde(scm, 0, 1, 2)
    assert_close(result.nie_value, 0.0)
    assert_close(result.nde_value, 0.8)
    assert result.is_abnormal is False
    assert is_invalid_condition(dag, 0, 1, 2) is False
    print("   -> PASSED\n")

    print("Test 4: Invalid condition detection")
    dag = nx.DiGraph([(0, 2), (1, 2)])
    assert is_invalid_condition(dag, 0, 1, 2) is True
    assert is_abnormal(dag, 0, 1, 2) is False
    print("   -> PASSED\n")

    print("Test 5: Treatment-induced mediator-outcome confounder is invalid")
    dag = nx.DiGraph([(0, 3), (3, 1), (3, 2), (1, 2)])
    assert is_invalid_condition(dag, 0, 1, 2) is True
    assert is_abnormal(dag, 0, 1, 2) is False
    print("   -> PASSED\n")

    print("=== All NIE/NDE query tests passed ===")
