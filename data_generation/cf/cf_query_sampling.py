"""
cf_query_sampling.py

Counterfactual query sampling + inference helpers mirroring the ATE module.

Implements a twin-network construction using independent selectors shared
across factual and counterfactual worlds.
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from data_generation.scm_mechanism_sampling import SCMBase


@dataclass
class CounterfactualQueryResult:
    """Inference-ready payload for P(Y_x | evidence)."""

    treatments: Dict[int, int]
    outcome: int
    evidence: Dict[int, int]
    factor_order: List[int]
    factor_values: Dict[Tuple[int, ...], float]
    normalization_ok: bool
    normalization_error: float

    def prob_all_ones(self) -> float:
        """Return P(all query vars = 1 | ...)."""
        if not self.factor_order:
            raise ValueError('Counterfactual query has no target variables')
        key = tuple(1 for _ in self.factor_order)
        return self.factor_values.get(key, 0.0)

    def to_json_dict(self) -> Dict:
        """Serialize to a JSON-friendly dict."""
        factor_values = {repr(k): v for k, v in self.factor_values.items()}
        return {
            'treatments': {str(k): v for k, v in self.treatments.items()},
            'outcome': self.outcome,
            'evidence': {str(k): v for k, v in self.evidence.items()},
            'factor_order': list(self.factor_order),
            'factor_values': factor_values,
            'normalization_ok': self.normalization_ok,
            'normalization_error': self.normalization_error,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> 'CounterfactualQueryResult':
        """Deserialize from a dict produced by to_json_dict()."""
        factor_values = {}
        for key, val in data['factor_values'].items():
            assignment = ast.literal_eval(key)
            if isinstance(assignment, int):
                assignment = (assignment,)
            factor_values[tuple(int(x) for x in assignment)] = val

        return cls(
            treatments={int(k): v for k, v in data['treatments'].items()},
            outcome=int(data['outcome']),
            evidence={int(k): v for k, v in data['evidence'].items()},
            factor_order=list(data['factor_order']),
            factor_values=factor_values,
            normalization_ok=bool(data['normalization_ok']),
            normalization_error=float(data['normalization_error']),
        )


def is_abnormal_counterfactual(
    dag: nx.DiGraph,
    treatments: Dict[int, int],
    outcome: int,
) -> bool:
    """Return True when no treatment node has a directed path to the outcome."""
    treatment_nodes = sorted(treatments.keys())
    if not treatment_nodes:
        raise ValueError('Counterfactual queries require at least one treatment node')
    return not any(nx.has_path(dag, treatment_node, outcome) for treatment_node in treatment_nodes)


def has_naive_conditioning_counterfactual(
    dag: nx.DiGraph,
    treatments: Dict[int, int],
    evidence: Dict[int, int],
) -> bool:
    """Return True when all evidence nodes are pre-treatment."""
    treatment_nodes = sorted(treatments.keys())
    if not treatment_nodes:
        raise ValueError('Counterfactual queries require at least one treatment node')
    if not evidence:
        return False

    return all(
        not any(nx.has_path(dag, treatment_node, evidence_node) for treatment_node in treatment_nodes)
        for evidence_node in evidence
    )


def _child_card(cpd: TabularCPD) -> int:
    if hasattr(cpd, 'variable_card'):
        return int(cpd.variable_card)
    return int(np.asarray(cpd.cardinality).flatten()[0])


def _evidence_list(cpd: TabularCPD) -> List[int]:
    if hasattr(cpd, 'get_evidence'):
        return list(cpd.get_evidence())
    variables = list(getattr(cpd, 'variables', []))
    return variables[1:]


def _cpd_prob_child_eq_1(cpd: TabularCPD, parent_assign: Dict[int, int]) -> float:
    """Return P(child=1 | parent_assign) using pgmpy accessors when possible."""
    child = cpd.variable
    if hasattr(cpd, 'reduce'):
        reduced = cpd
        for parent, value in parent_assign.items():
            reduced = reduced.reduce([(parent, int(value))], inplace=False)
        vals = np.asarray(reduced.values).reshape(-1)
        if len(vals) < 2:
            raise ValueError(f'CPD for {child} is not binary')
        return float(vals[1])

    parents = _evidence_list(cpd)
    if not parents:
        vals = np.asarray(cpd.values).reshape(-1)
        return float(vals[1])

    if hasattr(cpd, 'get_evidence_card'):
        cards = list(map(int, cpd.get_evidence_card()))
    elif hasattr(cpd, 'evidence_card'):
        cards = list(map(int, cpd.evidence_card))
    else:
        flat = list(map(int, np.asarray(cpd.cardinality).flatten()))
        cards = flat[1 : 1 + len(parents)]

    idx = 0
    for parent, card in zip(parents, cards):
        idx = idx * card + int(parent_assign[parent])

    table = np.asarray(cpd.values).reshape(_child_card(cpd), -1)
    return float(table[1, idx])


def _point_mass(variable, value):
    vals = [[1.0], [0.0]] if int(value) == 0 else [[0.0], [1.0]]
    return TabularCPD(variable, 2, vals, state_names={variable: [0, 1]})


def _deterministic_binary_cpd(variable, evidence, func):
    evidence = list(evidence)
    evidence_card = [2] * len(evidence)

    cols = []
    for ev_states in itertools.product([0, 1], repeat=len(evidence)):
        out = int(func(ev_states))
        col = [0.0, 0.0]
        col[out] = 1.0
        cols.append(col)

    values = np.asarray(cols).T.tolist()
    state_names = {variable: [0, 1], **{ev: [0, 1] for ev in evidence}}

    return TabularCPD(
        variable=variable,
        variable_card=2,
        values=values,
        evidence=evidence,
        evidence_card=evidence_card,
        state_names=state_names,
    )


def _selector_name(node: int, parents: List[int], pa_vals: Tuple[int, ...]) -> str:
    if not parents:
        return f'SEL__{node}__ROOT'
    tag = '__'.join(f'{p}_{v}' for p, v in zip(parents, pa_vals))
    return f'SEL__{node}__{tag}'


def build_independent_selectors_twin(
    scm: SCMBase,
    treatments: Dict[int, int],
) -> DiscreteBayesianNetwork:
    """Build the twin network that shares selector variables across worlds."""
    base_model = scm.pgmpy_model
    cpds = {cpd.variable: cpd for cpd in base_model.get_cpds()}
    nodes = list(base_model.nodes())

    twin = DiscreteBayesianNetwork()
    twin.add_nodes_from(nodes)
    twin.add_nodes_from([f'{node}_cf' for node in nodes])

    selector_info = {}

    for node in nodes:
        cpd = cpds.get(node)
        if cpd is None:
            raise ValueError(f'Missing CPD for node {node}')
        parents = _evidence_list(cpd)
        if len(parents) > 2:
            raise ValueError(
                f'Node {node} has {len(parents)} parents; max indegree 2 is required.'
            )
        pa_tuples = [tuple()] if not parents else list(
            itertools.product([0, 1], repeat=len(parents))
        )
        selectors = []
        tuple_to_index = {cfg: idx for idx, cfg in enumerate(pa_tuples)}

        for pa_vals in pa_tuples:
            selector = _selector_name(node, parents, pa_vals)
            selectors.append(selector)
            twin.add_node(selector)

            parent_assign = {p: v for p, v in zip(parents, pa_vals)}
            p1 = _cpd_prob_child_eq_1(cpd, parent_assign)
            twin.add_cpds(
                TabularCPD(
                    variable=selector,
                    variable_card=2,
                    values=[[1.0 - p1], [p1]],
                    state_names={selector: [0, 1]},
                )
            )

        selector_info[node] = {
            'parents': parents,
            'selectors': selectors,
            'tuple_to_index': tuple_to_index,
        }

    def add_structural_cpds(target, parents_world, selectors, tuple_to_index):
        for parent in parents_world:
            twin.add_edge(parent, target)
        for selector in selectors:
            twin.add_edge(selector, target)

        evidence = list(parents_world) + list(selectors)
        parent_len = len(parents_world)

        def func(ev_states):
            pa_vals = tuple(ev_states[:parent_len]) if parent_len else tuple()
            sel_idx = tuple_to_index[pa_vals]
            return int(ev_states[parent_len + sel_idx])

        twin.add_cpds(_deterministic_binary_cpd(target, evidence, func))

    for node in nodes:
        info = selector_info[node]
        add_structural_cpds(
            node,
            info['parents'],
            info['selectors'],
            info['tuple_to_index'],
        )

    for node in nodes:
        node_cf = f'{node}_cf'
        if node in treatments:
            twin.add_cpds(_point_mass(node_cf, treatments[node]))
            continue
        info = selector_info[node]
        parent_cf = [f'{parent}_cf' for parent in info['parents']]
        add_structural_cpds(
            node_cf,
            parent_cf,
            info['selectors'],
            info['tuple_to_index'],
        )

    if not twin.check_model():
        raise ValueError('Counterfactual twin network validation failed')
    return twin


def counterfactual_query_factor(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
) -> DiscreteFactor:
    """Return the joint factor over query variables in the CF world."""
    
    twin = build_independent_selectors_twin(
        scm,
        treatments=treatments,
    )
    ve = VariableElimination(twin)
    query_cf_vars = [f'{outcome}_cf']
    return ve.query(
        variables=query_cf_vars,
        evidence=evidence,
        joint=True,
        show_progress=False,
    )


def compute_counterfactual_result(
    scm: SCMBase,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    atol: float = 1e-5,
) -> CounterfactualQueryResult:
    """Primary entry point: compute the joint factor over query vars."""
    factor = counterfactual_query_factor(
        scm,
        treatments=treatments,
        outcome=outcome,
        evidence=evidence,
    )

    ordered_vars = [outcome]
    cf_names = [f'{var}_cf' for var in ordered_vars]
    factor_values = {}
    total = 0.0

    for assignment in itertools.product([0, 1], repeat=len(ordered_vars)):
        assignment_cf = {cf_name: val for cf_name, val in zip(cf_names, assignment)}
        prob = float(factor.get_value(**assignment_cf))
        factor_values[assignment] = prob
        total += prob

    normalization_error = abs(total - 1.0)
    return CounterfactualQueryResult(
        treatments=dict(treatments),
        outcome=int(outcome),
        evidence=dict(evidence),
        factor_order=ordered_vars,
        factor_values=factor_values,
        normalization_ok=normalization_error <= atol,
        normalization_error=normalization_error,
    )


# ============================================================
if __name__ == "__main__":
    import math

    import networkx as nx

    from data_generation.scm_mechanism_sampling import RLCausalSCM

    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing Counterfactual Query Sampling ===\n")

    # ==================================================================
    # Group 1: Serialization and validation helpers
    # ==================================================================
    print("--- Group 1: Serialization and validation helpers ---\n")

    print("Test 1: CounterfactualQueryResult JSON round-trip")
    result = CounterfactualQueryResult(
        treatments={0: 1},
        outcome=1,
        evidence={3: 0},
        factor_order=[1],
        factor_values={(0,): 0.6, (1,): 0.4},
        normalization_ok=True,
        normalization_error=0.0,
    )
    result_rt = CounterfactualQueryResult.from_json_dict(result.to_json_dict())
    assert result_rt == result
    assert_close(result_rt.prob_all_ones(), 0.4, atol=1e-12)
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: Single-outcome counterfactual inference
    # ==================================================================
    print("--- Group 2: Single-outcome counterfactual inference ---\n")

    print("Test 3: Root intervention with factual evidence on the treatment")
    chain_dag = nx.DiGraph()
    chain_dag.add_edges_from([(0, 1)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)

    chain_result = compute_counterfactual_result(
        chain_scm,
        treatments={0: 1},
        outcome=1,
        evidence={0: 0},
    )
    cpt_1 = chain_scm.get_node_cpt(1)
    expected_p = cpt_1[(1,)]
    assert chain_result.normalization_ok, f"Expected normalized factor, got error={chain_result.normalization_error}"
    assert chain_result.factor_order == [1]
    assert_close(
        chain_result.prob_all_ones(),
        expected_p,
        atol=1e-8,
        msg=f"Manual CF check failed: {chain_result.prob_all_ones()} vs {expected_p}",
    )
    assert_close(sum(chain_result.factor_values.values()), 1.0, atol=1e-8)
    print(f"   P(1_cf=1 | 0=0) = {chain_result.prob_all_ones():.10f} (manual = {expected_p:.10f})")
    print("   -> PASSED\n")

    print("Test 4: Descendant counterfactual on 0→1→2 with factual treatment evidence")
    long_chain_dag = nx.DiGraph()
    long_chain_dag.add_edges_from([(0, 1), (1, 2)])
    long_chain_scm = RLCausalSCM(long_chain_dag, seed=42)

    long_chain_result = compute_counterfactual_result(
        long_chain_scm,
        treatments={0: 1},
        outcome=2,
        evidence={0: 0},
    )
    cpt_1 = long_chain_scm.get_node_cpt(1)
    cpt_2 = long_chain_scm.get_node_cpt(2)
    expected_descendant_p = cpt_1[(1,)] * cpt_2[(1,)] + (1.0 - cpt_1[(1,)]) * cpt_2[(0,)]
    assert long_chain_result.normalization_ok
    assert_close(
        long_chain_result.prob_all_ones(),
        expected_descendant_p,
        atol=1e-8,
        msg=(
            f"Descendant CF check failed: {long_chain_result.prob_all_ones()} "
            f"vs {expected_descendant_p}"
        ),
    )
    print(
        f"   P(2_cf=1 | 0=0) = {long_chain_result.prob_all_ones():.10f} "
        f"(manual = {expected_descendant_p:.10f})"
    )
    print("   -> PASSED\n")

    # ==================================================================
    # Group 3: Single-outcome factor validation
    # ==================================================================
    print("--- Group 3: Single-outcome factor validation ---\n")

    print("Test 5: Single counterfactual outcome factor is normalized")
    fork_dag = nx.DiGraph()
    fork_dag.add_edges_from([(0, 1), (0, 2)])
    fork_scm = RLCausalSCM(fork_dag, seed=42)

    fork_result = compute_counterfactual_result(
        fork_scm,
        treatments={0: 1},
        outcome=1,
        evidence={0: 0},
    )
    cpt_left = fork_scm.get_node_cpt(1)
    expected_prob = cpt_left[(1,)]
    assert fork_result.normalization_ok
    assert fork_result.factor_order == [1]
    assert len(fork_result.factor_values) == 2
    assert_close(sum(fork_result.factor_values.values()), 1.0, atol=1e-8)
    assert_close(
        fork_result.prob_all_ones(),
        expected_prob,
        atol=1e-8,
        msg=f"Single-outcome CF check failed: {fork_result.prob_all_ones()} vs {expected_prob}",
    )
    print(
        f"   P(1_cf=1 | 0=0) = {fork_result.prob_all_ones():.10f} "
        f"(manual = {expected_prob:.10f})"
    )
    print("   -> PASSED\n")


    print("Test 6: is_abnormal_counterfactual detects disconnected treatment and outcome")
    disconnected_dag = nx.DiGraph()
    disconnected_dag.add_nodes_from([0, 1, 2])
    disconnected_dag.add_edge(0, 1)
    assert is_abnormal_counterfactual(disconnected_dag, {0: 1}, 2)
    assert not is_abnormal_counterfactual(chain_dag, {0: 1}, 1)
    print("   -> PASSED\n")

    print("Test 7: has_naive_conditioning_counterfactual flags all-pre-treatment evidence")
    naive_dag = nx.DiGraph()
    naive_dag.add_edges_from([(0, 1), (1, 2), (3, 2)])
    assert not has_naive_conditioning_counterfactual(naive_dag, {0: 1}, {1: 1})
    assert has_naive_conditioning_counterfactual(naive_dag, {0: 1}, {3: 0})
    assert not has_naive_conditioning_counterfactual(naive_dag, {0: 1}, {1: 1, 3: 0})
    print("   -> PASSED\n")

    print("All counterfactual query sampling tests passed.")
