"""
identification_query_sampling.py

Construct identification queries over partially observed DAGs.

Each query fixes a treatment X and outcome Y, then asks whether or under which
observed-variable sets P(Y | do(X)) is identifiable. Unobserved DAG nodes are
treated as latent for identification purposes. y0 is used for the primary
ID-algorithm path after converting hidden nodes into a mixed graph.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import networkx as nx
from pgmpy.base import DAG
from pgmpy.identification import Adjustment, Frontdoor
from y0.algorithm.identify import Identification, Query, Unidentifiable, identify
from y0.dsl import P, Variable
from y0.graph import NxMixedGraph


OPTION_LABELS = ["A", "B", "C", "D"]
QUERY_FORMATS = ["binary", "multi_choice", "multi_answer"]


@dataclass
class OptionIdentificationResult:
    """Identification outcome for one observed-variable-set option."""

    observed_variables: List[int]
    identifiable: bool
    method: Optional[str]
    witness: List[str]

    def to_json_dict(self) -> Dict:
        return {
            "observed_variables": self.observed_variables,
            "identifiable": self.identifiable,
            "method": self.method,
            "witness": self.witness,
        }


@dataclass
class IdentificationResult:
    """Ground-truth identification result for one identification query item."""

    query_format: str
    treatment: int
    outcome: int
    option_results: List[OptionIdentificationResult]
    correct_options: str
    is_abnormal: bool

    def to_json_dict(self) -> Dict:
        return {
            "query_format": self.query_format,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "option_results": [option.to_json_dict() for option in self.option_results],
            "correct_options": self.correct_options,
            "is_abnormal": self.is_abnormal,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict) -> "IdentificationResult":
        return cls(
            query_format=payload.get("query_format", "multi_answer"),
            treatment=payload["treatment"],
            outcome=payload["outcome"],
            option_results=[
                OptionIdentificationResult(
                    observed_variables=option["observed_variables"],
                    identifiable=option["identifiable"],
                    method=option["method"],
                    witness=option["witness"],
                )
                for option in payload["option_results"]
            ],
            correct_options=payload["correct_options"],
            is_abnormal=payload["is_abnormal"],
        )


def _node_label(node: int) -> str:
    return f"X{node}"


def is_abnormal_pair(dag: nx.DiGraph, treatment: int, outcome: int) -> bool:
    """Treat no directed path from treatment to outcome as structurally abnormal."""

    return not nx.has_path(dag, treatment, outcome)


def _normalize_role_variables(variables) -> List[str]:
    """Normalize pgmpy role payloads to a sorted list of variable names."""

    if variables is None:
        return []
    if isinstance(variables, str):
        return [variables]
    if isinstance(variables, (set, tuple, list)):
        return sorted(str(item) for item in variables)
    raise TypeError(f"Unsupported role payload type: {type(variables)!r}")


def _build_identification_dag(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    observed_variables: Sequence[int],
) -> DAG:
    observed_set = set(observed_variables)
    node_labels = sorted(_node_label(node) for node in dag.nodes())
    latent_labels = sorted(
        _node_label(node)
        for node in dag.nodes()
        if node not in observed_set and dag.degree(node) > 0
    )
    edges = [(_node_label(src), _node_label(dst)) for src, dst in dag.edges()]

    id_dag = DAG()
    id_dag.add_nodes_from(node_labels)
    id_dag.add_edges_from(edges)
    id_dag.latents = set(latent_labels)
    id_dag.exposures = {_node_label(treatment)}
    id_dag.outcomes = {_node_label(outcome)}
    return id_dag


def _node_variable(node: int) -> Variable:
    return Variable(_node_label(node))


def _edge_points_into(dag: nx.DiGraph, src: int, dst: int) -> bool:
    """Return True iff the edge between src and dst has an arrowhead into dst."""

    return dag.has_edge(src, dst)


def _has_hidden_directed_path(
    dag: nx.DiGraph,
    source: int,
    target: int,
    observed_set: set[int],
) -> bool:
    """Check for a directed path source -> ... -> target through hidden intermediates only."""

    stack = [source]
    visited = {source}
    while stack:
        current = stack.pop()
        for child in dag.successors(current):
            if child == target:
                return True
            if child in observed_set:
                continue
            if child in visited:
                continue
            visited.add(child)
            stack.append(child)
    return False


def _is_internal_noncollider(dag: nx.DiGraph, left: int, middle: int, right: int) -> bool:
    """Return True iff middle is a non-collider on the path left-middle-right."""

    left_into_middle = _edge_points_into(dag, left, middle)
    right_into_middle = _edge_points_into(dag, right, middle)
    return not (left_into_middle and right_into_middle)


def _has_hidden_confounding_path(
    dag: nx.DiGraph,
    left: int,
    right: int,
    observed_set: set[int],
) -> bool:
    """Check for a latent-confounding path that induces left <-> right in the projection."""

    skeleton = dag.to_undirected()
    for path in nx.all_simple_paths(skeleton, left, right):
        if len(path) < 3:
            continue

        internal_nodes = path[1:-1]
        if any(node in observed_set for node in internal_nodes):
            continue

        if not _edge_points_into(dag, path[1], path[0]):
            continue
        if not _edge_points_into(dag, path[-2], path[-1]):
            continue

        if all(
            _is_internal_noncollider(dag, prev_node, node, next_node)
            for prev_node, node, next_node in zip(path, path[1:], path[2:])
        ):
            return True

    return False


def _build_y0_identification_graph(
    dag: nx.DiGraph,
    observed_variables: Sequence[int],
) -> NxMixedGraph:
    observed_set = set(observed_variables)
    observed_nodes = sorted(observed_set)
    y0_nodes = [_node_variable(node) for node in observed_nodes]

    directed_edges = []
    for src in observed_nodes:
        for dst in observed_nodes:
            if src == dst:
                continue
            if _has_hidden_directed_path(dag, src, dst, observed_set):
                directed_edges.append((_node_variable(src), _node_variable(dst)))

    y0_undirected_edges = []
    for left_idx, left in enumerate(observed_nodes):
        for right in observed_nodes[left_idx + 1:]:
            if _has_hidden_confounding_path(dag, left, right, observed_set):
                y0_undirected_edges.append(
                    (_node_variable(left), _node_variable(right))
                )

    return NxMixedGraph.from_edges(
        nodes=y0_nodes,
        directed=directed_edges,
        undirected=y0_undirected_edges,
    )


def identify_observed_set_y0(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    observed_variables: Sequence[int],
) -> OptionIdentificationResult:
    """Determine identifiability using y0's ID algorithm on a latent-projected graph."""

    observed_variables = sorted(observed_variables)
    y0_graph = _build_y0_identification_graph(dag, observed_variables)
    treatment_var = _node_variable(treatment)
    outcome_var = _node_variable(outcome)
    observed_estimand = P(*( _node_variable(node) for node in observed_variables ))

    try:
        identified_estimand = identify(
            Identification(
                query=Query(outcomes={outcome_var}, treatments={treatment_var}),
                estimand=observed_estimand,
                graph=y0_graph,
            )
        )
    except Unidentifiable:
        return OptionIdentificationResult(
            observed_variables=observed_variables,
            identifiable=False,
            method=None,
            witness=[],
        )

    witness_lines = [
        line.strip()
        for line in str(identified_estimand).splitlines()
        if line.strip()
    ]
    return OptionIdentificationResult(
        observed_variables=observed_variables,
        identifiable=True,
        method="y0-id",
        witness=witness_lines,
    )


def identify_observed_set_pgmpy(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    observed_variables: Sequence[int],
) -> OptionIdentificationResult:
    """Determine whether P(Y | do(X)) is identifiable under one observed set."""

    id_dag = _build_identification_dag(
        dag,
        treatment,
        outcome,
        observed_variables,
    )

    identified_adjustment, adjustment_ok = Adjustment(variant="all").identify(id_dag)
    if adjustment_ok:
        if isinstance(identified_adjustment, list):
            witness = []
            if identified_adjustment:
                first_adjustment = identified_adjustment[0]
                if hasattr(first_adjustment, "get_role"):
                    witness = _normalize_role_variables(
                        first_adjustment.get_role("adjustment")
                    )
        elif hasattr(identified_adjustment, "get_role"):
            witness = _normalize_role_variables(
                identified_adjustment.get_role("adjustment")
            )
        else:
            witness = _normalize_role_variables(identified_adjustment)

        return OptionIdentificationResult(
            observed_variables=sorted(observed_variables),
            identifiable=True,
            method="adjustment",
            witness=witness,
        )

    identified_frontdoor, frontdoor_ok = Frontdoor().identify(id_dag)
    if frontdoor_ok:
        return OptionIdentificationResult(
            observed_variables=sorted(observed_variables),
            identifiable=True,
            method="frontdoor",
            witness=_normalize_role_variables(
                identified_frontdoor.get_role("frontdoor")
            ),
        )

    return OptionIdentificationResult(
        observed_variables=sorted(observed_variables),
        identifiable=False,
        method=None,
        witness=[],
    )


def enumerate_option_candidates(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    identify_observed_set_fn=identify_observed_set_y0,
) -> List[OptionIdentificationResult]:
    """Enumerate identification results for every observed set containing X and Y."""

    other_nodes = sorted(
        node for node in dag.nodes() if node not in {treatment, outcome}
    )
    candidates: List[OptionIdentificationResult] = []

    for subset_size in range(len(other_nodes) + 1):
        for subset in itertools.combinations(other_nodes, subset_size):
            observed_variables = sorted([treatment, outcome, *subset])
            candidates.append(
                identify_observed_set_fn(
                    dag,
                    treatment,
                    outcome,
                    observed_variables,
                )
            )

    return candidates


def _format_multi_choice_correct_option(options: List[OptionIdentificationResult]) -> str:
    if len(options) != 2:
        raise ValueError("multi_choice requires exactly 2 observed-set options")

    first_identifiable = options[0].identifiable
    second_identifiable = options[1].identifiable
    if not first_identifiable and not second_identifiable:
        return "A"
    if first_identifiable and not second_identifiable:
        return "B"
    if not first_identifiable and second_identifiable:
        return "C"
    return "D"


def _format_correct_options(options: List[OptionIdentificationResult]) -> str:
    labels = [
        label
        for label, option in zip(OPTION_LABELS, options)
        if option.identifiable
    ]
    if not labels:
        return ""
    return ",".join(labels)


def build_identification_binary(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    target_answer_label: Optional[str] = None,
    identify_observed_set_fn=identify_observed_set_y0,
) -> Optional[IdentificationResult]:
    candidates = enumerate_option_candidates(
        dag,
        treatment,
        outcome,
        identify_observed_set_fn=identify_observed_set_fn,
    )
    if target_answer_label is not None:
        if target_answer_label not in {"Yes", "No"}:
            raise ValueError(f"Unknown binary target_answer_label: {target_answer_label}")
        target_identifiable = target_answer_label == "Yes"
        candidates = [option for option in candidates if option.identifiable == target_identifiable]
    if not candidates:
        return None

    sampled_option = option_order_rng.choice(candidates)
    return IdentificationResult(
        query_format="binary",
        treatment=treatment,
        outcome=outcome,
        option_results=[sampled_option],
        correct_options="Yes" if sampled_option.identifiable else "No",
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_identification_multi_choice(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    target_answer_label: Optional[str] = None,
    identify_observed_set_fn=identify_observed_set_y0,
) -> Optional[IdentificationResult]:
    candidates = enumerate_option_candidates(
        dag,
        treatment,
        outcome,
        identify_observed_set_fn=identify_observed_set_fn,
    )
    if len(candidates) < 2:
        return None

    candidate_pairs = [
        list(pair)
        for pair in itertools.combinations(candidates, 2)
        if target_answer_label is None or _format_multi_choice_correct_option(list(pair)) == target_answer_label
    ]
    if not candidate_pairs:
        return None

    sampled_options = option_order_rng.choice(candidate_pairs)
    return IdentificationResult(
        query_format="multi_choice",
        treatment=treatment,
        outcome=outcome,
        option_results=sampled_options,
        correct_options=_format_multi_choice_correct_option(sampled_options),
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_identification_multi_answer(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    identify_observed_set_fn=identify_observed_set_y0,
) -> Optional[IdentificationResult]:
    candidates = enumerate_option_candidates(
        dag,
        treatment,
        outcome,
        identify_observed_set_fn=identify_observed_set_fn,
    )
    if len(candidates) < 4:
        return None

    sampled_options = option_order_rng.sample(candidates, 4)
    option_order_rng.shuffle(sampled_options)
    return IdentificationResult(
        query_format="multi_answer",
        treatment=treatment,
        outcome=outcome,
        option_results=sampled_options,
        correct_options=_format_correct_options(sampled_options),
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_identification_query(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    query_format: str,
    option_order_rng,
    target_answer_label: Optional[str] = None,
    identify_observed_set_fn=identify_observed_set_y0,
) -> Optional[IdentificationResult]:
    if query_format == "binary":
        return build_identification_binary(
            dag,
            treatment,
            outcome,
            option_order_rng=option_order_rng,
            target_answer_label=target_answer_label,
            identify_observed_set_fn=identify_observed_set_fn,
        )
    if query_format == "multi_choice":
        return build_identification_multi_choice(
            dag,
            treatment,
            outcome,
            option_order_rng=option_order_rng,
            target_answer_label=target_answer_label,
            identify_observed_set_fn=identify_observed_set_fn,
        )
    if query_format == "multi_answer":
        return build_identification_multi_answer(
            dag,
            treatment,
            outcome,
            option_order_rng=option_order_rng,
            identify_observed_set_fn=identify_observed_set_fn,
        )
    raise ValueError(f"Unknown query_format: {query_format}")


if __name__ == "__main__":
    import random

    print("=== Testing Identification Query Sampling ===\n")

    print("--- Group 1: Serialization and Helpers ---\n")

    print("Test 1: IdentificationResult JSON round-trip")
    roundtrip_result = IdentificationResult(
        query_format="multi_answer",
        treatment=0,
        outcome=1,
        option_results=[
            OptionIdentificationResult(
                observed_variables=[0, 1],
                identifiable=False,
                method=None,
                witness=[],
            ),
            OptionIdentificationResult(
                observed_variables=[0, 1, 2],
                identifiable=True,
                method="adjustment",
                witness=["X2"],
            ),
        ],
        correct_options="B",
        is_abnormal=False,
    )
    restored = IdentificationResult.from_json_dict(roundtrip_result.to_json_dict())
    assert restored == roundtrip_result
    print("   -> PASSED\n")

    print("Test 2: _normalize_role_variables handles common payload types")
    assert _normalize_role_variables(None) == []
    assert _normalize_role_variables("X2") == ["X2"]
    assert _normalize_role_variables({"X3", "X1"}) == ["X1", "X3"]
    assert _normalize_role_variables(("X2", "X0")) == ["X0", "X2"]
    print("   -> PASSED\n")

    print("Test 3: is_abnormal_pair matches path existence")
    path_dag = nx.DiGraph([(0, 1), (1, 2)])
    assert is_abnormal_pair(path_dag, 0, 2) is False
    assert is_abnormal_pair(path_dag, 2, 0) is True
    print("   -> PASSED\n")

    print("--- Group 2: Identification by Observed Set ---\n")

    print("Test 4: Backdoor adjustment is detected when confounder is observed")
    conf_dag = nx.DiGraph([(2, 0), (2, 1), (0, 1)])
    conf_observed = identify_observed_set_pgmpy(conf_dag, 0, 1, [0, 1, 2])
    assert conf_observed.identifiable is True
    assert conf_observed.method == "adjustment"
    assert conf_observed.witness == ["X2"]
    print("   -> PASSED\n")

    print("Test 5: Same backdoor graph is not identifiable when confounder is latent")
    conf_latent = identify_observed_set_pgmpy(conf_dag, 0, 1, [0, 1])
    assert conf_latent.identifiable is False
    assert conf_latent.method is None
    assert conf_latent.witness == []
    print("   -> PASSED\n")

    print("Test 6: Frontdoor identification is detected on canonical frontdoor graph")
    frontdoor_dag = nx.DiGraph([(0, 2), (2, 1), (3, 0), (3, 1)])
    frontdoor_result = identify_observed_set_pgmpy(frontdoor_dag, 0, 1, [0, 1, 2])
    assert frontdoor_result.identifiable is True
    assert frontdoor_result.method == "frontdoor"
    assert frontdoor_result.witness == ["X2"]
    print("   -> PASSED\n")

    print("Test 7: y0 ID algorithm detects backdoor-identifiable observed set")
    conf_y0 = identify_observed_set_y0(conf_dag, 0, 1, [0, 1, 2])
    assert conf_y0.identifiable is True
    assert conf_y0.method == "y0-id"
    assert len(conf_y0.witness) > 0
    print("   -> PASSED\n")

    print("Test 8: y0 ID algorithm detects frontdoor-identifiable observed set")
    frontdoor_y0 = identify_observed_set_y0(frontdoor_dag, 0, 1, [0, 1, 2])
    assert frontdoor_y0.identifiable is True
    assert frontdoor_y0.method == "y0-id"
    assert len(frontdoor_y0.witness) > 0
    print("   -> PASSED\n")

    latent_conf_dag = nx.DiGraph([(2, 1), (2, 0), (1, 0)])
    latent_conf_result = identify_observed_set_y0(latent_conf_dag, 1, 0, [0, 1])

    print("Test 9: pgmpy returns non-identifiable for latent confounding with a direct treatment effect")
    pgmpy_false_result = identify_observed_set_pgmpy(latent_conf_dag, 1, 0, [0, 1])
    assert pgmpy_false_result.identifiable is False
    assert pgmpy_false_result.method is None
    assert pgmpy_false_result.witness == []
    print("   -> PASSED\n")

    print("Test 10: y0 ID algorithm also returns non-identifiable for latent confounding with a direct treatment effect")
    assert latent_conf_result.identifiable is False
    assert latent_conf_result.method is None
    assert latent_conf_result.witness == []
    print("   -> PASSED\n")

    print("Test 11: y0 detects hidden-confounding chains in the latent projection")
    hidden_chain_dag = nx.DiGraph([(0, 1), (2, 0), (2, 3), (3, 1)])
    hidden_chain_result = identify_observed_set_y0(hidden_chain_dag, 0, 1, [0, 1])
    assert hidden_chain_result.identifiable is False
    assert hidden_chain_result.method is None
    assert hidden_chain_result.witness == []
    print("   -> PASSED\n")

    print("Test 11b: y0 handles observed isolated nodes that do not appear in projected edges")
    isolated_node_dag = nx.DiGraph([(0, 1)])
    isolated_node_dag.add_node(2)
    isolated_node_result = identify_observed_set_y0(isolated_node_dag, 0, 1, [0, 1, 2])
    assert isolated_node_result.identifiable is True
    assert isolated_node_result.method == "y0-id"
    assert len(isolated_node_result.witness) > 0
    print("   -> PASSED\n")

    print("Test 12: y0 identifies a mixed-graph case that pgmpy misses")
    latent_frontdoor_dag = nx.DiGraph(
        [
            (0, 2),
            (0, 1),
            (2, 3),
            (2, 1),
            (3, 1),
            (4, 0),
            (4, 3),
            (5, 2),
            (5, 1),
        ]
    )
    latent_frontdoor_y0 = identify_observed_set_y0(latent_frontdoor_dag, 0, 1, [0, 1, 2, 3])
    assert latent_frontdoor_y0.identifiable is True
    assert latent_frontdoor_y0.method == "y0-id"
    assert len(latent_frontdoor_y0.witness) > 0

    latent_frontdoor_pgmpy = identify_observed_set_pgmpy(latent_frontdoor_dag, 0, 1, [0, 1, 2, 3])
    assert latent_frontdoor_pgmpy.identifiable is False
    assert latent_frontdoor_pgmpy.method is None
    assert latent_frontdoor_pgmpy.witness == []
    print("   -> PASSED\n")

    print("Test 13: enumerate_option_candidates covers every observed set containing X and Y")
    candidates = enumerate_option_candidates(
        frontdoor_dag,
        0,
        1,
        identify_observed_set_fn=identify_observed_set_pgmpy,
    )
    assert len(candidates) == 4
    candidate_sets = {tuple(candidate.observed_variables) for candidate in candidates}
    assert candidate_sets == {
        (0, 1),
        (0, 1, 2),
        (0, 1, 3),
        (0, 1, 2, 3),
    }
    print("   -> PASSED\n")

    print("--- Group 3: Query Construction ---\n")

    print("Test 14: binary query can target both Yes and No answers")
    binary_yes_result = build_identification_binary(
        conf_dag,
        treatment=0,
        outcome=1,
        option_order_rng=random.Random(5),
        target_answer_label="Yes",
        identify_observed_set_fn=identify_observed_set_pgmpy,
    )
    binary_no_result = build_identification_binary(
        conf_dag,
        treatment=0,
        outcome=1,
        option_order_rng=random.Random(6),
        target_answer_label="No",
        identify_observed_set_fn=identify_observed_set_pgmpy,
    )
    assert binary_yes_result is not None
    assert binary_yes_result.correct_options == "Yes"
    assert binary_no_result is not None
    assert binary_no_result.correct_options == "No"
    print("   -> PASSED\n")

    print("Test 15: binary query returns yes/no answer")
    binary_result = build_identification_binary(
        conf_dag,
        treatment=0,
        outcome=1,
        option_order_rng=random.Random(5),
        identify_observed_set_fn=identify_observed_set_pgmpy,
    )
    assert binary_result is not None
    assert binary_result.query_format == "binary"
    assert len(binary_result.option_results) == 1
    assert binary_result.correct_options in {"Yes", "No"}
    print("   -> PASSED\n")

    print("Test 16: multi-choice query can target all A/B/C/D answer classes")
    choice_dag = nx.DiGraph([(2, 0), (2, 1), (0, 1)])
    choice_dag.add_nodes_from([0, 1, 2, 3, 4])
    for label in OPTION_LABELS:
        multi_choice_result = build_identification_multi_choice(
            choice_dag,
            treatment=0,
            outcome=1,
            option_order_rng=random.Random(7 + OPTION_LABELS.index(label)),
            target_answer_label=label,
            identify_observed_set_fn=identify_observed_set_pgmpy,
        )
        assert multi_choice_result is not None
        assert multi_choice_result.correct_options == label
    unavailable_multi_choice = build_identification_multi_choice(
        conf_dag,
        treatment=0,
        outcome=1,
        option_order_rng=random.Random(99),
        target_answer_label="D",
        identify_observed_set_fn=identify_observed_set_pgmpy,
    )
    assert unavailable_multi_choice is None
    print("   -> PASSED\n")

    print("Test 17: multi-answer query returns 4 options with at least one distractor")
    multi_answer_result = build_identification_multi_answer(
        frontdoor_dag,
        treatment=0,
        outcome=1,
        option_order_rng=random.Random(7),
    )
    assert multi_answer_result is not None
    assert multi_answer_result.query_format == "multi_answer"
    assert len(multi_answer_result.option_results) == 4
    assert any(not option.identifiable for option in multi_answer_result.option_results)
    print("   -> PASSED\n")

    print("Test 18: multi-answer query allows all four options to be correct")
    all_identifiable_dag = nx.DiGraph([(0, 1), (2, 0), (3, 0)])
    all_correct_result = build_identification_multi_answer(
        all_identifiable_dag,
        treatment=0,
        outcome=1,
        option_order_rng=random.Random(11),
    )
    assert all_correct_result is not None
    assert len(all_correct_result.option_results) == 4
    assert all(option.identifiable for option in all_correct_result.option_results)
    assert all_correct_result.correct_options == "A,B,C,D"
    print("   -> PASSED\n")

    print("=== All Identification Query Sampling Tests Passed ===")
