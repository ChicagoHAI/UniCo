"""
frontdoor_query_sampling.py

Construct frontdoor-set queries over DAGs.

Each query fixes a treatment X and outcome Y, then asks whether or under which
candidate variable sets a valid frontdoor set exists for estimating
P(Y | do(X)). pgmpy Frontdoor is used as the source of truth.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

import networkx as nx
from pgmpy.base import DAG
from pgmpy.identification import Frontdoor


OPTION_LABELS = ["A", "B", "C", "D"]
QUERY_FORMATS = ["binary", "multi_choice", "multi_answer"]


@dataclass
class OptionFrontdoorResult:
    """Frontdoor validity for one candidate frontdoor set."""

    frontdoor_variables: List[int]
    valid_frontdoor: bool
    method: str
    witness: List[str]

    def to_json_dict(self) -> Dict:
        return {
            "frontdoor_variables": self.frontdoor_variables,
            "valid_frontdoor": self.valid_frontdoor,
            "method": self.method,
            "witness": self.witness,
        }


@dataclass
class FrontdoorResult:
    """Ground-truth result for one frontdoor query item."""

    query_format: str
    treatment: int
    outcome: int
    option_results: List[OptionFrontdoorResult]
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
    def from_json_dict(cls, payload: Dict) -> "FrontdoorResult":
        return cls(
            query_format=payload.get("query_format", "multi_answer"),
            treatment=payload["treatment"],
            outcome=payload["outcome"],
            option_results=[
                OptionFrontdoorResult(
                    frontdoor_variables=option["frontdoor_variables"],
                    valid_frontdoor=option["valid_frontdoor"],
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


def _build_frontdoor_dag(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    frontdoor_variables: Sequence[int] | None = None,
) -> DAG:
    roles = {
        "exposures": _node_label(treatment),
        "outcomes": _node_label(outcome),
    }
    if frontdoor_variables is not None:
        roles["frontdoor"] = [_node_label(node) for node in sorted(frontdoor_variables)]
    pgmpy_dag = DAG(
        ebunch=[(_node_label(src), _node_label(dst)) for src, dst in dag.edges()],
        latents=[],
    )
    pgmpy_dag.add_nodes_from([_node_label(node) for node in dag.nodes()])
    pgmpy_dag.with_role(role="exposures", variables=roles["exposures"], inplace=True)
    pgmpy_dag.with_role(role="outcomes", variables=roles["outcomes"], inplace=True)
    pgmpy_dag.with_role(role="frontdoor", variables=roles.get("frontdoor", []), inplace=True)
    return pgmpy_dag


def _candidate_nodes(dag: nx.DiGraph, treatment: int, outcome: int) -> List[int]:
    return sorted(node for node in dag.nodes() if node not in {treatment, outcome})



def validate_frontdoor_set(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    frontdoor_variables: Sequence[int],
) -> OptionFrontdoorResult:
    """Validate one candidate frontdoor set using pgmpy Frontdoor().validate."""

    candidate = tuple(sorted(frontdoor_variables))
    candidate_dag = _build_frontdoor_dag(dag, treatment, outcome, candidate)
    valid_frontdoor = bool(Frontdoor().validate(candidate_dag))
    witness = [_node_label(node) for node in candidate] if valid_frontdoor else []
    return OptionFrontdoorResult(
        frontdoor_variables=list(candidate),
        valid_frontdoor=valid_frontdoor,
        method="frontdoor_validate",
        witness=witness,
    )


def enumerate_option_candidates(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
) -> List[OptionFrontdoorResult]:
    """Enumerate validation results for every candidate frontdoor set."""

    candidates: List[OptionFrontdoorResult] = []
    other_nodes = _candidate_nodes(dag, treatment, outcome)
    for subset_size in range(len(other_nodes) + 1):
        for subset in itertools.combinations(other_nodes, subset_size):
            candidates.append(validate_frontdoor_set(dag, treatment, outcome, subset))
    return candidates


def _format_multi_choice_correct_option(options: List[OptionFrontdoorResult]) -> str:
    if len(options) != 2:
        raise ValueError("multi_choice requires exactly 2 candidate frontdoor-set options")

    first_valid = options[0].valid_frontdoor
    second_valid = options[1].valid_frontdoor
    if not first_valid and not second_valid:
        return "A"
    if first_valid and not second_valid:
        return "B"
    if not first_valid and second_valid:
        return "C"
    return "D"


def _format_correct_options(options: List[OptionFrontdoorResult]) -> str:
    labels = [label for label, option in zip(OPTION_LABELS, options) if option.valid_frontdoor]
    return "" if not labels else ",".join(labels)


def build_frontdoor_binary(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    target_answer_label: Optional[str] = None,
) -> Optional[FrontdoorResult]:
    candidates = enumerate_option_candidates(dag, treatment, outcome)
    if target_answer_label is not None:
        if target_answer_label not in {"Yes", "No"}:
            raise ValueError(f"Unknown binary target_answer_label: {target_answer_label}")
        target_valid = target_answer_label == "Yes"
        candidates = [option for option in candidates if option.valid_frontdoor == target_valid]
    if not candidates:
        return None

    sampled_option = option_order_rng.choice(candidates)
    return FrontdoorResult(
        query_format="binary",
        treatment=treatment,
        outcome=outcome,
        option_results=[sampled_option],
        correct_options="Yes" if sampled_option.valid_frontdoor else "No",
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_frontdoor_multi_choice(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    target_answer_label: Optional[str] = None,
) -> Optional[FrontdoorResult]:
    candidates = enumerate_option_candidates(dag, treatment, outcome)
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
    return FrontdoorResult(
        query_format="multi_choice",
        treatment=treatment,
        outcome=outcome,
        option_results=sampled_options,
        correct_options=_format_multi_choice_correct_option(sampled_options),
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_frontdoor_multi_answer(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
) -> Optional[FrontdoorResult]:
    candidates = enumerate_option_candidates(dag, treatment, outcome)
    if len(candidates) < 4:
        return None

    sampled_options = option_order_rng.sample(candidates, 4)
    option_order_rng.shuffle(sampled_options)
    return FrontdoorResult(
        query_format="multi_answer",
        treatment=treatment,
        outcome=outcome,
        option_results=sampled_options,
        correct_options=_format_correct_options(sampled_options),
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_frontdoor_query(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    query_format: str,
    option_order_rng,
    target_answer_label: Optional[str] = None,
) -> Optional[FrontdoorResult]:
    if query_format == "binary":
        return build_frontdoor_binary(dag, treatment, outcome, option_order_rng=option_order_rng, target_answer_label=target_answer_label)
    if query_format == "multi_choice":
        return build_frontdoor_multi_choice(dag, treatment, outcome, option_order_rng=option_order_rng, target_answer_label=target_answer_label)
    if query_format == "multi_answer":
        return build_frontdoor_multi_answer(dag, treatment, outcome, option_order_rng=option_order_rng)
    raise ValueError(f"Unknown query_format: {query_format}")

if __name__ == "__main__":
    def _criterion_1_intercepts_all_directed_paths(
        dag: nx.DiGraph,
        treatment: int,
        outcome: int,
        frontdoor_variables: Sequence[int],
    ) -> bool:
        remaining_nodes = [
            node for node in dag.nodes() if node not in set(frontdoor_variables)
        ]
        subgraph = dag.subgraph(remaining_nodes)
        return not nx.has_path(subgraph, treatment, outcome)

    def _criterion_2_no_backdoor_path_from_x_to_z(
        dag: nx.DiGraph,
        treatment: int,
        frontdoor_variables: Sequence[int],
    ) -> bool:
        pgmpy_dag = DAG(
            ebunch=[(_node_label(src), _node_label(dst)) for src, dst in dag.edges()],
            latents=[],
        )
        observed = [_node_label(treatment)]
        for mediator in frontdoor_variables:
            for parent in dag.predecessors(treatment):
                if pgmpy_dag.is_dconnected(
                    start=_node_label(parent),
                    end=_node_label(mediator),
                    observed=observed,
                ):
                    return False
        return True

    def _criterion_3_all_backdoor_paths_from_z_to_y_blocked_by_x(
        dag: nx.DiGraph,
        treatment: int,
        outcome: int,
        frontdoor_variables: Sequence[int],
    ) -> bool:
        pgmpy_dag = DAG(
            ebunch=[(_node_label(src), _node_label(dst)) for src, dst in dag.edges()],
            latents=[],
        )
        for mediator in frontdoor_variables:
            observed = [_node_label(mediator), _node_label(treatment)]
            for parent in dag.predecessors(mediator):
                if pgmpy_dag.is_dconnected(
                    start=_node_label(parent),
                    end=_node_label(outcome),
                    observed=observed,
                ):
                    return False
        return True

    print("=== Testing Frontdoor Query Sampling ===\n")

    print("--- Group 1: Serialization and Helpers ---\n")

    print("Test 1: FrontdoorResult JSON round-trip")
    roundtrip_result = FrontdoorResult(
        query_format="multi_answer",
        treatment=0,
        outcome=2,
        option_results=[
            OptionFrontdoorResult(
                frontdoor_variables=[1, 4],
                valid_frontdoor=True,
                method="frontdoor_validate",
                witness=["X1", "X4"],
            ),
            OptionFrontdoorResult(
                frontdoor_variables=[3],
                valid_frontdoor=False,
                method="frontdoor_validate",
                witness=[],
            ),
        ],
        correct_options="A",
        is_abnormal=False,
    )
    restored = FrontdoorResult.from_json_dict(roundtrip_result.to_json_dict())
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

    print("--- Group 2: Frontdoor Validation ---\n")

    print("Test 4: criterion 1 requires Z to intercept all directed paths from X to Y")
    frontdoor_dag = nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 2), (0, 4), (4, 2)])
    assert _criterion_1_intercepts_all_directed_paths(frontdoor_dag, 0, 2, [1, 4]) is True
    assert _criterion_1_intercepts_all_directed_paths(frontdoor_dag, 0, 2, [1]) is False
    print("   -> PASSED\n")

    print("Test 5: criterion 2 requires no backdoor path from X to Z")
    criterion2_dag = nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 1)])
    assert _criterion_2_no_backdoor_path_from_x_to_z(frontdoor_dag, 0, [1, 4]) is True
    assert _criterion_2_no_backdoor_path_from_x_to_z(criterion2_dag, 0, [1]) is False
    assert validate_frontdoor_set(criterion2_dag, 0, 2, [1]).valid_frontdoor is False
    print("   -> PASSED\n")

    print("Test 6: criterion 3 requires all backdoor paths from Z to Y to be blocked by X")
    criterion3_dag = nx.DiGraph([(0, 1), (1, 2), (3, 1), (3, 2)])
    assert _criterion_3_all_backdoor_paths_from_z_to_y_blocked_by_x(frontdoor_dag, 0, 2, [1, 4]) is True
    assert _criterion_3_all_backdoor_paths_from_z_to_y_blocked_by_x(criterion3_dag, 0, 2, [1]) is False
    assert validate_frontdoor_set(criterion3_dag, 0, 2, [1]).valid_frontdoor is False
    print("   -> PASSED\n")

    print("Test 7: canonical frontdoor set is detected on a frontdoor graph")
    mediator_result = validate_frontdoor_set(frontdoor_dag, 0, 2, [1, 4])
    assert mediator_result.valid_frontdoor is True
    assert mediator_result.method == "frontdoor_validate"
    print("   -> PASSED\n")

    print("Test 8: candidate enumeration covers all non-treatment/non-outcome subsets")
    candidates = enumerate_option_candidates(frontdoor_dag, 0, 2)
    candidate_sets = {tuple(candidate.frontdoor_variables) for candidate in candidates}
    assert () in candidate_sets
    assert (1,) in candidate_sets
    assert (3,) in candidate_sets
    assert (4,) in candidate_sets
    assert (1, 4) in candidate_sets
    assert all(0 not in option.frontdoor_variables and 2 not in option.frontdoor_variables for option in candidates)
    print("   -> PASSED\n")

    print("--- Group 3: Query Construction ---\n")

    print("Test 9: binary query can target both Yes and No answers")
    binary_yes_result = build_frontdoor_binary(
        frontdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(5),
        target_answer_label="Yes",
    )
    binary_no_result = build_frontdoor_binary(
        frontdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(6),
        target_answer_label="No",
    )
    assert binary_yes_result is not None
    assert binary_yes_result.correct_options == "Yes"
    assert binary_no_result is not None
    assert binary_no_result.correct_options == "No"
    print("   -> PASSED\n")

    print("Test 10: binary query returns yes/no answer")
    binary_result = build_frontdoor_binary(
        frontdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(5),
    )
    assert binary_result is not None
    assert binary_result.query_format == "binary"
    assert len(binary_result.option_results) == 1
    assert binary_result.correct_options in {"Yes", "No"}
    print("   -> PASSED\n")

    print("Test 11: multi-choice query can target all A/B/C/D answer classes")
    multi_choice_dag = nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 2), (0, 4), (4, 2)])
    multi_choice_dag.add_node(5)
    for label in OPTION_LABELS:
        multi_choice_result = build_frontdoor_multi_choice(
            multi_choice_dag,
            treatment=0,
            outcome=2,
            option_order_rng=random.Random(7 + OPTION_LABELS.index(label)),
            target_answer_label=label,
        )
        assert multi_choice_result is not None
        assert multi_choice_result.correct_options == label
    unavailable_multi_choice = build_frontdoor_multi_choice(
        nx.DiGraph([(0, 1), (1, 2), (3, 0), (3, 2)]),
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(99),
        target_answer_label="D",
    )
    assert unavailable_multi_choice is None
    print("   -> PASSED\n")

    print("Test 12: multi-answer query returns 4 candidate sets")
    multi_answer_result = build_frontdoor_multi_answer(
        frontdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(9),
    )
    assert multi_answer_result is not None
    assert multi_answer_result.query_format == "multi_answer"
    assert len(multi_answer_result.option_results) == 4
    assert all(0 not in option.frontdoor_variables and 2 not in option.frontdoor_variables for option in multi_answer_result.option_results)
    print("   -> PASSED\n")

    print("=== All Frontdoor Query Sampling Tests Passed ===")
