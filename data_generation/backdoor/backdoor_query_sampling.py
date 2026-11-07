"""
backdoor_query_sampling.py

Construct backdoor-adjustment queries over DAGs.

Each query fixes a treatment X and outcome Y, then asks whether or under which
candidate variable sets a valid backdoor adjustment set exists for estimating
P(Y | do(X)). pgmpy Adjustment is used as the source of truth.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random

import networkx as nx
from pgmpy.base import DAG
from pgmpy.identification import Adjustment


OPTION_LABELS = ["A", "B", "C", "D"]
QUERY_FORMATS = ["binary", "multi_choice", "multi_answer"]


@dataclass
class OptionBackdoorResult:
    """Backdoor-adjustment validity for one candidate adjustment set."""

    adjustment_variables: List[int]
    valid_adjustment: bool
    method: str
    witness: List[str]

    def to_json_dict(self) -> Dict:
        return {
            "adjustment_variables": self.adjustment_variables,
            "valid_adjustment": self.valid_adjustment,
            "method": self.method,
            "witness": self.witness,
        }


@dataclass
class BackdoorResult:
    """Ground-truth result for one backdoor-adjustment query item."""

    query_format: str
    treatment: int
    outcome: int
    option_results: List[OptionBackdoorResult]
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
    def from_json_dict(cls, payload: Dict) -> "BackdoorResult":
        return cls(
            query_format=payload.get("query_format", "multi_answer"),
            treatment=payload["treatment"],
            outcome=payload["outcome"],
            option_results=[
                OptionBackdoorResult(
                    adjustment_variables=option["adjustment_variables"],
                    valid_adjustment=option["valid_adjustment"],
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


def _build_backdoor_dag(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    adjustment_variables: Sequence[int] | None = None,
) -> DAG:
    roles = {
        "exposures": _node_label(treatment),
        "outcomes": _node_label(outcome),
    }
    if adjustment_variables is not None:
        roles["adjustment"] = [_node_label(node) for node in sorted(adjustment_variables)]
    pgmpy_dag = DAG(
        ebunch=[(_node_label(src), _node_label(dst)) for src, dst in dag.edges()],
        latents=[],
    )
    pgmpy_dag.add_nodes_from([_node_label(node) for node in dag.nodes()])
    pgmpy_dag.with_role(role="exposures", variables=roles["exposures"], inplace=True)
    pgmpy_dag.with_role(role="outcomes", variables=roles["outcomes"], inplace=True)
    pgmpy_dag.with_role(role="adjustment", variables=roles.get("adjustment", []), inplace=True)
    return pgmpy_dag


def _candidate_nodes(dag: nx.DiGraph, treatment: int, outcome: int) -> List[int]:
    return sorted(node for node in dag.nodes() if node not in {treatment, outcome})



def validate_backdoor_set(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    adjustment_variables: Sequence[int],
) -> OptionBackdoorResult:
    """Validate one candidate adjustment set using the backdoor criterion plus pgmpy."""

    candidate = tuple(sorted(adjustment_variables))
    treatment_descendants = set(nx.descendants(dag, treatment))
    if any(node in treatment_descendants for node in candidate):
        return OptionBackdoorResult(
            adjustment_variables=list(candidate),
            valid_adjustment=False,
            method="descendant_of_treatment",
            witness=[],
        )

    candidate_dag = _build_backdoor_dag(dag, treatment, outcome, candidate)
    valid_adjustment = bool(Adjustment().validate(candidate_dag))
    witness = [_node_label(node) for node in candidate] if valid_adjustment else []
    return OptionBackdoorResult(
        adjustment_variables=list(candidate),
        valid_adjustment=valid_adjustment,
        method="adjustment_validate",
        witness=witness,
    )

def enumerate_option_candidates(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
) -> List[OptionBackdoorResult]:
    """Enumerate validation results for every candidate adjustment set."""

    candidates: List[OptionBackdoorResult] = []
    other_nodes = _candidate_nodes(dag, treatment, outcome)
    for subset_size in range(len(other_nodes) + 1):
        for subset in itertools.combinations(other_nodes, subset_size):
            candidates.append(validate_backdoor_set(dag, treatment, outcome, subset))
    return candidates

def _format_multi_choice_correct_option(options: List[OptionBackdoorResult]) -> str:
    if len(options) != 2:
        raise ValueError("multi_choice requires exactly 2 candidate adjustment-set options")

    first_valid = options[0].valid_adjustment
    second_valid = options[1].valid_adjustment
    if not first_valid and not second_valid:
        return "A"
    if first_valid and not second_valid:
        return "B"
    if not first_valid and second_valid:
        return "C"
    return "D"


def _format_correct_options(options: List[OptionBackdoorResult]) -> str:
    labels = [label for label, option in zip(OPTION_LABELS, options) if option.valid_adjustment]
    return "" if not labels else ",".join(labels)


def build_backdoor_binary(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    target_answer_label: Optional[str] = None,
) -> Optional[BackdoorResult]:
    candidates = enumerate_option_candidates(dag, treatment, outcome)
    if target_answer_label is not None:
        if target_answer_label not in {"Yes", "No"}:
            raise ValueError(f"Unknown binary target_answer_label: {target_answer_label}")
        target_valid = target_answer_label == "Yes"
        candidates = [option for option in candidates if option.valid_adjustment == target_valid]
    if not candidates:
        return None

    sampled_option = option_order_rng.choice(candidates)
    return BackdoorResult(
        query_format="binary",
        treatment=treatment,
        outcome=outcome,
        option_results=[sampled_option],
        correct_options="Yes" if sampled_option.valid_adjustment else "No",
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_backdoor_multi_choice(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
    target_answer_label: Optional[str] = None,
) -> Optional[BackdoorResult]:
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
    return BackdoorResult(
        query_format="multi_choice",
        treatment=treatment,
        outcome=outcome,
        option_results=sampled_options,
        correct_options=_format_multi_choice_correct_option(sampled_options),
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_backdoor_multi_answer(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    option_order_rng,
) -> Optional[BackdoorResult]:
    candidates = enumerate_option_candidates(dag, treatment, outcome)
    if len(candidates) < 4:
        return None

    sampled_options = option_order_rng.sample(candidates, 4)
    option_order_rng.shuffle(sampled_options)
    return BackdoorResult(
        query_format="multi_answer",
        treatment=treatment,
        outcome=outcome,
        option_results=sampled_options,
        correct_options=_format_correct_options(sampled_options),
        is_abnormal=is_abnormal_pair(dag, treatment, outcome),
    )


def build_backdoor_query(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
    *,
    query_format: str,
    option_order_rng,
    target_answer_label: Optional[str] = None,
) -> Optional[BackdoorResult]:
    if query_format == "binary":
        return build_backdoor_binary(dag, treatment, outcome, option_order_rng=option_order_rng, target_answer_label=target_answer_label)
    if query_format == "multi_choice":
        return build_backdoor_multi_choice(dag, treatment, outcome, option_order_rng=option_order_rng, target_answer_label=target_answer_label)
    if query_format == "multi_answer":
        return build_backdoor_multi_answer(dag, treatment, outcome, option_order_rng=option_order_rng)
    raise ValueError(f"Unknown query_format: {query_format}")

if __name__ == "__main__":
    print("=== Testing Backdoor Query Sampling ===\n")

    print("--- Group 1: Serialization and Helpers ---\n")

    print("Test 1: BackdoorResult JSON round-trip")
    roundtrip_result = BackdoorResult(
        query_format="multi_answer",
        treatment=0,
        outcome=2,
        option_results=[
            OptionBackdoorResult(
                adjustment_variables=[3],
                valid_adjustment=True,
                method="adjustment_validate",
                witness=["X3"],
            ),
            OptionBackdoorResult(
                adjustment_variables=[4],
                valid_adjustment=False,
                method="adjustment_validate",
                witness=[],
            ),
        ],
        correct_options="A",
        is_abnormal=False,
    )
    restored = BackdoorResult.from_json_dict(roundtrip_result.to_json_dict())
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

    print("--- Group 2: Backdoor Validation ---\n")

    print("Test 4: canonical backdoor set is detected on a confounded graph")
    backdoor_dag = nx.DiGraph([(3, 0), (3, 2), (0, 1), (1, 2), (0, 4), (4, 2), (0, 5), (2, 5)])
    confounder_result = validate_backdoor_set(backdoor_dag, 0, 2, [3])
    assert confounder_result.valid_adjustment is True
    assert confounder_result.method == "adjustment_validate"
    print("   -> PASSED\n")

    print("Test 5: adding a treatment descendant can invalidate an otherwise valid backdoor set")
    invalidating_dag = nx.DiGraph([(3, 0), (3, 2), (0, 4), (2, 4)])
    valid_base_result = validate_backdoor_set(invalidating_dag, 0, 2, [3])
    descendant_augmented_result = validate_backdoor_set(invalidating_dag, 0, 2, [3, 4])
    assert valid_base_result.valid_adjustment is True
    assert descendant_augmented_result.valid_adjustment is False
    print("   -> PASSED\n")

    print("Test 6: candidate enumeration now covers all non-treatment/non-outcome subsets")
    candidates = enumerate_option_candidates(backdoor_dag, 0, 2)
    candidate_sets = {tuple(candidate.adjustment_variables) for candidate in candidates}
    assert () in candidate_sets
    assert (3,) in candidate_sets
    assert (4,) in candidate_sets
    assert (5,) in candidate_sets
    assert (3, 4) in candidate_sets
    assert all(0 not in option.adjustment_variables and 2 not in option.adjustment_variables for option in candidates)
    print("   -> PASSED\n")

    print("--- Group 3: Query Construction ---\n")

    print("Test 7: binary query can target both Yes and No answers")
    binary_yes_result = build_backdoor_binary(
        backdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(5),
        target_answer_label="Yes",
    )
    binary_no_result = build_backdoor_binary(
        backdoor_dag,
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

    print("Test 8: binary query returns yes/no answer")
    binary_result = build_backdoor_binary(
        backdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(5),
    )
    assert binary_result is not None
    assert binary_result.query_format == "binary"
    assert len(binary_result.option_results) == 1
    assert binary_result.correct_options in {"Yes", "No"}
    print("   -> PASSED\n")

    print("Test 9: multi-choice query can target all A/B/C/D answer classes")
    multi_choice_dag = nx.DiGraph([(4, 0), (4, 2), (0, 1), (1, 2)])
    multi_choice_dag.add_node(5)
    for label in OPTION_LABELS:
        multi_choice_result = build_backdoor_multi_choice(
            multi_choice_dag,
            treatment=0,
            outcome=2,
            option_order_rng=random.Random(7 + OPTION_LABELS.index(label)),
            target_answer_label=label,
        )
        assert multi_choice_result is not None
        assert multi_choice_result.correct_options == label
    unavailable_multi_choice = build_backdoor_multi_choice(
        nx.DiGraph([(3, 0), (3, 2), (0, 1), (1, 2)]),
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(99),
        target_answer_label="D",
    )
    assert unavailable_multi_choice is None
    print("   -> PASSED\n")

    print("Test 10: multi-answer query returns 4 candidate sets")
    multi_answer_result = build_backdoor_multi_answer(
        backdoor_dag,
        treatment=0,
        outcome=2,
        option_order_rng=random.Random(9),
    )
    assert multi_answer_result is not None
    assert multi_answer_result.query_format == "multi_answer"
    assert len(multi_answer_result.option_results) == 4
    assert all(0 not in option.adjustment_variables and 2 not in option.adjustment_variables for option in multi_answer_result.option_results)
    print("   -> PASSED\n")

    print("=== All Backdoor Query Sampling Tests Passed ===")
