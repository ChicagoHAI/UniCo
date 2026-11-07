"""markov_blanket_query_sampling.py"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx

from data_generation.dataset_utils.association_shared import compute_markov_blanket, compute_markov_blanket_membership

OPTION_LABELS = ["A", "B", "C", "D"]
QUERY_FORMATS = ["binary", "multi_choice", "multi_answer"]


@dataclass
class OptionMarkovBlanketResult:
    target: int
    candidate: int
    in_markov_blanket: bool
    markov_blanket: List[int]
    method: str
    witness: List[str]

    def to_json_dict(self) -> Dict:
        return {
            "target": self.target,
            "candidate": self.candidate,
            "in_markov_blanket": self.in_markov_blanket,
            "markov_blanket": self.markov_blanket,
            "method": self.method,
            "witness": self.witness,
        }


@dataclass
class MarkovBlanketResult:
    query_format: str
    option_results: List[OptionMarkovBlanketResult]
    correct_options: str
    is_abnormal: bool

    def to_json_dict(self) -> Dict:
        return {
            "query_format": self.query_format,
            "option_results": [option.to_json_dict() for option in self.option_results],
            "correct_options": self.correct_options,
            "is_abnormal": self.is_abnormal,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "MarkovBlanketResult":
        return cls(
            query_format=data["query_format"],
            option_results=[OptionMarkovBlanketResult(**option) for option in data["option_results"]],
            correct_options=data["correct_options"],
            is_abnormal=bool(data.get("is_abnormal", False)),
        )


def validate_markov_blanket_membership(dag: nx.DiGraph, target: int, candidate: int) -> OptionMarkovBlanketResult:
    in_markov_blanket, blanket = compute_markov_blanket_membership(dag, target, candidate)
    witness = [f"X{candidate} in MB(X{target})"] if in_markov_blanket else []
    return OptionMarkovBlanketResult(target, candidate, in_markov_blanket, blanket, "markov_blanket_membership", witness)


def is_abnormal_markov_blanket_target(dag: nx.DiGraph, target: int) -> bool:
    return len(compute_markov_blanket(dag, target)) == 0


def is_abnormal_markov_blanket_query(dag: nx.DiGraph, result: MarkovBlanketResult) -> bool:
    return all(is_abnormal_markov_blanket_target(dag, option.target) for option in result.option_results)


def enumerate_option_candidates(dag: nx.DiGraph) -> List[OptionMarkovBlanketResult]:
    return [validate_markov_blanket_membership(dag, target, candidate) for target in dag.nodes for candidate in dag.nodes if candidate != target]


def _format_multi_choice_correct_option(options: List[OptionMarkovBlanketResult]) -> str:
    a, b = options[0].in_markov_blanket, options[1].in_markov_blanket
    if not a and not b:
        return "A"
    if a and not b:
        return "B"
    if not a and b:
        return "C"
    return "D"


def _format_correct_options(options: List[OptionMarkovBlanketResult]) -> str:
    labels = [label for label, option in zip(OPTION_LABELS, options) if option.in_markov_blanket]
    return "" if not labels else ",".join(labels)


def build_markov_blanket_binary(dag: nx.DiGraph, *, option_order_rng, target_answer_label: Optional[str] = None) -> Optional[MarkovBlanketResult]:
    candidates = enumerate_option_candidates(dag)
    if target_answer_label is not None:
        candidates = [option for option in candidates if option.in_markov_blanket == (target_answer_label == "Yes")]
    if not candidates:
        return None
    sampled = option_order_rng.choice(candidates)
    result = MarkovBlanketResult("binary", [sampled], "Yes" if sampled.in_markov_blanket else "No", False)
    result.is_abnormal = is_abnormal_markov_blanket_query(dag, result)
    return result


def build_markov_blanket_multi_choice(dag: nx.DiGraph, *, option_order_rng, target_answer_label: Optional[str] = None) -> Optional[MarkovBlanketResult]:
    pairs = [list(pair) for pair in itertools.combinations(enumerate_option_candidates(dag), 2) if target_answer_label is None or _format_multi_choice_correct_option(list(pair)) == target_answer_label]
    if not pairs:
        return None
    sampled = option_order_rng.choice(pairs)
    result = MarkovBlanketResult("multi_choice", sampled, _format_multi_choice_correct_option(sampled), False)
    result.is_abnormal = is_abnormal_markov_blanket_query(dag, result)
    return result


def build_markov_blanket_multi_answer(dag: nx.DiGraph, *, option_order_rng) -> Optional[MarkovBlanketResult]:
    candidates = enumerate_option_candidates(dag)
    if len(candidates) < 4:
        return None
    sampled = option_order_rng.sample(candidates, 4)
    option_order_rng.shuffle(sampled)
    result = MarkovBlanketResult("multi_answer", sampled, _format_correct_options(sampled), False)
    result.is_abnormal = is_abnormal_markov_blanket_query(dag, result)
    return result


def build_markov_blanket_query(dag: nx.DiGraph, *, query_format: str, option_order_rng, target_answer_label: Optional[str] = None) -> Optional[MarkovBlanketResult]:
    if query_format == "binary":
        return build_markov_blanket_binary(dag, option_order_rng=option_order_rng, target_answer_label=target_answer_label)
    if query_format == "multi_choice":
        return build_markov_blanket_multi_choice(dag, option_order_rng=option_order_rng, target_answer_label=target_answer_label)
    if query_format == "multi_answer":
        return build_markov_blanket_multi_answer(dag, option_order_rng=option_order_rng)
    raise ValueError(query_format)


if __name__ == "__main__":
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3), (4, 3)])
    assert validate_markov_blanket_membership(dag, 2, 0).in_markov_blanket is True
    assert validate_markov_blanket_membership(dag, 0, 4).in_markov_blanket is False
    singleton_dag = nx.DiGraph(); singleton_dag.add_nodes_from([0, 1])
    assert is_abnormal_markov_blanket_target(singleton_dag, 0) is True
    binary_yes = build_markov_blanket_binary(dag, option_order_rng=random.Random(0), target_answer_label='Yes')
    binary_no = build_markov_blanket_binary(dag, option_order_rng=random.Random(0), target_answer_label='No')
    assert binary_yes.correct_options == 'Yes'
    assert binary_yes.option_results[0].in_markov_blanket is True
    assert binary_no.correct_options == 'No'
    assert binary_no.option_results[0].in_markov_blanket is False
    for label in OPTION_LABELS:
        result = build_markov_blanket_multi_choice(dag, option_order_rng=random.Random(1), target_answer_label=label)
        assert result is not None and result.correct_options == label
        assert _format_multi_choice_correct_option(result.option_results) == result.correct_options
    multi_answer_result = build_markov_blanket_multi_answer(dag, option_order_rng=random.Random(3))
    assert len(multi_answer_result.option_results) == 4
    assert _format_correct_options(multi_answer_result.option_results) == multi_answer_result.correct_options
    print("Markov Blanket query sampling checks passed.")
