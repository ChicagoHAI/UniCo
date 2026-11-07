"""independence_test_query_sampling.py"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import networkx as nx

from data_generation.dataset_utils.association_shared import compute_independence_test, is_structurally_related
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM

OPTION_LABELS = ["A", "B", "C", "D"]
QUERY_FORMATS = ["binary", "multi_choice", "multi_answer"]


@dataclass
class OptionIndependenceResult:
    source: int
    target: int
    conditioning_nodes: List[int]
    independent: bool
    method: str
    witness: List[str]

    def to_json_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "conditioning_nodes": self.conditioning_nodes,
            "independent": self.independent,
            "method": self.method,
            "witness": self.witness,
        }


@dataclass
class IndependenceTestResult:
    query_format: str
    option_results: List[OptionIndependenceResult]
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
    def from_json_dict(cls, data: Dict) -> "IndependenceTestResult":
        return cls(
            query_format=data["query_format"],
            option_results=[OptionIndependenceResult(**option) for option in data["option_results"]],
            correct_options=data["correct_options"],
            is_abnormal=bool(data.get("is_abnormal", False)),
        )


def _statement_text(source: int, target: int, conditioning_nodes: Sequence[int]) -> str:
    if conditioning_nodes:
        return f"X{source} ⟂ X{target} | {{{', '.join(f'X{n}' for n in conditioning_nodes)}}}"
    return f"X{source} ⟂ X{target}"


def validate_independence_statement(scm: SCMBase, source: int, target: int, conditioning_nodes: Sequence[int]) -> OptionIndependenceResult:
    cond = sorted(conditioning_nodes)
    independent = compute_independence_test(scm, source, target, cond)
    witness = [_statement_text(source, target, cond)] if independent else []
    return OptionIndependenceResult(source, target, list(cond), independent, "exact_independence", witness)


def is_abnormal_independence_option(dag: nx.DiGraph, source: int, target: int) -> bool:
    return not is_structurally_related(dag, source, target)


def is_abnormal_independence_query(dag: nx.DiGraph, result: IndependenceTestResult) -> bool:
    return all(is_abnormal_independence_option(dag, option.source, option.target) for option in result.option_results)


def enumerate_option_candidates(scm: SCMBase, *, max_conditioning: int = 2) -> List[OptionIndependenceResult]:
    candidates = []
    for source, target in itertools.combinations(scm.nodes, 2):
        available = [node for node in scm.nodes if node not in {source, target}]
        for size in range(min(max_conditioning, len(available)) + 1):
            for subset in itertools.combinations(available, size):
                candidates.append(validate_independence_statement(scm, source, target, subset))
    return candidates


def _format_multi_choice_correct_option(options: List[OptionIndependenceResult]) -> str:
    a, b = options[0].independent, options[1].independent
    if not a and not b:
        return "A"
    if a and not b:
        return "B"
    if not a and b:
        return "C"
    return "D"


def _format_correct_options(options: List[OptionIndependenceResult]) -> str:
    labels = [label for label, option in zip(OPTION_LABELS, options) if option.independent]
    return "" if not labels else ",".join(labels)


def build_independence_test_binary(scm: SCMBase, *, option_order_rng, max_conditioning: int = 2, target_answer_label: Optional[str] = None) -> Optional[IndependenceTestResult]:
    candidates = enumerate_option_candidates(scm, max_conditioning=max_conditioning)
    if target_answer_label is not None:
        candidates = [option for option in candidates if option.independent == (target_answer_label == "Yes")]
    if not candidates:
        return None
    sampled = option_order_rng.choice(candidates)
    result = IndependenceTestResult("binary", [sampled], "Yes" if sampled.independent else "No", False)
    result.is_abnormal = is_abnormal_independence_query(scm.dag, result)
    return result


def build_independence_test_multi_choice(scm: SCMBase, *, option_order_rng, max_conditioning: int = 2, target_answer_label: Optional[str] = None) -> Optional[IndependenceTestResult]:
    candidates = enumerate_option_candidates(scm, max_conditioning=max_conditioning)
    candidate_pairs = [list(pair) for pair in itertools.combinations(candidates, 2) if target_answer_label is None or _format_multi_choice_correct_option(list(pair)) == target_answer_label]
    if not candidate_pairs:
        return None
    sampled = option_order_rng.choice(candidate_pairs)
    result = IndependenceTestResult("multi_choice", sampled, _format_multi_choice_correct_option(sampled), False)
    result.is_abnormal = is_abnormal_independence_query(scm.dag, result)
    return result


def build_independence_test_multi_answer(scm: SCMBase, *, option_order_rng, max_conditioning: int = 2) -> Optional[IndependenceTestResult]:
    candidates = enumerate_option_candidates(scm, max_conditioning=max_conditioning)
    if len(candidates) < 4:
        return None
    sampled = option_order_rng.sample(candidates, 4)
    option_order_rng.shuffle(sampled)
    result = IndependenceTestResult("multi_answer", sampled, _format_correct_options(sampled), False)
    result.is_abnormal = is_abnormal_independence_query(scm.dag, result)
    return result


def build_independence_test_query(scm: SCMBase, *, query_format: str, option_order_rng, max_conditioning: int = 2, target_answer_label: Optional[str] = None) -> Optional[IndependenceTestResult]:
    if query_format == "binary":
        return build_independence_test_binary(scm, option_order_rng=option_order_rng, max_conditioning=max_conditioning, target_answer_label=target_answer_label)
    if query_format == "multi_choice":
        return build_independence_test_multi_choice(scm, option_order_rng=option_order_rng, max_conditioning=max_conditioning, target_answer_label=target_answer_label)
    if query_format == "multi_answer":
        return build_independence_test_multi_answer(scm, option_order_rng=option_order_rng, max_conditioning=max_conditioning)
    raise ValueError(query_format)


if __name__ == "__main__":
    chain_dag = nx.DiGraph([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)
    assert validate_independence_statement(chain_scm, 0, 2, []).independent is False
    assert validate_independence_statement(chain_scm, 0, 2, [1]).independent is True
    assert is_abnormal_independence_option(chain_dag, 0, 2) is False
    unrelated_dag = nx.DiGraph([(0, 1), (2, 3)])
    assert is_abnormal_independence_option(unrelated_dag, 0, 2) is True
    binary_yes = build_independence_test_binary(chain_scm, option_order_rng=random.Random(0), target_answer_label="Yes")
    binary_no = build_independence_test_binary(chain_scm, option_order_rng=random.Random(0), target_answer_label="No")
    assert binary_yes.correct_options == "Yes"
    assert binary_yes.option_results[0].independent is True
    assert binary_no.correct_options == "No"
    assert binary_no.option_results[0].independent is False
    rich_dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    rich_scm = RLCausalSCM(rich_dag, seed=7)
    for label in OPTION_LABELS:
        result = build_independence_test_multi_choice(rich_scm, option_order_rng=random.Random(1), target_answer_label=label)
        assert result is not None and result.correct_options == label
        assert _format_multi_choice_correct_option(result.option_results) == result.correct_options
    multi_answer_result = build_independence_test_multi_answer(rich_scm, option_order_rng=random.Random(3))
    assert len(multi_answer_result.option_results) == 4
    assert _format_correct_options(multi_answer_result.option_results) == multi_answer_result.correct_options
    print("Independence Test query sampling checks passed.")
