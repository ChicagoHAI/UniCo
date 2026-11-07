"""conditional_prob_query_sampling.py"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from data_generation.dataset_utils.association_shared import (
    compute_conditional_prob,
    compute_markov_blanket,
    sample_conditioning_subset,
)
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class ConditionalProbResult:
    target: int
    evidence: Dict[int, int]
    probability: float
    verification_ok: bool

    def to_json_dict(self) -> Dict:
        return {
            "target": self.target,
            "evidence": self.evidence,
            "probability": self.probability,
            "verification_ok": self.verification_ok,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "ConditionalProbResult":
        return cls(
            target=int(data["target"]),
            evidence={int(k): int(v) for k, v in data["evidence"].items()},
            probability=float(data["probability"]),
            verification_ok=bool(data["verification_ok"]),
        )


def compute_conditional_prob_query(
    scm: SCMBase,
    *,
    target: int,
    evidence: Dict[int, int],
) -> ConditionalProbResult:
    return ConditionalProbResult(
        target=target,
        evidence=dict(evidence),
        probability=compute_conditional_prob(scm, target, evidence),
        verification_ok=True,
    )


def is_abnormal_conditional_query(dag: nx.DiGraph, target: int, evidence: Dict[int, int]) -> bool:
    target_ancestors = nx.ancestors(dag, target)
    return all(node not in target_ancestors for node in evidence)


def sample_conditional_prob_query(
    scm: SCMBase,
    rng: random.Random,
    *,
    max_conditioning: int = 2,
) -> Tuple[Dict, ConditionalProbResult]:
    candidate_targets = [node for node in scm.nodes if nx.ancestors(scm.dag, node)] or list(scm.nodes)
    target = rng.choice(candidate_targets)
    related_nodes = sorted(
        (
            set(compute_markov_blanket(scm.dag, target))
            | nx.ancestors(scm.dag, target)
            | nx.descendants(scm.dag, target)
        ) - {target}
    )
    if not related_nodes:
        related_nodes = [node for node in scm.nodes if node != target]
    evidence_nodes = sample_conditioning_subset(rng, related_nodes, max_conditioning, allow_empty=False)
    query_spec = {
        "target": target,
        "evidence": {node: rng.randint(0, 1) for node in evidence_nodes},
    }
    return query_spec, compute_conditional_prob_query(scm, **query_spec)


if __name__ == "__main__":
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    cpts = {
        0: {"parents": [], "table": {(): 0.3}},
        1: {"parents": [], "table": {(): 0.6}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.7, (1, 1): 0.9}},
        3: {"parents": [2], "table": {(0,): 0.2, (1,): 0.8}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=1)

    result = compute_conditional_prob_query(scm, target=2, evidence={0: 1, 1: 0})
    assert math.isclose(result.probability, 0.7, abs_tol=1e-9)
    assert ConditionalProbResult.from_json_dict(result.to_json_dict()) == result

    assert is_abnormal_conditional_query(dag, 2, {0: 1}) is False
    assert is_abnormal_conditional_query(dag, 2, {3: 1}) is True

    query_spec, sampled_result = sample_conditional_prob_query(scm, random.Random(7), max_conditioning=2)
    assert sampled_result.target == query_spec["target"]
    assert sampled_result.evidence == query_spec["evidence"]
    expected_sampled = compute_conditional_prob(scm, query_spec["target"], query_spec["evidence"])
    assert math.isclose(sampled_result.probability, expected_sampled, abs_tol=1e-9)
    print("Conditional Prob query sampling checks passed.")
