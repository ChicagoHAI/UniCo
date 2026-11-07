"""marginal_prob_query_sampling.py"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from data_generation.dataset_utils.association_shared import compute_marginal_prob
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class MarginalProbResult:
    target: int
    probability: float
    verification_ok: bool

    def to_json_dict(self) -> Dict:
        return {
            "target": self.target,
            "probability": self.probability,
            "verification_ok": self.verification_ok,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "MarginalProbResult":
        return cls(
            target=int(data["target"]),
            probability=float(data["probability"]),
            verification_ok=bool(data["verification_ok"]),
        )


def compute_marginal_prob_query(scm: SCMBase, *, target: int) -> MarginalProbResult:
    return MarginalProbResult(
        target=target,
        probability=compute_marginal_prob(scm, target),
        verification_ok=True,
    )


def is_abnormal_target(dag: nx.DiGraph, target: int) -> bool:
    return len(nx.ancestors(dag, target)) == 0


def sample_marginal_prob_query(
    scm: SCMBase,
    rng: random.Random,
) -> Tuple[Dict, MarginalProbResult]:
    query_spec = {"target": rng.choice(scm.nodes)}
    return query_spec, compute_marginal_prob_query(scm, **query_spec)


if __name__ == "__main__":
    print("Test 1: exact root marginal matches the CPT")
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    cpts = {
        0: {"parents": [], "table": {(): 0.3}},
        1: {"parents": [], "table": {(): 0.6}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.7, (1, 1): 0.9}},
        3: {"parents": [2], "table": {(0,): 0.2, (1,): 0.8}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=1)

    root_result = compute_marginal_prob_query(scm, target=0)
    assert math.isclose(root_result.probability, 0.3, abs_tol=1e-9)

    print("Test 2: exact two-parent marginal matches manual total-probability calculation")
    expected_target_two = (
        (1 - 0.3) * (1 - 0.6) * 0.1
        + (1 - 0.3) * 0.6 * 0.2
        + 0.3 * (1 - 0.6) * 0.7
        + 0.3 * 0.6 * 0.9
    )
    target_two_result = compute_marginal_prob_query(scm, target=2)
    assert math.isclose(expected_target_two, 0.358, abs_tol=1e-12)
    assert math.isclose(target_two_result.probability, expected_target_two, abs_tol=1e-9)

    print("Test 3: downstream marginal composes correctly through the intermediate node")
    expected_target_three = ((1 - expected_target_two) * 0.2) + (expected_target_two * 0.8)
    target_three_result = compute_marginal_prob_query(scm, target=3)
    assert math.isclose(expected_target_three, 0.4148, abs_tol=1e-12)
    assert math.isclose(target_three_result.probability, expected_target_three, abs_tol=1e-9)

    print("Test 4: result serialization preserves the exact computed value")
    assert MarginalProbResult.from_json_dict(target_two_result.to_json_dict()) == target_two_result

    print("Test 5: abnormal targets are exactly the nodes with no ancestors")
    assert is_abnormal_target(dag, 0) is True
    assert is_abnormal_target(dag, 1) is True
    assert is_abnormal_target(dag, 2) is False
    assert is_abnormal_target(dag, 3) is False

    print("Test 6: sampler returns a valid target and exact marginal result")
    query_spec, sampled_result = sample_marginal_prob_query(scm, random.Random(7))
    assert sampled_result.target in scm.nodes
    assert query_spec["target"] == sampled_result.target
    assert math.isclose(
        sampled_result.probability,
        compute_marginal_prob_query(scm, target=sampled_result.target).probability,
        abs_tol=1e-9,
    )
    print("Marginal Prob query sampling checks passed.")
