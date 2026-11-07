"""joint_prob_query_sampling.py"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx

from data_generation.dataset_utils.association_shared import compute_joint_prob, is_structurally_related
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class JointProbResult:
    target_nodes: List[int]
    probability: float
    verification_ok: bool

    def to_json_dict(self) -> Dict:
        return {
            "target_nodes": self.target_nodes,
            "probability": self.probability,
            "verification_ok": self.verification_ok,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "JointProbResult":
        return cls(
            target_nodes=[int(node) for node in data["target_nodes"]],
            probability=float(data["probability"]),
            verification_ok=bool(data["verification_ok"]),
        )


def compute_joint_prob_query(scm: SCMBase, *, target_nodes: List[int]) -> JointProbResult:
    return JointProbResult(
        target_nodes=list(target_nodes),
        probability=compute_joint_prob(scm, target_nodes),
        verification_ok=True,
    )


def is_abnormal_joint_query(dag: nx.DiGraph, target_nodes: List[int]) -> bool:
    return not is_structurally_related(dag, target_nodes[0], target_nodes[1])


def sample_joint_prob_query(scm: SCMBase, rng: random.Random) -> Tuple[Dict, JointProbResult]:
    pair = list(rng.choice(list(itertools.combinations(scm.nodes, 2))))
    query_spec = {"target_nodes": pair}
    return query_spec, compute_joint_prob_query(scm, **query_spec)


if __name__ == "__main__":
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3), (4, 5)])
    cpts = {
        0: {"parents": [], "table": {(): 0.3}},
        1: {"parents": [], "table": {(): 0.6}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.7, (1, 1): 0.9}},
        3: {"parents": [2], "table": {(0,): 0.2, (1,): 0.8}},
        4: {"parents": [], "table": {(): 0.4}},
        5: {"parents": [4], "table": {(0,): 0.3, (1,): 0.9}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=1)

    result = compute_joint_prob_query(scm, target_nodes=[0, 1])
    assert math.isclose(result.probability, 0.18, abs_tol=1e-9)
    assert JointProbResult.from_json_dict(result.to_json_dict()) == result

    assert is_abnormal_joint_query(dag, [0, 3]) is False
    assert is_abnormal_joint_query(dag, [0, 4]) is True

    query_spec, sampled_result = sample_joint_prob_query(scm, random.Random(7))
    assert sampled_result.target_nodes == query_spec["target_nodes"]
    expected_sampled = compute_joint_prob(scm, query_spec["target_nodes"])
    assert math.isclose(sampled_result.probability, expected_sampled, abs_tol=1e-9)
    print("Joint Prob query sampling checks passed.")
