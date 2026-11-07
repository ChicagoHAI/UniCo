"""observed_difference_query_sampling.py"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from data_generation.dataset_utils.association_shared import compute_observed_difference, is_structurally_related
from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM


@dataclass
class ObservedDifferenceResult:
    source: int
    target: int
    observed_difference: float
    verification_ok: bool

    def to_json_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "observed_difference": self.observed_difference,
            "verification_ok": self.verification_ok,
        }

    @classmethod
    def from_json_dict(cls, data: Dict) -> "ObservedDifferenceResult":
        return cls(
            source=int(data["source"]),
            target=int(data["target"]),
            observed_difference=float(data["observed_difference"]),
            verification_ok=bool(data["verification_ok"]),
        )


def compute_observed_difference_query(scm: SCMBase, *, source: int, target: int) -> ObservedDifferenceResult:
    return ObservedDifferenceResult(
        source=source,
        target=target,
        observed_difference=compute_observed_difference(scm, source, target),
        verification_ok=True,
    )


def is_abnormal_observed_difference_query(dag: nx.DiGraph, source: int, target: int) -> bool:
    return not is_structurally_related(dag, source, target)


def sample_observed_difference_query(scm: SCMBase, rng: random.Random) -> Tuple[Dict, ObservedDifferenceResult]:
    source, target = rng.choice(list(itertools.permutations(scm.nodes, 2)))
    query_spec = {"source": source, "target": target}
    return query_spec, compute_observed_difference_query(scm, **query_spec)


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

    result = compute_observed_difference_query(scm, source=0, target=3)
    expected_difference = compute_observed_difference(scm, 0, 3)
    expected_given_source_one = (1 - 0.82) * 0.2 + 0.82 * 0.8
    expected_given_source_zero = (1 - 0.16) * 0.2 + 0.16 * 0.8
    manual_expected_difference = expected_given_source_one - expected_given_source_zero
    assert math.isclose(manual_expected_difference, 0.396, abs_tol=1e-9)
    assert math.isclose(expected_difference, manual_expected_difference, abs_tol=1e-9)
    assert math.isclose(result.observed_difference, expected_difference, abs_tol=1e-9)
    assert ObservedDifferenceResult.from_json_dict(result.to_json_dict()) == result

    assert is_abnormal_observed_difference_query(dag, 0, 3) is False
    assert is_abnormal_observed_difference_query(dag, 0, 4) is True

    query_spec, sampled_result = sample_observed_difference_query(scm, random.Random(7))
    assert sampled_result.source == query_spec["source"]
    assert sampled_result.target == query_spec["target"]
    expected_sampled = compute_observed_difference(scm, query_spec["source"], query_spec["target"])
    assert abs(sampled_result.observed_difference - expected_sampled) < 1e-9
    print("Observed Difference query sampling checks passed.")
