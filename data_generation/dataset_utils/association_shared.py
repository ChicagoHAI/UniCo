"""
Shared observational-query helpers for standalone association-derived families.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from pgmpy.factors.discrete import DiscreteFactor

from data_generation.scm_mechanism_sampling import SCMBase


@dataclass
class QueryQualityAssessment:
    quality_ok: bool
    quality_score: float
    quality_reason: str


def factor_prob(factor: DiscreteFactor, assignments: Dict[int, int]) -> float:
    factor_variables = list(factor.variables)
    if set(assignments) != set(factor_variables):
        raise ValueError(
            f"Assignments {sorted(assignments)} do not match factor scope {factor_variables}"
        )
    factor_index = tuple(assignments[var] for var in factor_variables)
    return float(factor.values[factor_index])


def compute_marginal_prob(scm: SCMBase, target: int) -> float:
    factor = scm.query([target], evidence=None, show_progress=False)
    return factor_prob(factor, {target: 1})


def compute_conditional_prob(scm: SCMBase, target: int, evidence: Dict[int, int]) -> float:
    if target in evidence:
        raise ValueError("Conditional probability target cannot also appear in evidence")
    factor = scm.query([target], evidence=evidence, show_progress=False)
    return factor_prob(factor, {target: 1})


def compute_joint_prob(scm: SCMBase, target_nodes: List[int]) -> float:
    if len(target_nodes) < 2:
        raise ValueError("Joint probability requires at least two target nodes")
    factor = scm.query(target_nodes, evidence=None, show_progress=False)
    return factor_prob(factor, {node: 1 for node in target_nodes})


def compute_observed_difference(scm: SCMBase, source: int, target: int) -> float:
    if source == target:
        raise ValueError("Observed difference requires distinct source and target nodes")
    prob_target_given_source_1 = compute_conditional_prob(scm, target, {source: 1})
    prob_target_given_source_0 = compute_conditional_prob(scm, target, {source: 0})
    return prob_target_given_source_1 - prob_target_given_source_0


def compute_markov_blanket(dag: nx.DiGraph, target: int) -> List[int]:
    parents = set(dag.predecessors(target))
    children = set(dag.successors(target))
    spouses = set()
    for child in children:
        spouses.update(dag.predecessors(child))
    blanket = (parents | children | spouses) - {target}
    return sorted(blanket)


def compute_markov_blanket_membership(dag: nx.DiGraph, target: int, candidate: int) -> Tuple[bool, List[int]]:
    if target == candidate:
        raise ValueError("Markov blanket membership requires distinct target and candidate nodes")
    blanket = compute_markov_blanket(dag, target)
    return candidate in set(blanket), blanket


def compute_independence_violation_score(
    scm: SCMBase,
    left_node: int,
    right_node: int,
    conditioning_nodes: List[int],
) -> float:
    conditioning_assignments = list(itertools.product([0, 1], repeat=len(conditioning_nodes)))
    if not conditioning_assignments:
        conditioning_assignments = [()]

    max_violation = 0.0
    for conditioning_values in conditioning_assignments:
        evidence = {node: value for node, value in zip(conditioning_nodes, conditioning_values)}
        joint_factor = scm.query([left_node, right_node], evidence=evidence, show_progress=False)
        left_factor = scm.query([left_node], evidence=evidence, show_progress=False)
        right_factor = scm.query([right_node], evidence=evidence, show_progress=False)

        for left_value, right_value in itertools.product([0, 1], repeat=2):
            joint_prob = factor_prob(joint_factor, {left_node: left_value, right_node: right_value})
            left_prob = factor_prob(left_factor, {left_node: left_value})
            right_prob = factor_prob(right_factor, {right_node: right_value})
            max_violation = max(max_violation, abs(joint_prob - (left_prob * right_prob)))
    return max_violation


def compute_independence_test(
    scm: SCMBase,
    left_node: int,
    right_node: int,
    conditioning_nodes: List[int],
    *,
    atol: float = 1e-8,
) -> bool:
    if left_node == right_node:
        raise ValueError("Independence test requires distinct left/right nodes")
    if left_node in conditioning_nodes or right_node in conditioning_nodes:
        raise ValueError("Conditioning nodes must exclude the tested nodes")
    return compute_independence_violation_score(scm, left_node, right_node, conditioning_nodes) < atol


def is_structurally_related(dag: nx.DiGraph, left_node: int, right_node: int) -> bool:
    if nx.has_path(dag, left_node, right_node) or nx.has_path(dag, right_node, left_node):
        return True
    left_ancestors = nx.ancestors(dag, left_node)
    right_ancestors = nx.ancestors(dag, right_node)
    if left_ancestors & right_ancestors:
        return True
    left_descendants = nx.descendants(dag, left_node)
    right_descendants = nx.descendants(dag, right_node)
    return bool(left_descendants & right_descendants)


def assess_probability_query(probability: float, *, num_ancestors: int, min_prob: float, max_prob: float) -> QueryQualityAssessment:
    quality_ok = num_ancestors > 0 and min_prob <= probability <= max_prob
    quality_score = min(probability, 1.0 - probability) + 0.05 * num_ancestors
    return QueryQualityAssessment(quality_ok, quality_score, f"ancestors={num_ancestors}, probability={probability:.4f}")


def assess_observed_difference_query(value: float, *, min_observed_difference: float, structural_related: bool) -> QueryQualityAssessment:
    quality_ok = structural_related and abs(value) >= min_observed_difference
    quality_score = abs(value) + (0.2 if structural_related else 0.0)
    return QueryQualityAssessment(quality_ok, quality_score, f"structural_related={structural_related}, observed_difference={value:.4f}")


def assess_joint_query(probability: float, *, min_prob: float, max_prob: float, structural_related: bool) -> QueryQualityAssessment:
    quality_ok = structural_related and min_prob <= probability <= max_prob
    quality_score = min(probability, 1.0 - probability) + (0.15 if structural_related else 0.0)
    return QueryQualityAssessment(quality_ok, quality_score, f"structural_related={structural_related}, probability={probability:.4f}")


def assess_conditional_query(
    probability: float,
    unconditional_probability: float,
    *,
    min_prob: float,
    max_prob: float,
    min_conditional_gap: float,
) -> QueryQualityAssessment:
    gap = abs(probability - unconditional_probability)
    quality_ok = min_prob <= probability <= max_prob and gap >= min_conditional_gap
    quality_score = min(probability, 1.0 - probability) + gap
    return QueryQualityAssessment(quality_ok, quality_score, f"probability={probability:.4f}, unconditional={unconditional_probability:.4f}, gap={gap:.4f}")


def assess_independence_query(
    independent: bool,
    *,
    violation: float,
    conditioning_count: int,
    min_independence_violation: float,
    structurally_related: bool,
) -> QueryQualityAssessment:
    if independent:
        quality_ok = conditioning_count > 0 or structurally_related
        quality_score = 0.5 + 0.05 * conditioning_count
    else:
        quality_ok = violation >= min_independence_violation
        quality_score = violation + 0.05 * conditioning_count
    return QueryQualityAssessment(
        quality_ok,
        quality_score,
        f"independent={independent}, violation={violation:.4f}, conditioning={conditioning_count}",
    )


def assess_markov_blanket_query(
    in_markov_blanket: bool,
    blanket: List[int],
    *,
    num_children: int,
    num_parents: int,
    min_markov_blanket_size: int,
) -> QueryQualityAssessment:
    quality_ok = len(blanket) >= min_markov_blanket_size
    quality_score = len(blanket) + 0.25 * num_children + 0.1 * num_parents + (0.25 if in_markov_blanket else 0.15)
    return QueryQualityAssessment(
        quality_ok,
        quality_score,
        f"blanket_size={len(blanket)}, candidate_in_blanket={in_markov_blanket}, num_parents={num_parents}, num_children={num_children}",
    )


def sample_conditioning_subset(rng: random.Random, candidate_nodes: List[int], max_conditioning: int, *, allow_empty: bool) -> List[int]:
    if not candidate_nodes:
        return []
    lower = 0 if allow_empty else 1
    size = rng.randint(lower, min(max_conditioning, len(candidate_nodes)))
    return sorted(rng.sample(candidate_nodes, size))
