"""
ate_question_assembly.py

ATE Question Assembly: text rendering, condition variants, and full question
assembly from pre-computed ATE results.

Given an RLCausalSCM and a pre-computed ATEResult (from ate_query_sampling),
assembles three question variants with different levels of probability
conditions:
  1) Full SCM: original graph + CPDs for all variables
  2) Surgery-only: CPDs remaining after graph surgery (excluding treatment's CPD)
  3) Surgery + ancestor pruning: further prune non-ancestors of outcome in the
     mutilated graph

Only RLCausalSCM is supported as the input SCM class.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx
from pgmpy.factors.discrete import DiscreteFactor

from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM
from data_generation.ate.ate_query_sampling import ATEResult


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ConditionVariant:
    """One of three condition variants for a question."""
    variant_name: str       # "full" | "surgery_only" | "surgery_ancestor_prune"
    conditions_text: str    # rendered CPDs for this variant
    nodes_kept: List[int]   # nodes whose CPDs appear in conditions_text
    nodes_pruned: List[int] # all other nodes (CPDs not shown)


@dataclass
class AssembledQuestion:
    """A complete assembled question for one (treatment, outcome, variant) triple."""
    treatment: int
    outcome: int
    variant_name: str
    ate_ground_truth: float
    is_abnormal: bool
    graph_variant: int          # 1, 2, or 3
    query_variant: int          # 1, 2, or 3
    graph_text: str             # full original DAG edges (always same across variants)
    conditions_text: str        # CPDs for this variant
    query_text: str             # ATE query expression
    full_question_text: str     # combined: graph + conditions + query + answer format
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


# ============================================================
# Text Rendering
# ============================================================

def _node_label(node: int) -> str:
    """Format integer node ID as a variable name: 0 -> 'X0', 1 -> 'X1', etc."""
    return f"X{node}"


def render_dag_edges(dag: nx.DiGraph) -> str:
    """
    Render DAG edges as a sorted, human-readable string.

    Edges are sorted by (source, target) for deterministic output.
    Example: "X0 -> X2, X1 -> X2, X2 -> X3"
    """
    edges = sorted(dag.edges())
    if not edges:
        return "(no edges)"
    return ", ".join(
        f"{_node_label(u)} -> {_node_label(v)}" for u, v in edges
    )


def render_node_cpt(scm: SCMBase, node: int, *, decimal_places: int = 2) -> str:
    """
    Render a single node's CPT from scm.cpts in compact format.

    Root node example:
        P(X0 = 1) = 0.4213

    Conditional node example:
        P(X2 = 1 | X0, X1):
          X0=0, X1=0: 0.1234
          X0=0, X1=1: 0.5678
          X0=1, X1=0: 0.3456
          X0=1, X1=1: 0.8901
    """
    cpt_entry = scm.cpts[node]
    parents = cpt_entry['parents']
    table = cpt_entry['table']
    nl = _node_label(node)

    if not parents:
        p = table[()]
        return f"P({nl} = 1) = {p:.{decimal_places}f}"

    parent_labels = [_node_label(p) for p in parents]
    header = f"P({nl} = 1 | {', '.join(parent_labels)}):"

    lines = [header]
    configs = list(itertools.product([0, 1], repeat=len(parents)))
    for config in configs:
        p = table[config]
        assignments = ", ".join(
            f"{pl}={v}" for pl, v in zip(parent_labels, config)
        )
        lines.append(f"  {assignments}: {p:.{decimal_places}f}")

    return "\n".join(lines)


# ============================================================
# Condition Variants
# ============================================================

def _topo_sorted_subset(dag: nx.DiGraph, nodes: Set[int]) -> List[int]:
    """
    Return nodes in topological order, restricted to a given subset.

    Builds a temporary subgraph of dag restricted to `nodes` and returns
    its topological sort.
    """
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(
        (u, v) for u, v in dag.edges() if u in nodes and v in nodes
    )
    return list(nx.topological_sort(subgraph))


def condition_variant_full(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    """
    Variant 1 — Full SCM: CPDs for ALL nodes from the original model.

    Presents all CPDs in topological order of the original DAG.
    """
    all_nodes = set(scm.nodes)
    topo_nodes = _topo_sorted_subset(scm.dag, all_nodes)

    cpt_blocks = [render_node_cpt(scm, node, decimal_places=decimal_places) for node in topo_nodes]
    conditions_text = "\n".join(cpt_blocks)

    return ConditionVariant(
        variant_name="full",
        conditions_text=conditions_text,
        nodes_kept=sorted(all_nodes),
        nodes_pruned=[],
    )


def condition_variant_surgery_only(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    """
    Variant 2 — Surgery-only: CPDs for all nodes EXCEPT treatment.

    After graph surgery, the treatment node's mechanism is replaced by
    the intervention, so its CPD is omitted. Non-treatment CPDs are
    identical between the original and mutilated models, so we render
    from scm.cpts.
    """
    all_nodes = set(scm.nodes)
    cpd_nodes = all_nodes - {treatment}
    topo_nodes = _topo_sorted_subset(scm.dag, cpd_nodes)

    cpt_blocks = [render_node_cpt(scm, node, decimal_places=decimal_places) for node in topo_nodes]
    conditions_text = "\n".join(cpt_blocks)

    return ConditionVariant(
        variant_name="surgery_only",
        conditions_text=conditions_text,
        nodes_kept=sorted(cpd_nodes),
        nodes_pruned=sorted(all_nodes - cpd_nodes),
    )


def condition_variant_surgery_ancestor_prune(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    """
    Variant 3 — Surgery + ancestor pruning.

    After do(treatment), keep only ancestors of outcome in the mutilated
    graph (plus outcome itself). Treatment is excluded from CPDs since
    the intervention replaces its mechanism. Non-ancestor nodes are pruned.

    Steps:
    1. mutilated = model.do([treatment])
    2. anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])
       → includes outcome + all its ancestors in the mutilated graph
    3. cpd_nodes = anc_graph.nodes() - {treatment}
    4. Render CPDs for cpd_nodes from scm.cpts in topo order.
    """
    mutilated = scm.pgmpy_model.do([treatment])
    anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])  # includes outcome + its ancestors
    anc_nodes = set(anc_graph.nodes())

    cpd_nodes = anc_nodes - {treatment}
    all_nodes = set(scm.nodes)
    topo_nodes = _topo_sorted_subset(scm.dag, cpd_nodes)

    cpt_blocks = [render_node_cpt(scm, node, decimal_places=decimal_places) for node in topo_nodes]
    conditions_text = "\n".join(cpt_blocks)

    return ConditionVariant(
        variant_name="surgery_ancestor_prune",
        conditions_text=conditions_text,
        nodes_kept=sorted(cpd_nodes),
        nodes_pruned=sorted(all_nodes - cpd_nodes),
    )


def generate_condition_variants(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    *,
    decimal_places: int = 2,
) -> List[ConditionVariant]:
    """
    Generate all three condition variants for a (treatment, outcome) pair.

    Returns:
        List of three ConditionVariant objects:
            [0] full, [1] surgery_only, [2] surgery_ancestor_prune
    """
    return [
        condition_variant_full(scm, treatment, outcome, decimal_places=decimal_places),
        condition_variant_surgery_only(scm, treatment, outcome, decimal_places=decimal_places),
        condition_variant_surgery_ancestor_prune(scm, treatment, outcome, decimal_places=decimal_places),
    ]


# ============================================================
# Question Assembly
# ============================================================

def _render_query_text(treatment: int, outcome: int, *, query_variant: int) -> str:
    """
    Render the ATE query expression.

    Args:
        treatment: Treatment node ID.
        outcome: Outcome node ID.
        query_variant: Which verbalization style to use (1, 2, or 3).
    """
    t_label = _node_label(treatment)
    o_label = _node_label(outcome)
    line1 = f"What is the Average Treatment Effect (ATE) of {t_label} on {o_label}?"

    if query_variant == 1:
        line2 = (
            f"That is, how does the probability of {o_label} = 1 change "
            f"if we force {t_label} = 1 compared to forcing {t_label} = 0?"
        )
    elif query_variant == 2:
        line2 = (
            f"That is, what is the difference in P({o_label} = 1) when "
            f"{t_label} is set to 1 versus set to 0, regardless of how "
            f"{t_label} would naturally arise?"
        )
    elif query_variant == 3:
        line2 = (
            f"In other words, how much does intervening to change "
            f"{t_label} from 0 to 1 affect the probability that "
            f"{o_label} = 1?"
        )
    else:
        raise ValueError(f"Unknown query_variant: {query_variant}")

    return f"{line1}\n{line2}"


ANSWER_FORMAT_TEMPLATE = (
    "Think step-by-step to solve this causal reasoning question. "
    "Your final answer should be rounded to four decimal places "
    "and included inside \\boxed{}, as below: \\boxed{<your final answer>}."
)


def _render_graph_text(scm: SCMBase, graph_variant: int) -> str:
    """
    Render the full original causal graph section.

    Args:
        scm: The SCM containing the DAG.
        graph_variant: Which verbalization style to use (1, 2, or 3).
    """
    node_labels = ", ".join(_node_label(n) for n in scm.nodes)
    edges = sorted(scm.dag.edges())

    if graph_variant == 1:
        # Preamble + edge list
        edges_str = render_dag_edges(scm.dag)
        return (
            "Imagine a self-contained, hypothetical world with only the "
            "following conditions, and without any unmentioned factors or "
            "causal relationships:\n"
            f"Given the following causal graph among variables {node_labels}:\n"
            f"{edges_str}"
        )
    elif graph_variant == 2:
        # Sentence-per-edge
        edge_sentences = "\n".join(
            f"{_node_label(u)} directly causes {_node_label(v)}."
            for u, v in edges
        )
        return (
            f"Given the following causal relationships among variables {node_labels}:\n"
            f"{edge_sentences}\n"
            "These are the only direct causal relationships in this world. "
            "No other causal relationships or unobserved factors exist."
        )
    elif graph_variant == 3:
        # Parent-set
        parent_lines = []
        for n in scm.nodes:
            parents = sorted(scm.dag.predecessors(n))
            if not parents:
                parent_lines.append(f"{_node_label(n)} does not have any parent;")
            else:
                parent_set = ", ".join(_node_label(p) for p in parents)
                parent_lines.append(f"{_node_label(n)} has parents: {{{parent_set}}}")
        parent_block = "\n".join(parent_lines)
        return (
            f"Consider variables {node_labels}, where each variable is a function "
            "only of its direct causes (parents), listed below:\n"
            f"{parent_block}\n"
            "The parent sets listed above are exhaustive--each variable is "
            "affected only by its specified parents, and no unobserved or "
            "latent variables exist in this system."
        )
    else:
        raise ValueError(f"Unknown graph_variant: {graph_variant}")


def assemble_question(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    variant: ConditionVariant,
    ate_result: ATEResult,
    *,
    graph_variant: int,
    query_variant: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> AssembledQuestion:
    """
    Assemble a complete question from a condition variant and ATE result.

    The question has four parts:
    1. Full causal graph (original DAG edges, never mutilated)
    2. Probability conditions (from the variant)
    3. ATE query expression
    4. Answer format template
    """
    graph_text = _render_graph_text(scm, graph_variant)
    query_text = _render_query_text(treatment, outcome, query_variant=query_variant)

    full_question_text = (
        f"{graph_text}\n\n"
        f"The following probability relationships hold:\n"
        f"{variant.conditions_text}\n\n"
        f"{query_text}\n\n"
        f"{ANSWER_FORMAT_TEMPLATE}"
    )

    metadata = {
        "num_nodes": len(scm.nodes),
        "num_edges": scm.dag.number_of_edges(),
        "dag_edges": sorted(list(scm.dag.edges())),
        "treatment": treatment,
        "outcome": outcome,
        "variant_name": variant.variant_name,
        "is_abnormal": ate_result.is_abnormal,
        "ate_surgery": ate_result.ate_surgery,
        "ate_causal_query": ate_result.ate_causal_query,
        "verification_ok": ate_result.verification_ok,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "treatment_outcome_depth": treatment_outcome_depth,
        "effective_prob_conds": effective_prob_conds,
        "has_naive_conditioning": has_naive_conditioning,
    }

    return AssembledQuestion(
        treatment=treatment,
        outcome=outcome,
        variant_name=variant.variant_name,
        ate_ground_truth=ate_result.ate_surgery,
        is_abnormal=ate_result.is_abnormal,
        graph_variant=graph_variant,
        query_variant=query_variant,
        graph_text=graph_text,
        conditions_text=variant.conditions_text,
        query_text=query_text,
        full_question_text=full_question_text,
        nodes_kept=variant.nodes_kept,
        nodes_pruned=variant.nodes_pruned,
        metadata=metadata,
    )


def assemble_all_questions(
    scm: SCMBase,
    treatment: int,
    outcome: int,
    ate_result: ATEResult,
    *,
    decimal_places: int = 2,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledQuestion]:
    """
    Top-level function: generate all three condition variants and assemble
    three questions for a single (treatment, outcome) pair.

    For each example, a (graph_variant, query_variant) pair is sampled
    uniformly from {1,2,3} x {1,2,3} using example_seed. The same pair
    is used across all three condition variants.

    Args:
        scm: The SCM containing the DAG and CPTs.
        treatment: Treatment node ID.
        outcome: Outcome node ID.
        ate_result: Pre-computed ATEResult from compute_ate().
        decimal_places: Number of decimal places for CPT rendering.
        example_seed: Seed for deterministic graph/query variant sampling.
        treatment_outcome_depth: Pre-computed depth from treatment to outcome.
        effective_prob_conds: Pre-computed effective probability conditions.
        has_naive_conditioning: Pre-computed naive conditioning flag.

    Returns:
        List of three AssembledQuestion objects (one per variant).
    """
    rng = random.Random(example_seed)
    graph_variant = rng.randint(1, 3)
    query_variant = rng.randint(1, 3)

    variants = generate_condition_variants(scm, treatment, outcome, decimal_places=decimal_places)

    return [
        assemble_question(
            scm, treatment, outcome, variant, ate_result,
            graph_variant=graph_variant,
            query_variant=query_variant,
            treatment_outcome_depth=treatment_outcome_depth,
            effective_prob_conds=effective_prob_conds,
            has_naive_conditioning=has_naive_conditioning,
        )
        for variant in variants
    ]


# ============================================================
# TEST SUITE
# ============================================================
if __name__ == "__main__":
    import math
    from data_generation.ate.ate_query_sampling import compute_ate
    from data_generation.dataset_utils.difficulty_metrics import (
        compute_treatment_outcome_depth as _tod,
        compute_effective_prob_conds as _epc,
        has_naive_conditioning as _ns,
    )

    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing ATE Question Assembly ===\n")

    # ==================================================================
    # Group 1: Condition variants — base cases
    # ==================================================================
    print("--- Group 1: Condition Variants (Base Cases) ---\n")

    # Setup: chain and confounder SCMs
    chain_dag = nx.DiGraph()
    chain_dag.add_edge(0, 1)
    chain_scm = RLCausalSCM(chain_dag, seed=42)

    conf_dag = nx.DiGraph()
    conf_dag.add_edges_from([(2, 0), (2, 1), (0, 1)])
    conf_scm = RLCausalSCM(conf_dag, seed=42)

    indep_dag = nx.DiGraph()
    indep_dag.add_nodes_from([0, 1, 2])
    indep_dag.add_edge(0, 1)
    indep_scm = RLCausalSCM(indep_dag, seed=42)

    # Test 1: Variants for chain 0→1, ATE 0→1
    print("Test 1: Variants for chain 0→1, ATE 0→1")
    chain_variants = generate_condition_variants(chain_scm, 0, 1)
    assert len(chain_variants) == 3
    for v in chain_variants:
        assert isinstance(v, ConditionVariant)
        assert v.conditions_text  # non-empty

    # Full: all nodes kept
    assert chain_variants[0].nodes_kept == [0, 1]
    assert chain_variants[0].nodes_pruned == []
    # Surgery-only: treatment 0 pruned
    assert chain_variants[1].nodes_kept == [1]
    assert chain_variants[1].nodes_pruned == [0]
    # Surgery+prune: same as surgery-only for chain (0 is ancestor of 1 in
    # mutilated graph, so ancestor_prune keeps {0, 1} - {0} = {1})
    assert chain_variants[2].nodes_kept == [1]
    assert chain_variants[2].nodes_pruned == [0]
    print("   -> PASSED\n")

    # Test 2: Variants for confounder, ATE 0→1
    print("Test 2: Variants for confounder 2→0, 2→1, 0→1, ATE 0→1")
    conf_variants = generate_condition_variants(conf_scm, 0, 1)
    assert conf_variants[0].nodes_kept == [0, 1, 2]
    assert conf_variants[0].nodes_pruned == []
    assert conf_variants[1].nodes_kept == [1, 2]
    assert conf_variants[1].nodes_pruned == [0]
    # Surgery+prune: surgery on 0 removes edge 2→0. In mutilated graph,
    # ancestors of 1 = {0, 2} (from 0→1 and 2→1). anc_nodes = {0, 1, 2}.
    # cpd_nodes = {0, 1, 2} - {0} = {1, 2}. Same as surgery-only.
    assert conf_variants[2].nodes_kept == [1, 2]
    assert conf_variants[2].nodes_pruned == [0]
    print("   -> PASSED\n")

    # Test 3: Variants for independent node, ATE 0→2
    print("Test 3: Variants for independent node, ATE 0→2")
    indep_variants = generate_condition_variants(indep_scm, 0, 2)
    assert indep_variants[0].nodes_kept == [0, 1, 2]
    assert indep_variants[0].nodes_pruned == []
    assert indep_variants[1].nodes_kept == [1, 2]
    assert indep_variants[1].nodes_pruned == [0]
    # Surgery+prune: surgery on 0 removes nothing (0 has no parents).
    # Ancestors of 2 in mutilated = {} (2 is isolated). anc_nodes = {2}.
    # cpd_nodes = {2} - {0} = {2}. Node 1 pruned.
    assert indep_variants[2].nodes_kept == [2]
    assert indep_variants[2].nodes_pruned == [0, 1]
    print("   -> PASSED\n")

    # Test 4: Variants for abnormal chain, ATE 1→0
    print("Test 4: Variants for abnormal chain 0→1, ATE 1→0")
    abnormal_chain_variants = generate_condition_variants(chain_scm, 1, 0)
    assert len(abnormal_chain_variants) == 3
    for v in abnormal_chain_variants:
        assert isinstance(v, ConditionVariant)
    print("   -> PASSED\n")

    # Test 5: Variants for abnormal confounder, ATE 1→2
    print("Test 5: Variants for abnormal confounder, ATE 1→2")
    abnormal_conf_variants = generate_condition_variants(conf_scm, 1, 2)
    assert len(abnormal_conf_variants) == 3
    for v in abnormal_conf_variants:
        assert isinstance(v, ConditionVariant)
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: Condition variants — extra cases
    # ==================================================================
    print("--- Group 2: Condition Variants (Extra Cases) ---\n")

    # Test 6: Mediator + distractors
    print("Test 6: Mediator + distractors, ATE 2→4")
    med_dag = nx.DiGraph()
    med_dag.add_edges_from([
        (0, 2), (1, 2),   # Z1→X, Z2→X
        (2, 3), (3, 4),   # X→M, M→Y
        (5, 4),            # W→Y
        (6, 7), (7, 8),   # D1→D2, D2→D3
        (9, 10),           # U1→U2
        (4, 11), (11, 12), # Y→S1, S1→S2
        (3, 13),           # M→R
    ])
    med_scm = RLCausalSCM(med_dag, seed=42)
    med_variants = generate_condition_variants(med_scm, 2, 4)

    assert med_variants[2].nodes_kept == [3, 4, 5]
    pruned_expected = sorted(set(range(14)) - {3, 4, 5})
    assert med_variants[2].nodes_pruned == pruned_expected
    print(f"   Variant 3 nodes_kept: {med_variants[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {med_variants[2].nodes_pruned}")

    # Verify ATE is computable
    med_result = compute_ate(med_scm, 2, 4)
    assert -1.0 <= med_result.ate_surgery <= 1.0
    print(f"   ATE = {med_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 7: Treatment irrelevant (no causal path to outcome)
    print("Test 7: Treatment irrelevant, ATE 0→1 (should be ~0)")
    irrel_dag = nx.DiGraph()
    irrel_dag.add_edges_from([
        (4, 2), (2, 1),   # C→A→Y
        (5, 3), (3, 1),   # D→B→Y
        (8, 0), (0, 6),   # P→X, X→E
        (6, 7),            # E→F
        (9, 10),           # G→H (distractor)
    ])
    irrel_scm = RLCausalSCM(irrel_dag, seed=42)
    irrel_variants = generate_condition_variants(irrel_scm, 0, 1)

    assert irrel_variants[2].nodes_kept == [1, 2, 3, 4, 5]
    assert 0 in irrel_variants[2].nodes_pruned
    assert 6 in irrel_variants[2].nodes_pruned  # E (descendant of X)
    assert 7 in irrel_variants[2].nodes_pruned  # F
    assert 9 in irrel_variants[2].nodes_pruned  # G distractor
    assert 10 in irrel_variants[2].nodes_pruned  # H distractor

    irrel_result = compute_ate(irrel_scm, 0, 1)
    assert irrel_result.is_abnormal, "Treatment irrelevant ATE 0→1 should be abnormal (ATE=0)"
    assert_close(irrel_result.ate_surgery, 0.0, atol=1e-10,
                 msg=f"ATE should be 0 when no causal path, got {irrel_result.ate_surgery}")
    print(f"   Variant 3 nodes_kept: {irrel_variants[2].nodes_kept}")
    print(f"   ATE = {irrel_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # Test 8: Long chain + side parents + descendants
    print("Test 8: Long chain + side parents, ATE 1→4")
    long_dag = nx.DiGraph()
    long_dag.add_edges_from([
        (0, 1),            # R1→X
        (1, 2), (5, 2),   # X→M1, T1→M1
        (2, 3), (6, 3),   # M1→M2, V→M2
        (3, 4),            # M2→Y
        (4, 7), (7, 8), (8, 9),  # Y→S1→S2→S3
        (2, 10), (10, 11),       # M1→Q1→Q2
    ])
    long_scm = RLCausalSCM(long_dag, seed=42)
    long_variants = generate_condition_variants(long_scm, 1, 4)

    assert long_variants[2].nodes_kept == [2, 3, 4, 5, 6]
    assert 0 in long_variants[2].nodes_pruned  # R1 (former parent of X)
    assert 7 in long_variants[2].nodes_pruned  # S1 (descendant of Y)
    assert 8 in long_variants[2].nodes_pruned  # S2
    assert 9 in long_variants[2].nodes_pruned  # S3
    assert 10 in long_variants[2].nodes_pruned  # Q1 (descendant of M1)
    assert 11 in long_variants[2].nodes_pruned  # Q2

    long_result = compute_ate(long_scm, 1, 4)
    assert -1.0 <= long_result.ate_surgery <= 1.0
    print(f"   Variant 3 nodes_kept: {long_variants[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {long_variants[2].nodes_pruned}")
    print(f"   ATE = {long_result.ate_surgery:.10f}")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 3: Rendering
    # ==================================================================
    print("--- Group 3: Rendering ---\n")

    # Test 9: CPD rendering
    print("Test 9: CPD rendering")
    # Root node (default decimal_places=2)
    root_cpt_text = render_node_cpt(chain_scm, 0)
    assert root_cpt_text.startswith("P(X0 = 1) = ")
    cpt_0_val = chain_scm.get_node_cpt(0)[()]
    assert f"{cpt_0_val:.2f}" in root_cpt_text

    # Conditional node
    child_cpt_text = render_node_cpt(chain_scm, 1)
    assert "P(X1 = 1 | X0):" in child_cpt_text
    assert "X0=0:" in child_cpt_text
    assert "X0=1:" in child_cpt_text
    cpt_1_vals = chain_scm.get_node_cpt(1)
    assert f"{cpt_1_vals[(0,)]:.2f}" in child_cpt_text
    assert f"{cpt_1_vals[(1,)]:.2f}" in child_cpt_text

    # Multi-parent node
    conf_child_text = render_node_cpt(conf_scm, 1)
    assert "P(X1 = 1 | X0, X2):" in conf_child_text
    assert "X0=0, X2=0:" in conf_child_text
    assert "X0=1, X2=1:" in conf_child_text

    # Edge rendering
    edges_text = render_dag_edges(chain_scm.dag)
    assert edges_text == "X0 -> X1"

    conf_edges_text = render_dag_edges(conf_scm.dag)
    assert "X0 -> X1" in conf_edges_text
    assert "X2 -> X0" in conf_edges_text
    assert "X2 -> X1" in conf_edges_text
    print("   -> PASSED\n")

    # ==================================================================
    # Group 4: Question assembly
    # ==================================================================
    print("--- Group 4: Question Assembly ---\n")

    # Test 10: Full question assembly on chain
    print("Test 10: Full question assembly on chain")
    chain_result = compute_ate(chain_scm, 0, 1)
    chain_questions = assemble_all_questions(
        chain_scm, 0, 1, chain_result, example_seed=42,
        treatment_outcome_depth=_tod(chain_scm.dag, [0], 1),
        effective_prob_conds=_epc(chain_scm.dag, [0], 1, {}),
        has_naive_conditioning=_ns(chain_scm.dag, [0], 1, []),
    )
    assert len(chain_questions) == 3
    for q in chain_questions:
        assert isinstance(q, AssembledQuestion)
        assert q.full_question_text
        assert q.treatment == 0
        assert q.outcome == 1
        # Verify structure: conditions section + ATE query line 1 + answer format
        assert "probability relationships" in q.full_question_text.lower()
        assert "Average Treatment Effect" in q.full_question_text
        assert ANSWER_FORMAT_TEMPLATE in q.full_question_text
        # Verify graph/query variant fields
        assert q.graph_variant in (1, 2, 3)
        assert q.query_variant in (1, 2, 3)

    # Same graph/query variant across all condition variants
    assert all(q.graph_variant == chain_questions[0].graph_variant for q in chain_questions)
    assert all(q.query_variant == chain_questions[0].query_variant for q in chain_questions)

    # Verify variant names
    assert chain_questions[0].variant_name == "full"
    assert chain_questions[1].variant_name == "surgery_only"
    assert chain_questions[2].variant_name == "surgery_ancestor_prune"

    # All should have same ATE
    for q in chain_questions:
        assert_close(q.ate_ground_truth, chain_result.ate_surgery, atol=1e-15)
    print("   -> PASSED\n")

    # Test 11: Metadata correctness
    print("Test 11: Metadata correctness")
    q = chain_questions[0]
    expected_keys = {
        "num_nodes", "num_edges", "dag_edges", "treatment", "outcome",
        "variant_name", "is_abnormal", "ate_surgery", "ate_causal_query",
        "verification_ok", "graph_variant", "query_variant",
        "nodes_kept", "nodes_pruned",
    }
    assert expected_keys.issubset(set(q.metadata.keys())), \
        f"Missing keys: {expected_keys - set(q.metadata.keys())}"
    assert q.metadata["num_nodes"] == 2
    assert q.metadata["num_edges"] == 1
    assert q.metadata["treatment"] == 0
    assert q.metadata["outcome"] == 1
    assert q.metadata["graph_variant"] in (1, 2, 3)
    assert q.metadata["query_variant"] in (1, 2, 3)
    print("   -> PASSED\n")

    # ==================================================================
    # Group 5: End-to-end
    # ==================================================================
    print("--- Group 5: End-to-End ---\n")

    # Test 12: Random 5-node DAG
    print("Test 12: End-to-end on random 5-node DAG")
    from data_generation.rlcausal_dag_sampling import CausalGraphSampler
    dag_sampler = CausalGraphSampler(num_nodes=5, seed=42)
    random_dag = dag_sampler.sample()
    random_scm = RLCausalSCM(random_dag, seed=42)

    # Find a causal pair (treatment with a descendant)
    test_treatment = None
    test_outcome = None
    for t in sorted(random_dag.nodes()):
        descendants = nx.descendants(random_dag, t)
        if descendants:
            test_treatment = t
            test_outcome = min(descendants)
            break

    assert test_treatment is not None, "No causal pair found in random DAG"

    random_result = compute_ate(random_scm, test_treatment, test_outcome)
    e2e_questions = assemble_all_questions(
        random_scm, test_treatment, test_outcome, random_result, example_seed=42,
        treatment_outcome_depth=_tod(random_scm.dag, [test_treatment], test_outcome),
        effective_prob_conds=_epc(random_scm.dag, [test_treatment], test_outcome, {}),
        has_naive_conditioning=_ns(random_scm.dag, [test_treatment], test_outcome, []),
    )
    assert len(e2e_questions) == 3
    for q in e2e_questions:
        assert -1.0 <= q.ate_ground_truth <= 1.0
        assert q.full_question_text
        assert q.treatment == test_treatment
        assert q.outcome == test_outcome

    print(f"   DAG edges: {sorted(random_dag.edges())}")
    print(f"   Treatment: {test_treatment}, Outcome: {test_outcome}")
    print(f"   ATE = {e2e_questions[0].ate_ground_truth:.10f}")
    print(f"   Variant 3 nodes_kept: {e2e_questions[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {e2e_questions[2].nodes_pruned}")

    # Print sample questions for visual inspection
    for q in e2e_questions:
        print(f"\n   --- Sample question (variant: {q.variant_name}) ---")
        print(q.full_question_text)
        print("   --- End sample ---")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 6: Graph/query verbalization variants
    # ==================================================================
    print("--- Group 6: Graph/Query Verbalization Variants ---\n")

    # Test 13: Graph variant renderers
    print("Test 13: Graph variant renderers")
    for gv in (1, 2, 3):
        gt = _render_graph_text(conf_scm, gv)
        assert gt, f"Empty graph_text for graph_variant={gv}"
    # V1: preamble + edge list
    gt1 = _render_graph_text(conf_scm, 1)
    assert "hypothetical world" in gt1
    assert "X0 -> X1" in gt1
    # V2: sentence-per-edge
    gt2 = _render_graph_text(conf_scm, 2)
    assert "directly causes" in gt2
    assert "No other causal relationships" in gt2
    # V3: parent-set
    gt3 = _render_graph_text(conf_scm, 3)
    assert "has parents:" in gt3
    assert "does not have any parent;" in gt3
    print("   -> PASSED\n")

    # Test 14: Query variant renderers
    print("Test 14: Query variant renderers")
    for qv in (1, 2, 3):
        qt = _render_query_text(0, 1, query_variant=qv)
        assert "Average Treatment Effect" in qt
        assert qt, f"Empty query_text for query_variant={qv}"
    # V1
    qt1 = _render_query_text(0, 1, query_variant=1)
    assert "force X0 = 1" in qt1
    # V2
    qt2 = _render_query_text(0, 1, query_variant=2)
    assert "regardless of how" in qt2
    # V3
    qt3 = _render_query_text(0, 1, query_variant=3)
    assert "intervening to change" in qt3
    print("   -> PASSED\n")

    # Test 15: Deterministic variant sampling
    print("Test 15: Deterministic variant sampling")
    _chain_dm = dict(
        treatment_outcome_depth=_tod(chain_scm.dag, [0], 1),
        effective_prob_conds=_epc(chain_scm.dag, [0], 1, {}),
        has_naive_conditioning=_ns(chain_scm.dag, [0], 1, []),
    )
    q_a = assemble_all_questions(chain_scm, 0, 1, chain_result, example_seed=123, **_chain_dm)
    q_b = assemble_all_questions(chain_scm, 0, 1, chain_result, example_seed=123, **_chain_dm)
    assert q_a[0].graph_variant == q_b[0].graph_variant
    assert q_a[0].query_variant == q_b[0].query_variant
    # Different seed should (with high probability) produce different variants
    q_c = assemble_all_questions(chain_scm, 0, 1, chain_result, example_seed=999, **_chain_dm)
    # Just verify it runs; variants may coincidentally match
    assert q_c[0].graph_variant in (1, 2, 3)
    print("   -> PASSED\n")

    # Test 16: Answer format template in all questions
    print("Test 16: Answer format template present in all questions")
    for q in chain_questions:
        assert "\\boxed{" in q.full_question_text, \
            f"Missing \\boxed{{}} in variant {q.variant_name}"
    print("   -> PASSED\n")

    print("=== All Tests Passed Successfully ===")
