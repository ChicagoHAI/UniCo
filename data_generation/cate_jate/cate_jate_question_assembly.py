"""
cate_jate_question_assembly.py

CATE/JATE Question Assembly: text rendering, condition variants, and
question assembly for Conditional ATE and Joint ATE queries.

Given an RLCausalSCM, query specification (treatments, outcome, evidence),
and a pre-computed CATEJATEResult, assembles three question variants with
different levels of probability conditions.

Only RLCausalSCM is supported as the input SCM class.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from data_generation.scm_mechanism_sampling import SCMBase, RLCausalSCM
from data_generation.cate_jate.cate_jate_query_sampling import CATEJATEResult


# ============================================================
# Local Utilities
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
        P(X0 = 1) = 0.42

    Conditional node example:
        P(X2 = 1 | X0, X1):
          X0=0, X1=0: 0.12
          X0=0, X1=1: 0.57
          X0=1, X1=0: 0.35
          X0=1, X1=1: 0.89
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
    """A complete assembled question for one (treatments, outcome, evidence, variant) tuple."""
    treatments: List[int]
    outcome: int
    evidence: Dict[int, int]
    query_type: str
    variant_name: str
    effect_ground_truth: float
    is_abnormal: bool
    graph_variant: int          # 1, 2, or 3
    query_variant: int          # 1, 2, or 3
    graph_text: str
    conditions_text: str
    query_text: str
    full_question_text: str     # combined: graph + conditions + query + answer format
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


# ============================================================
# Section A: Query Type Determination
# ============================================================

def _determine_query_type(
    treatments: List[int],
    evidence: Dict[int, int],
) -> str:
    """
    Determine query type from treatments and evidence.

    Returns:
        "cate" if single treatment with evidence,
        "jate" if multiple treatments without evidence.

    Raises:
        ValueError: For unsupported shapes (multi-treatment + evidence,
            no treatments, etc.)
    """
    if len(treatments) == 0:
        raise ValueError("No treatments provided.")
    if len(treatments) >= 2 and len(evidence) > 0:
        raise ValueError(
            f"Unsupported shape: multiple treatments AND evidence. "
            f"treatments={treatments}, evidence={evidence}"
        )
    if len(treatments) == 1 and len(evidence) > 0:
        return "cate"
    if len(treatments) >= 2 and len(evidence) == 0:
        return "jate"
    # Single treatment, no evidence — treat as degenerate CATE (plain ATE)
    return "cate"


# ============================================================
# Section B: Text Rendering
# ============================================================

def _render_query_text(
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    query_variant: int,
) -> str:
    """
    Render the query expression for CATE or JATE.

    Args:
        treatments: List of treatment node IDs.
        outcome: Outcome node ID.
        evidence: Evidence dict (node ID -> observed value).
        query_variant: Which verbalization style to use (1, 2, or 3).
    """
    query_type = _determine_query_type(treatments, evidence)
    o_label = _node_label(outcome)
    sorted_treatments = sorted(treatments)
    t_labels = [_node_label(t) for t in sorted_treatments]

    if query_type == "cate":
        t_label = t_labels[0]
        # Evidence assignments sorted by node ID
        sorted_evidence = sorted(evidence.items())
        given_clause = ", ".join(
            f"{_node_label(e)} = {v}" for e, v in sorted_evidence
        )

        line1 = (
            f"What is the Conditional Average Treatment Effect (CATE) "
            f"of {t_label} on {o_label}, given {given_clause}?"
        )

        if query_variant == 1:
            line2 = (
                f"That is, among cases where {given_clause}, "
                f"how does the probability of {o_label} = 1 change "
                f"if we force {t_label} = 1 compared to forcing {t_label} = 0?"
            )
        elif query_variant == 2:
            line2 = (
                f"That is, among cases where {given_clause}, "
                f"what is the difference in P({o_label} = 1) when "
                f"{t_label} is set to 1 versus set to 0, regardless of how "
                f"{t_label} would naturally arise?"
            )
        elif query_variant == 3:
            line2 = (
                f"In other words, among cases where {given_clause}, "
                f"how much does intervening to change {t_label} from 0 to 1 "
                f"affect the probability that {o_label} = 1?"
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")

    else:  # jate
        t_list_str = " and ".join(t_labels)
        do_1_str = " and ".join(f"{tl} = 1" for tl in t_labels)
        do_0_str = " and ".join(f"{tl} = 0" for tl in t_labels)
        both_or_all = "both" if len(treatments) == 2 else "all"

        line1 = (
            f"What is the Joint Average Treatment Effect (JATE) "
            f"of {t_list_str} on {o_label}?"
        )

        if query_variant == 1:
            line2 = (
                f"That is, how does the probability of {o_label} = 1 change "
                f"if we simultaneously force {do_1_str}, compared to "
                f"simultaneously forcing {do_0_str}?"
            )
        elif query_variant == 2:
            line2 = (
                f"That is, what is the difference in P({o_label} = 1) when "
                f"{t_list_str} are jointly set to 1 versus jointly set to 0, "
                f"regardless of how they would naturally arise?"
            )
        elif query_variant == 3:
            line2 = (
                f"In other words, how much does simultaneously intervening "
                f"to set {do_1_str}, compared to setting {both_or_all} to 0, "
                f"affect the probability that {o_label} = 1?"
            )
        else:
            raise ValueError(f"Unknown query_variant: {query_variant}")

    return f"{line1}\n{line2}"


# ============================================================
# Section C: Condition Variants
# ============================================================

def condition_variant_full(
    scm: SCMBase,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
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
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    """
    Variant 2 — Surgery-only: CPDs for all nodes EXCEPT treatment(s).

    After graph surgery, the treatment node(s)' mechanism is replaced by
    the intervention, so their CPDs are omitted. Non-treatment CPDs are
    identical between the original and mutilated models, so we render
    from scm.cpts.
    """
    all_nodes = set(scm.nodes)
    cpd_nodes = all_nodes - set(treatments)
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
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    decimal_places: int = 2,
) -> ConditionVariant:
    """
    Variant 3 — Surgery + ancestor pruning.

    After do(treatments), keep only ancestors of outcome AND ancestors of
    evidence nodes in the mutilated graph (plus outcome and evidence nodes
    themselves). Treatment(s) are excluded from CPDs since the intervention
    replaces their mechanism.

    Steps:
    1. mutilated = model.do(treatments)
    2. relevant_nodes = [outcome] + sorted(evidence.keys())
    3. anc_graph = mutilated.get_ancestral_graph(nodes=relevant_nodes)
       → includes relevant_nodes + all their ancestors in the mutilated graph
    4. cpd_nodes = anc_graph.nodes() - set(treatments)
    5. Render CPDs for cpd_nodes from scm.cpts in topo order.
    """
    mutilated = scm.pgmpy_model.do(treatments)

    # Relevant target nodes: outcome + evidence nodes
    relevant_nodes = [outcome] + sorted(evidence.keys())
    anc_graph = mutilated.get_ancestral_graph(nodes=relevant_nodes)
    anc_nodes = set(anc_graph.nodes())

    cpd_nodes = anc_nodes - set(treatments)
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
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    *,
    decimal_places: int = 2,
) -> List[ConditionVariant]:
    """
    Generate all three condition variants for a CATE/JATE query.

    Returns:
        List of three ConditionVariant objects:
            [0] full, [1] surgery_only, [2] surgery_ancestor_prune
    """
    return [
        condition_variant_full(scm, treatments, outcome, evidence, decimal_places=decimal_places),
        condition_variant_surgery_only(scm, treatments, outcome, evidence, decimal_places=decimal_places),
        condition_variant_surgery_ancestor_prune(scm, treatments, outcome, evidence, decimal_places=decimal_places),
    ]


# ============================================================
# Section D: Question Assembly
# ============================================================

def assemble_question(
    scm: SCMBase,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    variant: ConditionVariant,
    cate_jate_result: CATEJATEResult,
    *,
    graph_variant: int,
    query_variant: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> AssembledQuestion:
    """
    Assemble a complete question from a condition variant and effect result.

    The question has four parts:
    1. Full causal graph (original DAG edges, never mutilated)
    2. Probability conditions (from the variant)
    3. Query expression (CATE or JATE)
    4. Answer format template
    """
    graph_text = _render_graph_text(scm, graph_variant)
    query_text = _render_query_text(treatments, outcome, evidence, query_variant=query_variant)

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
        "treatments": cate_jate_result.treatments,
        "outcome": cate_jate_result.outcome,
        "evidence": cate_jate_result.evidence,
        "query_type": cate_jate_result.query_type,
        "variant_name": variant.variant_name,
        "is_abnormal": cate_jate_result.is_abnormal,
        "effect_surgery": cate_jate_result.effect_surgery,
        "effect_causal_query": cate_jate_result.effect_causal_query,
        "verification_ok": cate_jate_result.verification_ok,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": variant.nodes_kept,
        "nodes_pruned": variant.nodes_pruned,
        "treatment_outcome_depth": treatment_outcome_depth,
        "effective_prob_conds": effective_prob_conds,
        "has_naive_conditioning": has_naive_conditioning,
    }

    return AssembledQuestion(
        treatments=treatments,
        outcome=outcome,
        evidence=evidence,
        query_type=cate_jate_result.query_type,
        variant_name=variant.variant_name,
        effect_ground_truth=cate_jate_result.effect_surgery,
        is_abnormal=cate_jate_result.is_abnormal,
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
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    cate_jate_result: CATEJATEResult,
    *,
    decimal_places: int = 2,
    example_seed: int,
    treatment_outcome_depth: float,
    effective_prob_conds: int,
    has_naive_conditioning: bool,
) -> List[AssembledQuestion]:
    """
    Top-level function: generate all three condition variants and assemble
    three questions from a pre-computed CATEJATEResult.

    For each example, a (graph_variant, query_variant) pair is sampled
    uniformly from {1,2,3} x {1,2,3} using example_seed. The same pair
    is used across all three condition variants.

    Args:
        scm: The SCM containing the DAG and CPTs.
        treatments: List of treatment node IDs.
        outcome: Outcome node ID.
        evidence: Evidence dict (node ID -> observed value).
        cate_jate_result: Pre-computed CATEJATEResult.
        decimal_places: Number of decimal places for CPT rendering.
        example_seed: Seed for deterministic graph/query variant sampling.
        treatment_outcome_depth: Pre-computed depth from treatment(s) to outcome.
        effective_prob_conds: Pre-computed effective probability conditions.
        has_naive_conditioning: Pre-computed naive conditioning flag.

    Returns:
        List of three AssembledQuestion objects (one per variant).
    """
    rng = random.Random(example_seed)
    graph_variant = rng.randint(1, 3)
    query_variant = rng.randint(1, 3)

    variants = generate_condition_variants(
        scm, treatments, outcome, evidence,
        decimal_places=decimal_places,
    )

    return [
        assemble_question(
            scm, treatments, outcome, evidence, variant, cate_jate_result,
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
    import random
    from data_generation.dataset_utils.difficulty_metrics import (
        compute_treatment_outcome_depth as _tod,
        compute_effective_prob_conds as _epc,
        has_naive_conditioning as _ns,
    )

    from data_generation.rlcausal_dag_sampling import CausalGraphSampler
    from data_generation.cate_jate.cate_jate_query_sampling import (
        compute_cate_jate,
        is_abnormal_triplets,
    )

    def assert_close(a, b, *, atol=1e-5, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing CATE/JATE Question Assembly ===\n")

    # ==================================================================
    # Group 1: Condition Variants (Base Cases)
    # ==================================================================
    print("--- Group 1: Condition Variants (Base Cases) ---\n")

    # Test 1: CATE variants for confounder, treatment=0, outcome=1, evidence={2: 1}
    print("Test 1: CATE variants for confounder 2→0, 2→1, 0→1")
    conf_dag = nx.DiGraph()
    conf_dag.add_edges_from([(2, 0), (2, 1), (0, 1)])
    conf_scm = RLCausalSCM(conf_dag, seed=42)

    conf_variants = generate_condition_variants(conf_scm, [0], 1, {2: 1})
    assert len(conf_variants) == 3

    # Full: all nodes
    assert conf_variants[0].nodes_kept == [0, 1, 2]
    assert conf_variants[0].nodes_pruned == []
    # Surgery-only: exclude treatment 0
    assert conf_variants[1].nodes_kept == [1, 2]
    assert conf_variants[1].nodes_pruned == [0]
    # Surgery+prune: surgery on 0 removes 2→0. Ancestors of 1: {0, 2} (via 0→1, 2→1).
    # Ancestors of evidence node 2: {} (2 is a root). anc_nodes = {0, 1, 2}.
    # cpd_nodes = {0, 1, 2} - {0} = {1, 2}. Same as surgery_only.
    assert conf_variants[2].nodes_kept == [1, 2]
    assert conf_variants[2].nodes_pruned == [0]
    print("   -> PASSED\n")

    # Test 2: JATE variants for fork 0→2, 1→2
    print("Test 2: JATE variants for fork 0→2, 1→2")
    fork_dag = nx.DiGraph()
    fork_dag.add_edges_from([(0, 2), (1, 2)])
    fork_scm = RLCausalSCM(fork_dag, seed=42)

    fork_variants = generate_condition_variants(fork_scm, [0, 1], 2, {})
    assert len(fork_variants) == 3

    # Full: all nodes
    assert fork_variants[0].nodes_kept == [0, 1, 2]
    assert fork_variants[0].nodes_pruned == []
    # Surgery-only: exclude treatments 0, 1
    assert fork_variants[1].nodes_kept == [2]
    assert fork_variants[1].nodes_pruned == [0, 1]
    # Surgery+prune: surgery on [0, 1] removes their parents (none, they're roots).
    # Ancestors of 2 in mutilated: {0, 1} (via 0→2, 1→2). anc_nodes = {0, 1, 2}.
    # cpd_nodes = {0, 1, 2} - {0, 1} = {2}.
    assert fork_variants[2].nodes_kept == [2]
    assert fork_variants[2].nodes_pruned == [0, 1]
    print("   -> PASSED\n")

    # Test 3: CATE ancestor prune keeps evidence ancestors
    # DAG: 3→2, 2→0, 0→1, 4→5 (distractor)
    # treatment=0, outcome=1, evidence={2: 1}
    # Surgery on 0: removes 2→0. Mutilated: 3→2, 0→1, 4→5.
    # Ancestors of outcome 1 in mutilated: {0} (via 0→1). anc_nodes for 1 = {0, 1}.
    # Ancestors of evidence 2 in mutilated: {3} (via 3→2). anc_nodes for 2 = {2, 3}.
    # Combined anc_nodes = {0, 1, 2, 3}. cpd_nodes = {0, 1, 2, 3} - {0} = {1, 2, 3}.
    print("Test 3: CATE ancestor prune keeps evidence ancestors")
    ev_anc_dag = nx.DiGraph()
    ev_anc_dag.add_edges_from([(3, 2), (2, 0), (0, 1), (4, 5)])
    ev_anc_scm = RLCausalSCM(ev_anc_dag, seed=42)

    ev_anc_variants = generate_condition_variants(ev_anc_scm, [0], 1, {2: 1})
    assert ev_anc_variants[2].nodes_kept == [1, 2, 3], \
        f"Expected [1, 2, 3], got {ev_anc_variants[2].nodes_kept}"
    assert 4 in ev_anc_variants[2].nodes_pruned  # distractor
    assert 5 in ev_anc_variants[2].nodes_pruned  # distractor
    print(f"   Variant 3 nodes_kept: {ev_anc_variants[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {ev_anc_variants[2].nodes_pruned}")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: Condition Variants (Extra Cases)
    # ==================================================================
    print("--- Group 2: Condition Variants (Extra Cases) ---\n")

    # Test 4: CATE with complex evidence ancestry
    # DAG: 5→3, 3→0, 0→1, 4→2, 2→1, 6→7 (distractor), 8→4
    # treatment=0, outcome=1, evidence={2: 0}
    # Surgery on 0: removes 3→0. Mutilated: 5→3, 0→1, 4→2, 2→1, 6→7, 8→4.
    # Ancestors of 1 in mutilated: {0, 2, 4, 8} (via 0→1, 2→1, 4→2, 8→4).
    # anc_nodes for 1 = {0, 1, 2, 4, 8}.
    # Ancestors of 2 in mutilated: {4, 8} (via 4→2, 8→4).
    # anc_nodes for 2 = {2, 4, 8}.
    # Combined = {0, 1, 2, 4, 8}. cpd_nodes = combined - {0} = {1, 2, 4, 8}.
    print("Test 4: CATE with complex evidence ancestry")
    complex_dag = nx.DiGraph()
    complex_dag.add_edges_from([(5, 3), (3, 0), (0, 1), (4, 2), (2, 1), (6, 7), (8, 4)])
    complex_scm = RLCausalSCM(complex_dag, seed=42)

    complex_variants = generate_condition_variants(complex_scm, [0], 1, {2: 0})
    assert complex_variants[2].nodes_kept == [1, 2, 4, 8], \
        f"Expected [1, 2, 4, 8], got {complex_variants[2].nodes_kept}"
    assert 3 in complex_variants[2].nodes_pruned  # former parent of treatment
    assert 5 in complex_variants[2].nodes_pruned  # grandparent of treatment
    assert 6 in complex_variants[2].nodes_pruned  # distractor
    assert 7 in complex_variants[2].nodes_pruned  # distractor
    print(f"   Variant 3 nodes_kept: {complex_variants[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {complex_variants[2].nodes_pruned}")
    print("   -> PASSED\n")

    # Test 5: JATE with distractors
    # DAG: 0→3, 1→3, 2→4, 4→5 (distractors 2, 4, 5)
    # treatment=[0, 1], outcome=3
    # Surgery on [0, 1]: they're roots, no change.
    # Ancestors of 3 in mutilated: {0, 1} (via 0→3, 1→3). anc_nodes = {0, 1, 3}.
    # cpd_nodes = {0, 1, 3} - {0, 1} = {3}.
    print("Test 5: JATE with distractors")
    jate_distract_dag = nx.DiGraph()
    jate_distract_dag.add_edges_from([(0, 3), (1, 3), (2, 4), (4, 5)])
    jate_distract_scm = RLCausalSCM(jate_distract_dag, seed=42)

    jate_distract_variants = generate_condition_variants(
        jate_distract_scm, [0, 1], 3, {}
    )
    assert jate_distract_variants[2].nodes_kept == [3], \
        f"Expected [3], got {jate_distract_variants[2].nodes_kept}"
    assert 2 in jate_distract_variants[2].nodes_pruned
    assert 4 in jate_distract_variants[2].nodes_pruned
    assert 5 in jate_distract_variants[2].nodes_pruned
    print(f"   Variant 3 nodes_kept: {jate_distract_variants[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {jate_distract_variants[2].nodes_pruned}")
    print("   -> PASSED\n")

    # Test 6: CATE with two evidence nodes having different ancestry
    # DAG: 0→3, 4→3, 5→1, 6→2
    # treatment=0, outcome=3, evidence={1: 0, 2: 1}
    # Surgery on 0: no parents removed (0 is root).
    # Ancestors of 3 in mutilated: {0, 4}. anc_nodes for 3 = {0, 3, 4}.
    # Ancestors of 1 in mutilated: {5}. anc_nodes for 1 = {1, 5}.
    # Ancestors of 2 in mutilated: {6}. anc_nodes for 2 = {2, 6}.
    # Combined = {0, 1, 2, 3, 4, 5, 6}. cpd_nodes = combined - {0} = {1, 2, 3, 4, 5, 6}.
    print("Test 6: CATE with two evidence nodes")
    multi_ev_dag = nx.DiGraph()
    multi_ev_dag.add_edges_from([(0, 3), (4, 3), (5, 1), (6, 2)])
    multi_ev_scm = RLCausalSCM(multi_ev_dag, seed=42)

    multi_ev_variants = generate_condition_variants(
        multi_ev_scm, [0], 3, {1: 0, 2: 1}
    )
    assert multi_ev_variants[2].nodes_kept == [1, 2, 3, 4, 5, 6], \
        f"Expected [1, 2, 3, 4, 5, 6], got {multi_ev_variants[2].nodes_kept}"
    assert 0 in multi_ev_variants[2].nodes_pruned  # treatment
    print(f"   Variant 3 nodes_kept: {multi_ev_variants[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {multi_ev_variants[2].nodes_pruned}")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 3: Rendering
    # ==================================================================
    print("--- Group 3: Rendering ---\n")

    # Test 7: CATE query text variants
    print("Test 7: CATE query text variants")
    for qv in (1, 2, 3):
        cate_qt = _render_query_text([0], 3, {1: 1, 2: 0}, query_variant=qv)
        assert "Conditional Average Treatment Effect (CATE)" in cate_qt
        assert "given X1 = 1, X2 = 0" in cate_qt
    # V1
    cate_qt1 = _render_query_text([0], 3, {1: 1, 2: 0}, query_variant=1)
    assert "force X0 = 1" in cate_qt1
    # V2
    cate_qt2 = _render_query_text([0], 3, {1: 1, 2: 0}, query_variant=2)
    assert "regardless of how" in cate_qt2
    # V3
    cate_qt3 = _render_query_text([0], 3, {1: 1, 2: 0}, query_variant=3)
    assert "intervening to change" in cate_qt3
    print(f"   CATE V1:\n{cate_qt1}")
    print("   -> PASSED\n")

    # Test 8: JATE query text variants
    print("Test 8: JATE query text variants")
    for qv in (1, 2, 3):
        jate_qt = _render_query_text([0, 1], 3, {}, query_variant=qv)
        assert "Joint Average Treatment Effect (JATE)" in jate_qt
        assert "X0 and X1 on X3" in jate_qt
    # V1
    jate_qt1 = _render_query_text([0, 1], 3, {}, query_variant=1)
    assert "simultaneously force" in jate_qt1
    # V2
    jate_qt2 = _render_query_text([0, 1], 3, {}, query_variant=2)
    assert "jointly set to 1" in jate_qt2
    # V3
    jate_qt3 = _render_query_text([0, 1], 3, {}, query_variant=3)
    assert "simultaneously intervening" in jate_qt3
    assert "both" in jate_qt3  # 2 treatments -> "both"
    print(f"   JATE V1:\n{jate_qt1}")
    print("   -> PASSED\n")

    # Test 9: CPD rendering
    print("Test 9: CPD rendering")
    chain_dag = nx.DiGraph()
    chain_dag.add_edges_from([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=42)

    root_cpt_text = render_node_cpt(chain_scm, 0)
    assert root_cpt_text.startswith("P(X0 = 1) = ")
    child_cpt_text = render_node_cpt(chain_scm, 1)
    assert "P(X1 = 1 | X0):" in child_cpt_text
    assert "X0=0:" in child_cpt_text
    assert "X0=1:" in child_cpt_text
    print("   -> PASSED\n")

    # Test 10: Edge rendering
    print("Test 10: Edge rendering")
    edges_text = render_dag_edges(chain_scm.dag)
    assert "X0 -> X1" in edges_text
    assert "X1 -> X2" in edges_text
    print("   -> PASSED\n")

    # ==================================================================
    # Group 4: Question Assembly
    # ==================================================================
    print("--- Group 4: Question Assembly ---\n")

    # Test 11: Full CATE question assembly
    print("Test 11: Full CATE question assembly")
    conf_cate_result = compute_cate_jate(conf_scm, [0], 1, {2: 1})
    cate_questions = assemble_all_questions(
        conf_scm, [0], 1, {2: 1}, conf_cate_result, example_seed=42,
        treatment_outcome_depth=_tod(conf_scm.dag, [0], 1),
        effective_prob_conds=_epc(conf_scm.dag, [0], 1, {2: 1}),
        has_naive_conditioning=_ns(conf_scm.dag, [0], 1, [2]),
    )
    assert len(cate_questions) == 3
    for q in cate_questions:
        assert isinstance(q, AssembledQuestion)
        assert q.full_question_text
        assert q.treatments == [0]
        assert q.outcome == 1
        assert q.evidence == {2: 1}
        assert q.query_type == "cate"
        assert "probability relationships" in q.full_question_text.lower()
        assert "Conditional Average Treatment Effect" in q.full_question_text
        assert ANSWER_FORMAT_TEMPLATE in q.full_question_text
        assert q.graph_variant in (1, 2, 3)
        assert q.query_variant in (1, 2, 3)

    # Same graph/query variant across all condition variants
    assert all(q.graph_variant == cate_questions[0].graph_variant for q in cate_questions)
    assert all(q.query_variant == cate_questions[0].query_variant for q in cate_questions)

    assert cate_questions[0].variant_name == "full"
    assert cate_questions[1].variant_name == "surgery_only"
    assert cate_questions[2].variant_name == "surgery_ancestor_prune"

    for q in cate_questions:
        assert_close(q.effect_ground_truth, conf_cate_result.effect_surgery, atol=1e-15)
    print("   -> PASSED\n")

    # Test 12: Full JATE question assembly
    print("Test 12: Full JATE question assembly")
    fork_jate_result = compute_cate_jate(fork_scm, [0, 1], 2, {})
    jate_questions = assemble_all_questions(
        fork_scm, [0, 1], 2, {}, fork_jate_result, example_seed=42,
        treatment_outcome_depth=_tod(fork_scm.dag, [0, 1], 2),
        effective_prob_conds=_epc(fork_scm.dag, [0, 1], 2, {}),
        has_naive_conditioning=_ns(fork_scm.dag, [0, 1], 2, []),
    )
    assert len(jate_questions) == 3
    for q in jate_questions:
        assert isinstance(q, AssembledQuestion)
        assert q.treatments == [0, 1]
        assert q.outcome == 2
        assert q.evidence == {}
        assert q.query_type == "jate"
        assert "Joint Average Treatment Effect" in q.full_question_text

    for q in jate_questions:
        assert_close(q.effect_ground_truth, fork_jate_result.effect_surgery, atol=1e-15)
    print("   -> PASSED\n")

    # Test 13: Metadata correctness
    print("Test 13: Metadata correctness")
    q = cate_questions[0]
    expected_keys = {
        "num_nodes", "num_edges", "dag_edges", "treatments", "outcome",
        "evidence", "query_type", "variant_name", "is_abnormal",
        "effect_surgery", "effect_causal_query", "verification_ok",
        "graph_variant", "query_variant",
        "nodes_kept", "nodes_pruned",
    }
    assert expected_keys.issubset(set(q.metadata.keys())), \
        f"Missing keys: {expected_keys - set(q.metadata.keys())}"
    assert q.metadata["num_nodes"] == 3
    assert q.metadata["treatments"] == [0]
    assert q.metadata["outcome"] == 1
    assert q.metadata["evidence"] == {2: 1}
    assert q.metadata["query_type"] == "cate"
    assert q.metadata["graph_variant"] in (1, 2, 3)
    assert q.metadata["query_variant"] in (1, 2, 3)
    print("   -> PASSED\n")

    # ==================================================================
    # Group 5: End-to-End (rejection sampling)
    # ==================================================================
    print("--- Group 5: End-to-End ---\n")

    # Test 14: E2E CATE via rejection sampling on random 6-node DAG
    print("Test 14: End-to-end CATE on random 6-node DAG (rejection sampling)")
    dag_sampler = CausalGraphSampler(num_nodes=6, seed=42)
    random_dag = dag_sampler.sample()
    random_scm = RLCausalSCM(random_dag, seed=42)

    # Rejection sampling: randomly sample non-overlapping (treatment, outcome, evidence)
    # triplets and filter via is_abnormal_triplets until a valid one is found.
    rng_cate = random.Random(42)
    cate_trials = 0
    cate_treatment = None
    cate_outcome = None
    cate_evidence_node = None
    MAX_REJECTION_TRIALS = 5000

    while cate_trials < MAX_REJECTION_TRIALS:
        cate_trials += 1
        # Sample 3 distinct nodes: treatment, outcome, evidence
        t, o, e = rng_cate.sample(random_scm.nodes, 3)
        # Feed into is_abnormal_triplets to check validity
        if not is_abnormal_triplets(random_dag, t, o, evidence=[e]):
            cate_treatment = t
            cate_outcome = o
            cate_evidence_node = e
            break

    assert cate_treatment is not None, (
        f"No valid CATE triplet found after {MAX_REJECTION_TRIALS} trials"
    )
    print(f"   Found valid CATE triplet after {cate_trials} trial(s)")

    cate_result = compute_cate_jate(random_scm, [cate_treatment], cate_outcome, {cate_evidence_node: 1})
    e2e_cate_questions = assemble_all_questions(
        random_scm, [cate_treatment], cate_outcome, {cate_evidence_node: 1}, cate_result,
        example_seed=42,
        treatment_outcome_depth=_tod(random_scm.dag, [cate_treatment], cate_outcome),
        effective_prob_conds=_epc(random_scm.dag, [cate_treatment], cate_outcome, {cate_evidence_node: 1}),
        has_naive_conditioning=_ns(random_scm.dag, [cate_treatment], cate_outcome, [cate_evidence_node]),
    )
    assert len(e2e_cate_questions) == 3
    for q in e2e_cate_questions:
        assert -1.0 <= q.effect_ground_truth <= 1.0
        assert q.full_question_text
        assert q.query_type == "cate"

    print(f"   DAG edges: {sorted(random_dag.edges())}")
    print(f"   Treatment: {cate_treatment}, Outcome: {cate_outcome}, Evidence: {{{cate_evidence_node}: 1}}")
    print(f"   CATE = {e2e_cate_questions[0].effect_ground_truth:.10f}")
    print(f"   Variant 3 nodes_kept: {e2e_cate_questions[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {e2e_cate_questions[2].nodes_pruned}")

    for q in e2e_cate_questions:
        print(f"\n   --- Sample CATE question (variant: {q.variant_name}) ---")
        print(q.full_question_text)
        print("   --- End sample ---")
    print("   -> PASSED\n")

    # Test 15: E2E JATE via rejection sampling on random 8-node DAG
    print("Test 15: End-to-end JATE on random 8-node DAG (rejection sampling)")
    dag_sampler_jate = CausalGraphSampler(num_nodes=8, seed=100)
    jate_dag = dag_sampler_jate.sample()
    jate_scm = RLCausalSCM(jate_dag, seed=100)

    # Rejection sampling: randomly sample non-overlapping (treatment1, treatment2, outcome)
    # triplets and filter via is_abnormal_triplets until a valid one is found.
    rng_jate = random.Random(100)
    jate_trials = 0
    jate_t1 = None
    jate_t2 = None
    jate_outcome = None

    while jate_trials < MAX_REJECTION_TRIALS:
        jate_trials += 1
        # Sample 3 distinct nodes: two treatments and one outcome
        t1, t2, o = rng_jate.sample(jate_scm.nodes, 3)
        # Feed into is_abnormal_triplets to check validity (no evidence for JATE)
        if not is_abnormal_triplets(jate_dag, [t1, t2], o):
            jate_t1 = t1
            jate_t2 = t2
            jate_outcome = o
            break

    assert jate_t1 is not None, (
        f"No valid JATE triplet found after {MAX_REJECTION_TRIALS} trials"
    )
    print(f"   Found valid JATE triplet after {jate_trials} trial(s)")

    jate_result = compute_cate_jate(jate_scm, [jate_t1, jate_t2], jate_outcome, {})
    e2e_jate_questions = assemble_all_questions(
        jate_scm, [jate_t1, jate_t2], jate_outcome, {}, jate_result,
        example_seed=100,
        treatment_outcome_depth=_tod(jate_scm.dag, [jate_t1, jate_t2], jate_outcome),
        effective_prob_conds=_epc(jate_scm.dag, [jate_t1, jate_t2], jate_outcome, {}),
        has_naive_conditioning=_ns(jate_scm.dag, [jate_t1, jate_t2], jate_outcome, []),
    )
    assert len(e2e_jate_questions) == 3
    for q in e2e_jate_questions:
        assert -1.0 <= q.effect_ground_truth <= 1.0
        assert q.full_question_text
        assert q.query_type == "jate"

    print(f"   DAG edges: {sorted(jate_dag.edges())}")
    print(f"   Treatments: [{jate_t1}, {jate_t2}], Outcome: {jate_outcome}")
    print(f"   JATE = {e2e_jate_questions[0].effect_ground_truth:.10f}")
    print(f"   Variant 3 nodes_kept: {e2e_jate_questions[2].nodes_kept}")
    print(f"   Variant 3 nodes_pruned: {e2e_jate_questions[2].nodes_pruned}")

    for q in e2e_jate_questions:
        print(f"\n   --- Sample JATE question (variant: {q.variant_name}) ---")
        print(q.full_question_text)
        print("   --- End sample ---")
    print("   -> PASSED\n")

    # ==================================================================
    # Group 6: Graph/query verbalization variants
    # ==================================================================
    print("--- Group 6: Graph/Query Verbalization Variants ---\n")

    # Test 16: Graph variant renderers
    print("Test 16: Graph variant renderers")
    for gv in (1, 2, 3):
        gt = _render_graph_text(conf_scm, gv)
        assert gt, f"Empty graph_text for graph_variant={gv}"
    # V1: preamble + edge list
    gt1 = _render_graph_text(conf_scm, 1)
    assert "hypothetical world" in gt1
    # V2: sentence-per-edge
    gt2 = _render_graph_text(conf_scm, 2)
    assert "directly causes" in gt2
    assert "No other causal relationships" in gt2
    # V3: parent-set
    gt3 = _render_graph_text(conf_scm, 3)
    assert "has parents:" in gt3
    print("   -> PASSED\n")

    # Test 17: Deterministic variant sampling
    print("Test 17: Deterministic variant sampling")
    _conf_dm = dict(
        treatment_outcome_depth=_tod(conf_scm.dag, [0], 1),
        effective_prob_conds=_epc(conf_scm.dag, [0], 1, {2: 1}),
        has_naive_conditioning=_ns(conf_scm.dag, [0], 1, [2]),
    )
    q_a = assemble_all_questions(conf_scm, [0], 1, {2: 1}, conf_cate_result, example_seed=123, **_conf_dm)
    q_b = assemble_all_questions(conf_scm, [0], 1, {2: 1}, conf_cate_result, example_seed=123, **_conf_dm)
    assert q_a[0].graph_variant == q_b[0].graph_variant
    assert q_a[0].query_variant == q_b[0].query_variant
    print("   -> PASSED\n")

    # Test 18: Answer format template in all questions
    print("Test 18: Answer format template present in all questions")
    for q in cate_questions:
        assert "\\boxed{" in q.full_question_text, \
            f"Missing \\boxed{{}} in variant {q.variant_name}"
    for q in jate_questions:
        assert "\\boxed{" in q.full_question_text, \
            f"Missing \\boxed{{}} in variant {q.variant_name}"
    print("   -> PASSED\n")

    # Test 19: JATE with >2 treatments uses "all" not "both"
    print("Test 19: JATE >2 treatments uses 'all' not 'both'")
    jate_3t_qt = _render_query_text([0, 1, 2], 3, {}, query_variant=3)
    assert "all" in jate_3t_qt
    assert "both" not in jate_3t_qt
    jate_2t_qt = _render_query_text([0, 1], 3, {}, query_variant=3)
    assert "both" in jate_2t_qt
    print("   -> PASSED\n")

    print("=== All Tests Passed Successfully ===")
