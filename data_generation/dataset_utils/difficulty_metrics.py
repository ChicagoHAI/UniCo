"""
difficulty_metrics.py

Difficulty proxy metrics for causal reasoning questions. All functions
operate on a NetworkX DiGraph and query parameters — no SCM required.

Metrics:
  - treatment_outcome_depth: average shortest directed path from T to Y
  - effective_prob_conds: probability conditions after surgery + ancestor prune
  - has_naive_conditioning: Rule-2 d-separation check (do → conditioning)
"""

import networkx as nx
from typing import Dict, List


def compute_treatment_outcome_depth(
    dag: nx.DiGraph,
    treatments: List[int],
    outcome: int,
) -> float:
    """Sum of lengths of all directed paths from each treatment to outcome,
    averaged across treatments.

    For a single treatment, returns the sum of lengths of every simple
    directed path from that treatment to the outcome (0 if no path exists).
    For multiple treatments, returns the mean of per-treatment sums.
    """
    total_depth = 0
    for t in treatments:
        for path in nx.all_simple_paths(dag, t, outcome):
            total_depth += len(path) - 1  # path is node list; length = #edges
    return total_depth / len(treatments)


def compute_effective_prob_conds(
    dag: nx.DiGraph,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
) -> int:
    """Count effective probability conditions after surgery + ancestor pruning.

    Replicates the surgery_ancestor_prune logic from question assembly:
    1. Mutilate graph by removing incoming edges to treatments (do-operation).
    2. Find ancestors of outcome + evidence nodes in mutilated graph.
    3. cpd_nodes = (ancestors ∪ {outcome, evidence}) - treatments.
    4. Sum 2^(number of parents in ORIGINAL dag) for each cpd_node.
    """
    # Build mutilated graph (remove incoming edges to treatments)
    mutilated = dag.copy()
    for t in treatments:
        mutilated.remove_edges_from(list(mutilated.in_edges(t)))

    # Find ancestors of relevant nodes in mutilated graph
    relevant_nodes = {outcome} | set(evidence.keys())
    anc_nodes = set()
    for node in relevant_nodes:
        anc_nodes |= nx.ancestors(mutilated, node)
    anc_nodes |= relevant_nodes

    # CPD nodes = ancestors (including relevant) minus treatments
    cpd_nodes = anc_nodes - set(treatments)

    # Count probability conditions using original DAG parent counts
    total = 0
    for node in cpd_nodes:
        num_parents = dag.in_degree(node)
        total += 2 ** num_parents
    return total


def compute_counterfactual_effective_prob_conds(
    dag: nx.DiGraph,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
) -> int:
    """Count effective probability conditions for counterfactual-style pruning.

    Mirrors the CF/ATT `surgery_ancestor_prune` rule:
    1. Factual abduction side: keep evidence nodes and their ancestors in the
       original DAG.
    2. Counterfactual prediction side: mutilate the graph by removing incoming
       edges to treatments, then keep the outcome node and its ancestors in the
       mutilated graph.
    3. Keep the union of those two relevance sets. Treatment nodes are NOT
       automatically removed because they may still be relevant on the factual
       side.
    4. Sum 2^(number of parents in ORIGINAL dag) for each kept node.
    """
    factual_nodes = set(evidence.keys())
    for node in evidence:
        factual_nodes |= nx.ancestors(dag, node)

    mutilated = dag.copy()
    for t in treatments:
        mutilated.remove_edges_from(list(mutilated.in_edges(t)))

    counterfactual_nodes = {outcome} | nx.ancestors(mutilated, outcome)
    cpd_nodes = factual_nodes | counterfactual_nodes

    total = 0
    for node in cpd_nodes:
        num_parents = dag.in_degree(node)
        total += 2 ** num_parents
    return total


def has_naive_conditioning(
    dag: nx.DiGraph,
    treatments: List[int],
    outcome: int,
    evidence_nodes: List[int],
) -> bool:
    """Check if naive conditioning holds via Rule-2 d-separation.

    Builds the Rule-2 exchange graph G_{underline T} by removing outgoing
    edges from all treatment nodes, then tests d-separation between
    treatments and outcome given evidence. If d-separated, Rule 2
    guarantees P(Y|do(T),E) = P(Y|T,E) structurally.

    Note: this removes outgoing edges (not incoming) — the correct
    construction for the action/observation exchange condition of
    do-calculus Rule 2.
    """
    G_underline_T = dag.copy()
    for t in treatments:
        G_underline_T.remove_edges_from(list(G_underline_T.out_edges(t)))
    return nx.is_d_separator(
        G_underline_T, set(treatments), {outcome}, set(evidence_nodes)
    )


# ============================================================
# TEST SUITE
# ============================================================
if __name__ == "__main__":
    import math

    def assert_close(a, b, *, atol=1e-9, msg=""):
        if not math.isclose(a, b, abs_tol=atol, rel_tol=1e-9):
            raise AssertionError(msg or f"Not close: {a} vs {b} (atol={atol})")

    print("=== Testing Difficulty Metrics ===\n")

    # ==================================================================
    # Group 1: compute_treatment_outcome_depth
    # ==================================================================
    print("--- Group 1: treatment_outcome_depth ---\n")

    # Test 1: Simple chain 0->1->2 (single path per pair)
    print("Test 1: Chain 0->1->2, single treatment")
    chain = nx.DiGraph([(0, 1), (1, 2)])
    # Only one path 0->1->2, length 2. Sum = 2.
    assert_close(compute_treatment_outcome_depth(chain, [0], 2), 2.0)
    # Only one path 0->1, length 1.
    assert_close(compute_treatment_outcome_depth(chain, [0], 1), 1.0)
    # Only one path 1->2, length 1.
    assert_close(compute_treatment_outcome_depth(chain, [1], 2), 1.0)
    print("   -> PASSED\n")

    # Test 2: No directed path → depth 0
    print("Test 2: No directed path (reverse direction)")
    assert_close(compute_treatment_outcome_depth(chain, [2], 0), 0.0)
    assert_close(compute_treatment_outcome_depth(chain, [1], 0), 0.0)
    print("   -> PASSED\n")

    # Test 3: Multiple directed paths (diamond)
    print("Test 3: Diamond — multiple paths summed")
    diamond = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    # Two paths from 0 to 3: 0->1->3 (len 2) and 0->2->3 (len 2). Sum = 4.
    assert_close(compute_treatment_outcome_depth(diamond, [0], 3), 4.0)
    # Single path from 1 to 3: 1->3 (len 1). Sum = 1.
    assert_close(compute_treatment_outcome_depth(diamond, [1], 3), 1.0)
    print("   -> PASSED\n")

    # Test 4: Triangle — two paths of different lengths
    print("Test 4: Triangle 0->1, 0->2, 1->2 — paths of different lengths")
    triangle = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    # Two paths from 0 to 2: 0->2 (len 1) and 0->1->2 (len 2). Sum = 3.
    assert_close(compute_treatment_outcome_depth(triangle, [0], 2), 3.0)
    print("   -> PASSED\n")

    # Test 5: Multiple treatments — average of per-treatment sums
    print("Test 5: Multiple treatments, averaged")
    # Diamond: T=[0,1], Y=3.
    # T=0: sum=4 (two paths, each len 2). T=1: sum=1 (one path len 1). Avg=(4+1)/2=2.5
    assert_close(compute_treatment_outcome_depth(diamond, [0, 1], 3), 2.5)
    # T=[1,2], Y=3: both have one path of len 1. Avg=(1+1)/2=1.0
    assert_close(compute_treatment_outcome_depth(diamond, [1, 2], 3), 1.0)
    print("   -> PASSED\n")

    # Test 6: Multiple treatments, some with no path
    print("Test 6: Mixed reachability")
    # T=[3], Y=0: no path. Sum=0.
    assert_close(compute_treatment_outcome_depth(diamond, [3], 0), 0.0)
    # T=[0, 3], Y=1: 0 has path 0->1 (sum=1), 3 has no path (sum=0). Avg=0.5.
    assert_close(compute_treatment_outcome_depth(diamond, [0, 3], 1), 0.5)
    print("   -> PASSED\n")

    # Test 7: Disconnected nodes
    print("Test 7: Disconnected graph")
    disconnected = nx.DiGraph()
    disconnected.add_nodes_from([0, 1, 2])
    disconnected.add_edge(0, 1)
    assert_close(compute_treatment_outcome_depth(disconnected, [0], 2), 0.0)
    print("   -> PASSED\n")

    # Test 8: Treatment equals outcome (no simple path of length > 0)
    print("Test 8: Treatment equals outcome")
    # all_simple_paths(G, 1, 1) returns no paths (source == target yields nothing).
    assert_close(compute_treatment_outcome_depth(chain, [1], 1), 0.0)
    print("   -> PASSED\n")

    # ==================================================================
    # Group 2: compute_effective_prob_conds
    # ==================================================================
    print("--- Group 2: effective_prob_conds ---\n")

    # Test 7: Chain 0->1->2, T=0, Y=2, no evidence
    # Mutilated: remove in-edges to 0 (none). Anc of {2} = {0,1}. cpd = {1,2}.
    # node 1: in_degree=1 (from 0) -> 2^1=2. node 2: in_degree=1 (from 1) -> 2^1=2.
    # Total = 4.
    print("Test 7: Chain T=0, Y=2")
    assert compute_effective_prob_conds(chain, [0], 2, {}) == 4
    print("   -> PASSED\n")

    # Test 8: Chain 0->1->2, T=1, Y=2
    # Mutilated: remove in-edges to 1 (edge 0->1 removed).
    # Anc of {2} in mutilated = {1} (only 1->2 remains upstream of 2). cpd = {2}.
    # node 2: in_degree=1 (from 1 in original) -> 2^1=2.
    # Total = 2.
    print("Test 8: Chain T=1, Y=2")
    assert compute_effective_prob_conds(chain, [1], 2, {}) == 2
    print("   -> PASSED\n")

    # Test 9: Diamond 0->1, 0->2, 1->3, 2->3, T=0, Y=3
    # Mutilated: remove in-edges to 0 (none, 0 is root).
    # Anc of {3} in mutilated = {0,1,2}. cpd = {1,2,3}.
    # node 1: in_degree=1 -> 2. node 2: in_degree=1 -> 2. node 3: in_degree=2 -> 4.
    # Total = 8.
    print("Test 9: Diamond T=0, Y=3")
    assert compute_effective_prob_conds(diamond, [0], 3, {}) == 8
    print("   -> PASSED\n")

    # Test 10: With evidence
    # Chain 0->1->2, T=0, Y=2, E={1:0}
    # Mutilated: same (0 has no in-edges). Relevant = {2, 1}.
    # Anc of {2,1} in mutilated = {0}. Union = {0,1,2}. cpd = {1,2}.
    # Same as test 7: total = 4.
    print("Test 10: Chain T=0, Y=2, E={1:0}")
    assert compute_effective_prob_conds(chain, [0], 2, {1: 0}) == 4
    print("   -> PASSED\n")

    # Test 11: JATE — multiple treatments
    # Diamond T=[1,2], Y=3
    # Mutilated: remove in-edges to 1 (0->1) and to 2 (0->2).
    # Anc of {3} in mutilated: only 1->3 and 2->3 remain, so anc={1,2}. Union={1,2,3}.
    # cpd = {3} (minus treatments {1,2}).
    # node 3: in_degree=2 in original -> 2^2=4.
    # Total = 4.
    print("Test 11: Diamond JATE T=[1,2], Y=3")
    assert compute_effective_prob_conds(diamond, [1, 2], 3, {}) == 4
    print("   -> PASSED\n")

    # Test 12: Root treatment, single-edge graph
    # 0->1, T=0, Y=1. Mutilated: no in-edges to remove.
    # Anc of {1} = {0}. cpd = {1}. node 1: in_degree=1 -> 2.
    print("Test 12: Single edge T=0, Y=1")
    single = nx.DiGraph([(0, 1)])
    assert compute_effective_prob_conds(single, [0], 1, {}) == 2
    print("   -> PASSED\n")

    # Test 13: Treatment with parents (non-root)
    # Confounder: 2->0, 2->1, 0->1. T=0, Y=1.
    # Mutilated: remove in-edges to 0: remove (2->0). Remaining: 2->1, 0->1.
    # Anc of {1} in mutilated = {0, 2}. Union = {0,1,2}. cpd = {2, 1} (minus {0}).
    # node 2: in_degree=0 -> 2^0=1. node 1: in_degree=2 (from 2,0 in original) -> 2^2=4.
    # Total = 5.
    print("Test 13: Confounder T=0, Y=1")
    conf = nx.DiGraph([(2, 0), (2, 1), (0, 1)])
    assert compute_effective_prob_conds(conf, [0], 1, {}) == 5
    print("   -> PASSED\n")

    # Test 14: Pruning removes non-ancestral nodes
    # 0->1, 2->3, T=0, Y=1 (node 2,3 disconnected from Y)
    # Mutilated: same. Anc of {1} = {0}. cpd = {1}. Total = 2.
    print("Test 14: Disconnected side branch pruned")
    pruning_dag = nx.DiGraph([(0, 1), (2, 3)])
    pruning_dag.add_nodes_from([0, 1, 2, 3])
    assert compute_effective_prob_conds(pruning_dag, [0], 1, {}) == 2
    print("   -> PASSED\n")

    # Test 15: Counterfactual-style effective probability conditions include
    # evidence-side ancestors in the original DAG.
    print("Test 15: Counterfactual metric includes evidence-side ancestors")
    cf_dag = nx.DiGraph([(2, 0), (0, 1), (2, 1)])
    assert compute_counterfactual_effective_prob_conds(cf_dag, [0], 1, {0: 1}) == 7
    print("   -> PASSED\n")

    # Test 16: Counterfactual metric matches simple chain expectations.
    print("Test 16: Counterfactual metric on chain with treatment evidence")
    assert compute_counterfactual_effective_prob_conds(chain, [0], 2, {0: 1}) == 5
    print("   -> PASSED\n")

    # Test 17: Counterfactual metric can keep treatment ancestors missed by ATE metric.
    print("Test 17: Counterfactual metric keeps treatment-side ancestors")
    att_dag = nx.DiGraph([(2, 0), (0, 1)])
    assert compute_counterfactual_effective_prob_conds(att_dag, [0], 1, {0: 1}) == 5
    print("   -> PASSED\n")

    # ==================================================================
    # Group 3: has_naive_conditioning
    # ==================================================================
    print("--- Group 3: has_naive_conditioning ---\n")

    # Test 15: Chain 0->1->2, T=0, Y=2
    # G_underline_T: remove out-edges of 0 -> removes 0->1. Left: 1->2.
    # d-sep({0}, {2}, {}) = True (no active path).
    print("Test 15: Chain T=0, Y=2 (shortcut)")
    assert has_naive_conditioning(chain, [0], 2, []) is True
    print("   -> PASSED\n")

    # Test 16: Confounder 2->0, 2->1, 0->1, T=0, Y=1
    # G_underline_T: remove out-edges of 0 -> removes 0->1. Left: 2->0, 2->1.
    # Path: 0<-2->1 (fork at 2, unobserved). Active path exists.
    # d-sep({0}, {1}, {}) = False.
    print("Test 16: Confounder T=0, Y=1 (no shortcut)")
    assert has_naive_conditioning(conf, [0], 1, []) is False
    print("   -> PASSED\n")

    # Test 17: Confounder with evidence blocking the fork
    # Same confounder, E={2}. Conditioning on 2 blocks the fork path.
    # d-sep({0}, {1}, {2}) = True.
    print("Test 17: Confounder T=0, Y=1, E={2} (shortcut via blocking)")
    assert has_naive_conditioning(conf, [0], 1, [2]) is True
    print("   -> PASSED\n")

    # Test 18: Root treatment (trivial shortcut)
    # If T is a root node and we remove its out-edges, T becomes isolated.
    # Isolated node is always d-sep from everything.
    print("Test 18: Root treatment always shortcut")
    root_dag = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    assert has_naive_conditioning(root_dag, [0], 3, []) is True
    print("   -> PASSED\n")

    # Test 19: Fork structure, T=1, Y=2 (non-root, common cause)
    # 0->1, 0->2. G_underline_T: remove out-edges of 1 (none, 1 is leaf).
    # Left: 0->1, 0->2. Path: 1<-0->2, fork, 0 unobserved -> active.
    # d-sep = False.
    print("Test 19: Fork T=1, Y=2 (no shortcut)")
    fork = nx.DiGraph([(0, 1), (0, 2)])
    assert has_naive_conditioning(fork, [1], 2, []) is False
    print("   -> PASSED\n")

    # Test 20: Fork with common cause observed
    # E={0}. Conditioning on 0 blocks the fork.
    print("Test 20: Fork T=1, Y=2, E={0} (shortcut)")
    assert has_naive_conditioning(fork, [1], 2, [0]) is True
    print("   -> PASSED\n")

    # Test 21: Collider structure
    # 0->2, 1->2. G_underline_T (T=0): remove out-edges of 0 -> removes 0->2.
    # Left: 1->2. d-sep({0}, {2}, {}) = True (no active path).
    print("Test 21: Collider T=0, Y=2 (shortcut)")
    collider = nx.DiGraph([(0, 2), (1, 2)])
    assert has_naive_conditioning(collider, [0], 2, []) is True
    print("   -> PASSED\n")

    # Test 22: Collider with evidence opening path
    # E={2}. Conditioning on collider 2 opens path 0->2<-1.
    # But we only have T=0, Y... wait. We need to think about what outcome is.
    # T=0, Y=1, E={2}. G_underline_T: remove out-edges of 0 -> removes 0->2.
    # Left: 1->2. Path from 0 to 1: 0...2...1? In G_underline_T, no edge from 0.
    # Only edge is 1->2. No path 0 to 1 at all. d-sep = True.
    # But let's use a structure where conditioning opens a path:
    # 0->2, 1->2, 1->3. T=0, Y=3, E={2}.
    # G_underline_T: remove out-edges of 0 -> removes 0->2. Left: 1->2, 1->3.
    # d-sep({0}, {3}, {2}): paths from 0 to 3? Only through 2: 0...2<-1->3.
    # But 0->2 is removed. No path from 0 to 3 or 2 in G_underline_T. d-sep = True.
    # Hmm, hard to create a case with collider opening. Let me try:
    # 0->2, 1->2, 2->3. T=1, Y=3, E={2}.
    # G_underline_T: remove out-edges of 1 -> removes 1->2. Left: 0->2, 2->3.
    # d-sep({1}, {3}, {2}): 1 is isolated (no edges in/out after removal).
    # d-sep = True.
    # Let me try: 0->1, 2->1, 1->3, 2->3. T=0, Y=3, E={1}.
    # G_underline_T: remove out-edges of 0 -> removes 0->1. Left: 2->1, 1->3, 2->3.
    # d-sep({0}, {3}, {1}): 0 has no edges in G_underline_T. d-sep = True.
    print("Test 22: JATE T=[0,1], Y=2 (two treatments)")
    fork2 = nx.DiGraph([(0, 2), (1, 2)])
    # G_underline_T: remove out-edges of 0 and 1 -> remove 0->2, 1->2. No edges left.
    # d-sep({0,1}, {2}, {}) = True.
    assert has_naive_conditioning(fork2, [0, 1], 2, []) is True
    print("   -> PASSED\n")

    # Test 23: Chain with mediator, no shortcut
    # 0->1->2, 3->1. T=3, Y=2, no evidence.
    # G_underline_T: remove out-edges of 3 -> removes 3->1. Left: 0->1, 1->2.
    # d-sep({3}, {2}, {}) = True (3 disconnected in G_underline_T).
    print("Test 23: Side-cause T=3, Y=2 (shortcut, T disconnected)")
    side_dag = nx.DiGraph([(0, 1), (1, 2), (3, 1)])
    assert has_naive_conditioning(side_dag, [3], 2, []) is True
    print("   -> PASSED\n")

    # Test 24: No shortcut — instrument-like structure
    # 0->1, 1->2, 3->1, 3->2. T=1, Y=2.
    # G_underline_T: remove out-edges of 1 -> removes 1->2. Left: 0->1, 3->1, 3->2.
    # Path 1<-3->2, fork at 3, unobserved -> active.
    # d-sep({1}, {2}, {}) = False.
    print("Test 24: Confounded mediator (no shortcut)")
    inst = nx.DiGraph([(0, 1), (1, 2), (3, 1), (3, 2)])
    assert has_naive_conditioning(inst, [1], 2, []) is False
    print("   -> PASSED\n")

    # Test 25: Same structure, observe confounder → shortcut
    print("Test 25: Confounded mediator, E={3} (shortcut)")
    assert has_naive_conditioning(inst, [1], 2, [3]) is True
    print("   -> PASSED\n")

    # ==================================================================
    # Group 4: Cross-metric consistency
    # ==================================================================
    print("--- Group 4: Cross-Metric Consistency ---\n")

    # Test 26: Abnormal pair (no T→Y path) → depth 0
    print("Test 26: Abnormal pair has depth 0")
    assert_close(compute_treatment_outcome_depth(chain, [2], 0), 0.0)
    # Also check effective_prob_conds for abnormal: T=2, Y=0 in chain
    # Mutilated: remove in-edges to 2 (1->2 removed). Anc of {0} in mutilated = {}.
    # cpd = {0} - {2} = {0}. node 0: in_degree=0 -> 2^0=1. Total = 1.
    assert compute_effective_prob_conds(chain, [2], 0, {}) == 1
    print("   -> PASSED\n")

    # Test 27: Original DAG is not mutated by any function
    print("Test 27: Functions do not mutate input DAG")
    dag_copy = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    edges_before = set(dag_copy.edges())
    compute_treatment_outcome_depth(dag_copy, [0], 3)
    assert set(dag_copy.edges()) == edges_before
    compute_effective_prob_conds(dag_copy, [1], 3, {0: 1})
    assert set(dag_copy.edges()) == edges_before
    has_naive_conditioning(dag_copy, [1], 3, [0])
    assert set(dag_copy.edges()) == edges_before
    print("   -> PASSED\n")

    # Test 28: Larger graph (8 nodes), end-to-end
    print("Test 28: 8-node graph, all metrics")
    big = nx.DiGraph([
        (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 7),
    ])
    # T=0, Y=7: one path 0->2->3->4->5->6->7 (len 6). Sum = 6.
    assert_close(compute_treatment_outcome_depth(big, [0], 7), 6.0)
    # T=1, Y=7: two paths — 1->7 (len 1) and 1->2->3->4->5->6->7 (len 6). Sum = 7.
    assert_close(compute_treatment_outcome_depth(big, [1], 7), 7.0)
    # T=[0,1], Y=7: avg = (6+7)/2 = 6.5
    assert_close(compute_treatment_outcome_depth(big, [0, 1], 7), 6.5)
    # Effective prob conds, T=0, Y=7: mutilate removes in-edges to 0 (none).
    # All nodes 0..7 connected, so all are ancestors of 7.
    # cpd = {1,2,3,4,5,6,7}. Parent counts in original:
    #   1:0->1, 2:2(from 0,1), 3:1(from 2), 4:1, 5:1, 6:1, 7:2(from 6,1)
    #   2^0+2^2+2^1+2^1+2^1+2^1+2^2 = 1+4+2+2+2+2+4 = 17
    epc = compute_effective_prob_conds(big, [0], 7, {})
    assert epc == 17, f"Expected 17, got {epc}"
    print("   -> PASSED\n")

    print("=== All Difficulty Metrics Tests Passed ===\n")
