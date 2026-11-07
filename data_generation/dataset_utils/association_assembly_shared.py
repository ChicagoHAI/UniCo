"""
Shared question-assembly helpers for standalone association-derived families.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx

from data_generation.ate.ate_question_assembly import _node_label, render_dag_edges
from data_generation.scm_mechanism_sampling import SCMBase


@dataclass
class ConditionVariant:
    variant_name: str
    conditions_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]


def render_node_cpt(scm: SCMBase, node: int, *, decimal_places: int = 2) -> str:
    cpt_entry = scm.cpts[node]
    parents = cpt_entry["parents"]
    table = cpt_entry["table"]
    node_label = _node_label(node)
    if not parents:
        return f"P({node_label} = 1) = {table[()]:.{decimal_places}f}"
    parent_labels = [_node_label(parent) for parent in parents]
    lines = [f"P({node_label} = 1 | {', '.join(parent_labels)}):"]
    for config in itertools.product([0, 1], repeat=len(parents)):
        assignments = ", ".join(
            f"{parent_label}={value}" for parent_label, value in zip(parent_labels, config)
        )
        lines.append(f"  {assignments}: {table[config]:.{decimal_places}f}")
    return "\n".join(lines)


def render_graph_text(scm: SCMBase, graph_variant: int) -> str:
    node_labels = ", ".join(_node_label(node) for node in scm.nodes)
    edges = sorted(scm.dag.edges())
    if graph_variant == 1:
        return (
            "Imagine a self-contained world with only the following causal graph, and no unmentioned variables or causal relationships:\n"
            f"Variables: {node_labels}\n"
            f"Edges: {render_dag_edges(scm.dag)}"
        )
    if graph_variant == 2:
        edge_lines = "\n".join(
            f"{_node_label(source)} directly causes {_node_label(target)}."
            for source, target in edges
        )
        return (
            f"Consider binary variables {node_labels} with the following direct causal relationships:\n"
            f"{edge_lines}\n"
            "These are the only direct causal relationships in the system."
        )
    if graph_variant == 3:
        parent_lines = []
        for node in scm.nodes:
            parents = sorted(scm.dag.predecessors(node))
            if parents:
                parent_lines.append(
                    f"{_node_label(node)} has parents: {{{', '.join(_node_label(parent) for parent in parents)}}}"
                )
            else:
                parent_lines.append(f"{_node_label(node)} has no parents")
        parent_block = "\n".join(parent_lines)
        return (
            f"Let {node_labels} be binary variables in a DAG with exhaustive parent sets:\n"
            f"{parent_block}"
        )
    raise ValueError(f"Unknown graph_variant: {graph_variant}")


def topo_sorted_subset(dag: nx.DiGraph, nodes: Set[int]) -> List[int]:
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(
        (source, target)
        for source, target in dag.edges()
        if source in nodes and target in nodes
    )
    return list(nx.topological_sort(subgraph))


def ancestor_closure(dag: nx.DiGraph, nodes: Set[int]) -> Set[int]:
    closed_nodes = set(nodes)
    for node in list(nodes):
        closed_nodes.update(nx.ancestors(dag, node))
    return closed_nodes


def build_condition_variant(
    scm: SCMBase,
    kept_nodes: Set[int],
    *,
    variant_name: str,
    decimal_places: int,
) -> ConditionVariant:
    topo_nodes = topo_sorted_subset(scm.dag, kept_nodes)
    conditions_text = "\n".join(
        render_node_cpt(scm, node, decimal_places=decimal_places)
        for node in topo_nodes
    )
    all_nodes = set(scm.nodes)
    return ConditionVariant(
        variant_name=variant_name,
        conditions_text=conditions_text,
        nodes_kept=sorted(kept_nodes),
        nodes_pruned=sorted(all_nodes - kept_nodes),
    )


def numerical_condition_variants(
    scm: SCMBase,
    core_nodes: Set[int],
    *,
    decimal_places: int,
) -> List[ConditionVariant]:
    ancestor_nodes = ancestor_closure(scm.dag, core_nodes)
    return [
        build_condition_variant(scm, set(scm.nodes), variant_name="full", decimal_places=decimal_places),
        build_condition_variant(scm, ancestor_nodes, variant_name="query_ancestor_prune", decimal_places=decimal_places),
    ]


def structural_condition_variants(
    scm: SCMBase,
    focus_nodes: Set[int],
    *,
    prune_text: str,
) -> List[ConditionVariant]:
    all_nodes = set(scm.nodes)
    focus_with_ancestors = ancestor_closure(scm.dag, focus_nodes)
    full_text = "No probability tables are needed for this structural query. Use the DAG only."
    return [
        ConditionVariant("full", full_text, sorted(all_nodes), []),
        ConditionVariant("query_ancestor_prune", prune_text, sorted(focus_with_ancestors), sorted(all_nodes - focus_with_ancestors)),
    ]


def render_assignment(node_values: Dict[int, int]) -> str:
    return ", ".join(
        f"{_node_label(node)} = {value}" for node, value in sorted(node_values.items())
    )
