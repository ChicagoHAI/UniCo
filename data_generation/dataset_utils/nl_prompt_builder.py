"""NL domain prompt builder for causal reasoning question translation.

Constructs prompts for the v2 (two-step) and v3 (three-step) NL domain
translation pipelines.

All functions accept serialised triplet data (``dag_edges``, ``cpts`` dict,
etc.) rather than live SCM objects, so the module is independent of
``scm_mechanism_sampling``.
"""

from __future__ import annotations

import ast
import itertools
import json
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from data_generation.dataset_utils.nl_query_specs import (
    is_v2_2_query_type,
    selector_assumptions_required,
    get_query_spec,
)

# ---------------------------------------------------------------------------
# Node label helper
# ---------------------------------------------------------------------------

def node_label(node: int) -> str:
    return f"X{node}"


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------

def build_dag(dag_edges: List[List[int]], num_nodes: int) -> nx.DiGraph:
    """Build a ``networkx.DiGraph`` from serialised edges."""
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_nodes))
    dag.add_edges_from(tuple(e) for e in dag_edges)
    return dag


def topo_sorted_nodes(dag: nx.DiGraph) -> List[int]:
    """Return nodes in a *deterministic* topological order.

    Within each topological generation, nodes are sorted by their integer
    ID so the ordering is reproducible.
    """
    result: List[int] = []
    for generation in nx.topological_generations(dag):
        result.extend(sorted(generation))
    return result


# ---------------------------------------------------------------------------
# DAG edge rendering (topological order)
# ---------------------------------------------------------------------------

def render_dag_edges_topo(dag: nx.DiGraph) -> str:
    """Render edges sorted by topological position of source, then target.

    Example output::

        X7 -> X6, X7 -> X1, X7 -> X2, X1 -> X2, X1 -> X4, ...
    """
    topo_order = topo_sorted_nodes(dag)
    pos = {n: i for i, n in enumerate(topo_order)}
    edges = sorted(dag.edges(), key=lambda e: (pos[e[0]], pos[e[1]]))
    if not edges:
        return "(no edges)"
    return ", ".join(f"{node_label(u)} -> {node_label(v)}" for u, v in edges)


def render_node_labels_topo(dag: nx.DiGraph) -> str:
    """Comma-separated node labels in topological order."""
    return ", ".join(node_label(n) for n in topo_sorted_nodes(dag))


# ---------------------------------------------------------------------------
# CPT rendering from triplet data
# ---------------------------------------------------------------------------

def _parse_table_key(key_str: str) -> Tuple[int, ...]:
    """Parse a stringified tuple key like ``'(0,)'`` or ``'(0, 1)'``."""
    return ast.literal_eval(key_str)


def render_node_cpt_from_triplet(
    node: int,
    cpt_entry: Dict[str, Any],
    decimal_places: int,
) -> str:
    """Render one node's CPT from triplet-serialised ``cpts[node]``.

    Matches the format of ``ate_question_assembly.render_node_cpt``.
    """
    parents: List[int] = cpt_entry["parents"]
    table: Dict[str, float] = cpt_entry["table"]
    nl = node_label(node)

    if not parents:
        p = table["()"]
        return f"P({nl} = 1) = {p:.{decimal_places}f}"

    parent_labels = [node_label(p) for p in parents]
    header = f"P({nl} = 1 | {', '.join(parent_labels)}):"

    lines = [header]
    configs = list(itertools.product([0, 1], repeat=len(parents)))
    for config in configs:
        key_str = str(config)
        p = table[key_str]
        assignments = ", ".join(
            f"{pl}={v}" for pl, v in zip(parent_labels, config)
        )
        lines.append(f"  {assignments}: {p:.{decimal_places}f}")

    return "\n".join(lines)


def render_scm_text(
    cpts: Dict[str, Dict[str, Any]],
    dag: nx.DiGraph,
    decimal_places: int,
    nodes: Optional[List[int]] = None,
) -> str:
    """Render CPTs for *nodes* (default: all) in topological order."""
    topo = topo_sorted_nodes(dag)
    if nodes is not None:
        node_set = set(nodes)
        topo = [n for n in topo if n in node_set]

    blocks = [
        render_node_cpt_from_triplet(n, cpts[str(n)], decimal_places)
        for n in topo
    ]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Exogenous selector schema rendering
# ---------------------------------------------------------------------------

def _selector_parent_assignment_keys(
    cpts: Dict[str, Dict[str, Any]],
    node: int,
) -> List[str]:
    entry = cpts[str(node)]
    parents = entry["parents"]
    if not parents:
        return ["()"]
    return [str(config) for config in itertools.product([0, 1], repeat=len(parents))]


def _selector_assignment_label(
    cpts: Dict[str, Dict[str, Any]],
    node: int,
    key: str,
) -> str:
    parents = cpts[str(node)]["parents"]
    if not parents:
        return "()"
    config = _parse_table_key(key)
    return "(" + ", ".join(
        f"{node_label(parent)}={value}"
        for parent, value in zip(parents, config)
    ) + ")"


def _selector_assumption_text() -> str:
    return (
        "Selector-variable assumption: each variable v is assumed to depend "
        "deterministically on its parents pa(v) and a collection of mutually "
        "independent exogenous selector variables, one selector U_{v,a} for "
        "each possible joint assignment a to pa(v). For each assignment a, "
        "P(U_{v,a}=1)=P(v=1 | pa(v)=a), and when pa(v)=a, the variable v "
        "takes the value U_{v,a}."
    )


def _selector_prompt_instruction() -> str:
    return (
        "Because this causal graph is intended to represent the context of a "
        "counterfactual query type, the entity mapping must also represent "
        "the selector-variable assumption stated above. For each graph node, "
        "provide an `exogenous_selectors` object with one selector entry for "
        "each possible joint assignment to that node's parents. These selector "
        "entries are not additional observed graph variables; they are latent "
        "response-pattern entities used to explain the deterministic-selector "
        "SCM assumption. Each selector entry must have `entity`, `0`, and `1` "
        "fields, where `0` and `1` align with the same node's binary values "
        "under that parent assignment."
    )


def _selector_schema_lines(
    dag: nx.DiGraph,
    cpts: Dict[str, Dict[str, Any]],
    *,
    include_node_interpretations: bool,
    copy_node_entity: bool = False,
) -> str:
    def render_node_block(node: int) -> str:
        label = node_label(node)
        entity_text = (
            f"{{entity for {label}, copied from above}}"
            if copy_node_entity else f"{{entity for {label}}}"
        )
        lines = [
            f'  "{label}": {{',
            f'    "entity": "{entity_text}",',
        ]
        if include_node_interpretations:
            lines.extend([
                f'    "0": "{{interpretation for {label}=0}}",',
                f'    "1": "{{interpretation for {label}=1}}",',
            ])
        lines.append('    "exogenous_selectors": {')
        selector_keys = _selector_parent_assignment_keys(cpts, node)
        for key_idx, key in enumerate(selector_keys):
            assignment_label = _selector_assignment_label(cpts, node, key)
            comma = "," if key_idx < len(selector_keys) - 1 else ""
            lines.extend([
                f'      "{assignment_label}": {{',
                f'        "entity": "{{selector entity for {label} when parents are {assignment_label}}}",',
                f'        "0": "{{meaning of selector value 0 for {label} under {assignment_label}}}",',
                f'        "1": "{{meaning of selector value 1 for {label} under {assignment_label}}}"',
                f"      }}{comma}",
            ])
        lines.append("    }")
        lines.append("  }")
        return "\n".join(lines)

    topo = topo_sorted_nodes(dag)
    if not topo:
        return "{}"
    if len(topo) == 1:
        return "{\n" + render_node_block(topo[0]) + "\n}"

    first_block = render_node_block(topo[0])
    last_block = render_node_block(topo[-1])
    return "{\n" + first_block + ",\n  ...\n" + last_block + "\n}"


# ---------------------------------------------------------------------------
# Query text rendering (NL version — self-contained sentence only)
# ---------------------------------------------------------------------------

def _determine_query_type(
    treatments: List[int],
    evidence: Dict[int, int],
) -> str:
    if len(treatments) == 1 and evidence:
        return "cate"
    if len(treatments) >= 2 and not evidence:
        return "jate"
    return "ate"


def render_query_text_nl(
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    query_variant: int,
) -> str:
    """Render the self-contained NL query sentence (no ATE/CATE/JATE terms).

    This is the second line of the symbolic query text with the connector
    prefix ("That is, " / "In other words, ") stripped and capitalised.
    """
    query_type = _determine_query_type(treatments, evidence)
    o_label = node_label(outcome)
    sorted_treatments = sorted(treatments)
    t_labels = [node_label(t) for t in sorted_treatments]

    if query_type == "ate":
        t_label = t_labels[0]
        if query_variant == 1:
            return (
                f"How does the probability of {o_label} = 1 change "
                f"if we force {t_label} = 1 compared to forcing "
                f"{t_label} = 0?"
            )
        elif query_variant == 2:
            return (
                f"What is the difference in P({o_label} = 1) when "
                f"{t_label} is set to 1 versus set to 0, regardless "
                f"of how {t_label} would naturally arise?"
            )
        elif query_variant == 3:
            return (
                f"How much does intervening to change {t_label} from "
                f"0 to 1 affect the probability that {o_label} = 1?"
            )

    elif query_type == "cate":
        t_label = t_labels[0]
        sorted_evidence = sorted(evidence.items())
        given_clause = ", ".join(
            f"{node_label(e)} = {v}" for e, v in sorted_evidence
        )
        if query_variant == 1:
            return (
                f"Among cases where {given_clause}, how does the "
                f"probability of {o_label} = 1 change if we force "
                f"{t_label} = 1 compared to forcing {t_label} = 0?"
            )
        elif query_variant == 2:
            return (
                f"Among cases where {given_clause}, what is the "
                f"difference in P({o_label} = 1) when {t_label} is "
                f"set to 1 versus set to 0, regardless of how "
                f"{t_label} would naturally arise?"
            )
        elif query_variant == 3:
            return (
                f"Among cases where {given_clause}, how much does "
                f"intervening to change {t_label} from 0 to 1 affect "
                f"the probability that {o_label} = 1?"
            )

    else:  # jate
        t_list_str = " and ".join(t_labels)
        do_1_str = " and ".join(f"{tl} = 1" for tl in t_labels)
        do_0_str = " and ".join(f"{tl} = 0" for tl in t_labels)
        both_or_all = "both" if len(treatments) == 2 else "all"

        if query_variant == 1:
            return (
                f"How does the probability of {o_label} = 1 change "
                f"if we simultaneously force {do_1_str}, compared to "
                f"simultaneously forcing {do_0_str}?"
            )
        elif query_variant == 2:
            return (
                f"What is the difference in P({o_label} = 1) when "
                f"{t_list_str} are jointly set to 1 versus jointly "
                f"set to 0, regardless of how they would naturally "
                f"arise?"
            )
        elif query_variant == 3:
            return (
                f"How much does simultaneously intervening to set "
                f"{do_1_str}, compared to setting {both_or_all} to 0, "
                f"affect the probability that {o_label} = 1?"
            )

    raise ValueError(f"Unknown query_variant: {query_variant}")


# ---------------------------------------------------------------------------
# Condition variant node selection (pure networkx, no pgmpy needed)
# ---------------------------------------------------------------------------

def get_full_condition_nodes(dag: nx.DiGraph) -> List[int]:
    """All nodes in topological order (full condition variant)."""
    return topo_sorted_nodes(dag)


def get_surgery_ancestor_prune_nodes_ate(
    dag: nx.DiGraph,
    treatment: int,
    outcome: int,
) -> List[int]:
    """Nodes to keep for surgery_ancestor_prune (ATE).

    1. Remove edges into treatment (graph surgery).
    2. Keep ancestors of outcome in the mutilated graph + outcome itself.
    3. Exclude treatment from CPD nodes.
    """
    mutilated = dag.copy()
    for parent in list(mutilated.predecessors(treatment)):
        mutilated.remove_edge(parent, treatment)

    ancestors = nx.ancestors(mutilated, outcome) | {outcome}
    cpd_nodes = ancestors - {treatment}
    return sorted(cpd_nodes)


def get_surgery_ancestor_prune_nodes_cate_jate(
    dag: nx.DiGraph,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
) -> List[int]:
    """Nodes to keep for surgery_ancestor_prune (CATE/JATE).

    1. Remove edges into all treatment nodes (graph surgery).
    2. Relevant target nodes = outcome + evidence keys.
    3. Keep ancestors of all relevant nodes in mutilated graph + the nodes.
    4. Exclude all treatment nodes from CPD nodes.
    """
    mutilated = dag.copy()
    for t in treatments:
        for parent in list(mutilated.predecessors(t)):
            mutilated.remove_edge(parent, t)

    relevant_nodes = [outcome] + sorted(evidence.keys())
    all_ancestors = set()
    for node in relevant_nodes:
        all_ancestors |= nx.ancestors(mutilated, node) | {node}

    cpd_nodes = all_ancestors - set(treatments)
    return sorted(cpd_nodes)


def derive_query_variant(example_seed: int) -> int:
    """Derive query_variant from example_seed using same RNG sequence as
    symbolic assembly (graph_variant draw discarded, query_variant kept)."""
    rng = random.Random(example_seed)
    _ = rng.randint(1, 3)       # graph_variant slot (discarded for NL)
    return rng.randint(1, 3)    # query_variant


# ---------------------------------------------------------------------------
# Symbolic question builder (for question generation prompt)
# ---------------------------------------------------------------------------

def build_symbolic_question_text(
    dag: nx.DiGraph,
    cpts: Dict[str, Dict[str, Any]],
    decimal_places: int,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
    query_variant: int,
    condition_nodes: Optional[List[int]] = None,
) -> str:
    """Build the full symbolic question embedded in the question gen prompt.

    Parameters
    ----------
    condition_nodes : list[int] | None
        Nodes whose CPTs to include.  ``None`` means all nodes (full variant).
    """
    node_labels_str = render_node_labels_topo(dag)
    edges_str = render_dag_edges_topo(dag)
    conditions_text = render_scm_text(
        cpts, dag, decimal_places, nodes=condition_nodes,
    )
    query_text = render_query_text_nl(
        treatments, outcome, evidence, query_variant,
    )

    return (
        "Imagine a self-contained, hypothetical world with only the "
        "following conditions, and without any unmentioned factors or "
        "causal relationships:\n"
        f"Given the following causal graph among variables {node_labels_str}:\n"
        f"{edges_str}\n"
        "\n"
        "The following probability relationships hold:\n"
        f"{conditions_text}\n"
        "\n"
        f"{query_text}"
    )


# ---------------------------------------------------------------------------
# v2 Step 1 — Entity assignment + interpretation
# ---------------------------------------------------------------------------

_V2_STEP1_INSTRUCTION = (
    "I want you to convert each graph node above into a real-world entity, "
    "and it is required that after the conversion, the causal and probability "
    "relationships above still hold under the commonsense for these real-world "
    "entities. Notably, you should traverse the graph in topological ordering "
    "as shown above, and appropriately determine what real-world "
    "interpretations each node's 0 and 1 values should have, so "
    "that the relative scales of these probabilities make sense. For example, "
    'if "X4 -> X2" and P(X2=1 | X4=1) < P(X2=1 | X4=0) hold, then the '
    "following two options are both reasonable interpretations:\n"
    '(1) X4\'s entity is "Advancements in artificial intelligence", X2\'s '
    'entity is "Demand for Nvidia\'s chips", "X2=1" is interpreted as '
    '"Demand for Nvidia\'s chips is strong", "X4=1" as "AI advancements '
    'are limited or absent", and "X4=0" as "AI is advancing rapidly".\n'
    '(2) X4\'s entity is "the market has already largely priced in Nvidia\'s '
    'AI growth story", X2\'s entity is "fresh AI-driven demand for Nvidia '
    'chips is still accelerating", "X2=1" is interpreted as "there is a '
    'fresh acceleration in AI-driven chip demand", "X4=1" as "Nvidia\'s AI '
    'growth story is already largely priced into the market", and "X4=0" as '
    '"Nvidia\'s AI growth story is not yet fully priced into the market".'
)

_V2_STEP1_PASSAGE_INTRO = (
    "Moreover, you should derive the appropriate entities from the background "
    "revealed in the following passage. You may come up with new entities if "
    "the ones in the passage cannot constitute the given causal graph, but "
    "make sure they still reasonably fit into the background."
)

def _v2_step1_output_format(
    dag: nx.DiGraph,
    cpts: Optional[Dict[str, Dict[str, Any]]] = None,
    query_type: Optional[str] = None,
) -> str:
    """Build v2 step 1 output schema with concrete node names."""
    if query_type and selector_assumptions_required(query_type):
        if cpts is None:
            raise ValueError("cpts are required for selector-aware output schema")
        return (
            "Return the node-entity mapping in json format, with the nodes "
            "arranged in topological ordering. For each node, return the "
            "node entity, the real-world interpretations of its binary "
            "assignment, and the exogenous selector entries exactly in the "
            "nested shape below:\n"
            "\n"
            "```json\n"
            f"{_selector_schema_lines(dag, cpts, include_node_interpretations=True)}\n"
            "```"
        )
    topo = topo_sorted_nodes(dag)
    first, last = node_label(topo[0]), node_label(topo[-1])
    return (
        "Return the node-entity mapping in json format, with the nodes arranged "
        "in topological ordering. Note that for each node, you should return "
        "(1) the name of its real-world entity, and (2) the real-world "
        "interpretations of its binary assignment.\n"
        "\n"
        "```json\n"
        "{\n"
        f'  "{first}": {{\n'
        f'    "entity": "{{entity for {first}}}",\n'
        f'    "0": "{{interpretation for {first}=0}}",\n'
        f'    "1": "{{interpretation for {first}=1}}"\n'
        "  },\n"
        "  ...\n"
        f'  "{last}": {{\n'
        f'    "entity": "{{entity for {last}}}",\n'
        f'    "0": "{{interpretation for {last}=0}}",\n'
        f'    "1": "{{interpretation for {last}=1}}"\n'
        "  }\n"
        "}\n"
        "```"
    )


def build_v2_step1_prompt(
    dag: nx.DiGraph,
    cpts: Dict[str, Dict[str, Any]],
    decimal_places: int,
    passage_text: str,
    query_type: Optional[str] = None,
) -> str:
    """Build v2 step 1 prompt: entity assignment + interpretation."""
    node_labels_str = render_node_labels_topo(dag)
    edges_str = render_dag_edges_topo(dag)
    scm_text = render_scm_text(cpts, dag, decimal_places)
    selector_assumption_text = ""
    selector_instruction = ""
    if query_type and selector_assumptions_required(query_type):
        selector_assumption_text = "\n\n" + _selector_assumption_text()
        selector_instruction = "\n\n" + _selector_prompt_instruction()

    return (
        f"Given the following causal graph among variables {node_labels_str}, "
        "where each variable only takes binary values, along with the "
        "probability relationships among them:\n"
        "\n"
        '"""\n'
        f"{edges_str}\n"
        "\n"
        f"{scm_text}"
        f"{selector_assumption_text}\n"
        '"""\n'
        "\n"
        f"{_V2_STEP1_INSTRUCTION}{selector_instruction}\n"
        "\n"
        f"{_V2_STEP1_PASSAGE_INTRO}\n"
        "\n"
        '"""\n'
        f"{passage_text}\n"
        '"""\n'
        "\n"
        f"{_v2_step1_output_format(dag, cpts, query_type)}"
    )


# ---------------------------------------------------------------------------
# v3 Step 1 — Entity assignment only
# ---------------------------------------------------------------------------

_V3_STEP1_INSTRUCTION = (
    "I want you to traverse the graph in topological ordering as shown above, "
    "and convert each graph node above into a real-world entity. Notably, it "
    "is required that after the conversion, the causal relationships above "
    "still hold under the commonsense for these real-world entities. Moreover, "
    "you should derive the appropriate entities from the background revealed "
    "in the following passage. You may come up with new entities if the ones "
    "in the passage cannot constitute the given causal graph, but make sure "
    "they still reasonably fit into the background."
)

def _v3_step1_output_format(
    dag: nx.DiGraph,
    cpts: Optional[Dict[str, Dict[str, Any]]] = None,
    query_type: Optional[str] = None,
) -> str:
    """Build v3 step 1 output schema with concrete node names."""
    if query_type and selector_assumptions_required(query_type):
        if cpts is None:
            raise ValueError("cpts are required for selector-aware output schema")
        return (
            "Return the node-entity mapping in json format, with the nodes "
            "arranged in topological ordering. For each node, return the "
            "node entity and its exogenous selector entries exactly in the "
            "nested shape below. Even though this step does not interpret "
            "the graph node's 0/1 values yet, each selector entry must still "
            "include `0` and `1` meanings aligned with the eventual node "
            "value under that parent assignment:\n"
            "\n"
            "```json\n"
            f"{_selector_schema_lines(dag, cpts, include_node_interpretations=False)}\n"
            "```"
        )
    topo = topo_sorted_nodes(dag)
    first, last = node_label(topo[0]), node_label(topo[-1])
    return (
        "Return the node-entity mapping in json format, with the nodes arranged "
        "in topological ordering, as below:\n"
        "\n"
        "```json\n"
        "{\n"
        f'  "{first}": {{\n'
        f'    "entity": "{{entity for {first}}}"\n'
        "  },\n"
        "  ...\n"
        f'  "{last}": {{\n'
        f'    "entity": "{{entity for {last}}}"\n'
        "  }\n"
        "}\n"
        "```"
    )


def build_v3_step1_prompt(
    dag: nx.DiGraph,
    passage_text: str,
    cpts: Optional[Dict[str, Dict[str, Any]]] = None,
    query_type: Optional[str] = None,
) -> str:
    """Build v3 step 1 prompt: entity assignment (no CPTs)."""
    node_labels_str = render_node_labels_topo(dag)
    edges_str = render_dag_edges_topo(dag)
    selector_assumption_text = ""
    selector_instruction = ""
    if query_type and selector_assumptions_required(query_type):
        selector_assumption_text = "\n\n" + _selector_assumption_text()
        selector_instruction = "\n\n" + _selector_prompt_instruction()

    return (
        f"Given the following causal graph among variables {node_labels_str}, "
        "where each variable only takes binary values (0/1):\n"
        "\n"
        '"""\n'
        f"{edges_str}"
        f"{selector_assumption_text}\n"
        '"""\n'
        "\n"
        f"{_V3_STEP1_INSTRUCTION}{selector_instruction}\n"
        "\n"
        '"""\n'
        f"{passage_text}\n"
        '"""\n'
        "\n"
        f"{_v3_step1_output_format(dag, cpts, query_type)}"
    )


# ---------------------------------------------------------------------------
# v3 Step 2 — Entity interpretation aligned with probability conditions
# ---------------------------------------------------------------------------

_V3_STEP2_INSTRUCTION_PREFIX = (
    "And also given the following mapping from each graph node into a "
    "real-world entity, which preserves the above inter-node causal "
    "relationships under the commonsense:"
)

_V3_STEP2_INSTRUCTION_BODY = (
    "I want you to traverse the graph in topological ordering as shown above, "
    "and appropriately determine what real-world interpretations each node's "
    "0 and 1 values should have, so that the relative scales of "
    "the probability conditions given above still make sense under the "
    'commonsense. For example, if "X4 -> X2" and P(X2=1 | X4=1) < '
    "P(X2=1 | X4=0) hold, X4's entity is \"Advancements in artificial "
    'intelligence", X2\'s entity is "Demand for Nvidia\'s chips", then a '
    'reasonable option is to interpret "X2=1" as "Demand for Nvidia\'s chips '
    'is strong", "X4=1" as "AI advancements are limited or absent", and '
    '"X4=0"  as "AI is advancing rapidly". Note that you only need to '
    "determine the interpretations of each node's 0/1 values, but should "
    "NOT change the given node-entity mappings."
)

def _v3_step2_output_format(
    dag: nx.DiGraph,
    cpts: Optional[Dict[str, Dict[str, Any]]] = None,
    query_type: Optional[str] = None,
) -> str:
    """Build v3 step 2 output schema with concrete node names."""
    if query_type and selector_assumptions_required(query_type):
        if cpts is None:
            raise ValueError("cpts are required for selector-aware output schema")
        return (
            "Return your node-entity interpretations in json format, with "
            "the nodes arranged in topological ordering. For each node, "
            "copy the node entity from above, add the node's own 0/1 "
            "interpretations, and preserve the exogenous selector entries "
            "in the nested shape below:\n"
            "\n"
            "```json\n"
            f"{_selector_schema_lines(dag, cpts, include_node_interpretations=True, copy_node_entity=True)}\n"
            "```"
        )
    topo = topo_sorted_nodes(dag)
    first, last = node_label(topo[0]), node_label(topo[-1])
    return (
        "Return your node-entity interpretations in json format, with the nodes "
        "arranged in topological ordering. Note that for each node, you should "
        "return both (1) the name of its real-world entity (i.e., exactly copied "
        "from what's given above), and (2) the real-world interpretations of its "
        "binary assignment.\n"
        "\n"
        "```json\n"
        "{\n"
        f'  "{first}": {{\n'
        f'    "entity": "{{entity for {first}, copied from above}}",\n'
        f'    "0": "{{interpretation for {first}=0}}",\n'
        f'    "1": "{{interpretation for {first}=1}}"\n'
        "  },\n"
        "  ...\n"
        f'  "{last}": {{\n'
        f'    "entity": "{{entity for {last}, copied from above}}",\n'
        f'    "0": "{{interpretation for {last}=0}}",\n'
        f'    "1": "{{interpretation for {last}=1}}"\n'
        "  }\n"
        "}\n"
        "```"
    )


def build_v3_step2_prompt(
    dag: nx.DiGraph,
    cpts: Dict[str, Dict[str, Any]],
    decimal_places: int,
    entity_mapping_json: str,
    query_type: Optional[str] = None,
) -> str:
    """Build v3 step 2 prompt: entity interpretation with CPTs."""
    node_labels_str = render_node_labels_topo(dag)
    edges_str = render_dag_edges_topo(dag)
    scm_text = render_scm_text(cpts, dag, decimal_places)
    selector_assumption_text = ""
    selector_instruction = ""
    if query_type and selector_assumptions_required(query_type):
        selector_assumption_text = "\n\n" + _selector_assumption_text()
        selector_instruction = (
            "\n\nThe provided entity mapping may already include "
            "`exogenous_selectors`. Preserve those selector entries as part "
            "of the selector-variable assumption stated above, including "
            "their `entity`, `0`, and `1` fields, while adding the graph "
            "node's own binary interpretations."
        )

    return (
        f"Given the following causal graph among variables {node_labels_str}, "
        "where each variable only takes binary values, along with the "
        "probability relationships among them:\n"
        "\n"
        '"""\n'
        f"{edges_str}\n"
        "\n"
        f"{scm_text}"
        f"{selector_assumption_text}\n"
        '"""\n'
        "\n"
        f"{_V3_STEP2_INSTRUCTION_PREFIX}\n"
        "\n"
        f"```json\n{entity_mapping_json}\n```\n"
        "\n"
        f"{_V3_STEP2_INSTRUCTION_BODY}{selector_instruction}\n"
        "\n"
        f"{_v3_step2_output_format(dag, cpts, query_type)}"
    )


# ---------------------------------------------------------------------------
# Question generation prompt (shared v2 step 2 / v3 step 3)
# ---------------------------------------------------------------------------

_QUESTION_GEN_PREAMBLE = (
    "Below is a probabilistic graph-based causal reasoning question in the "
    "symbolic form without any real-world semantics:"
)

_QUESTION_GEN_INSTRUCTION_1 = (
    "Your task is to convert it into a new question under real-world context, "
    "but **not** to solve it yourself. More specifically, you should:\n"
    "\n"
    "1. Map each symbolic node in the causal graph to a real-world entity, "
    "as listed below:"
)

_QUESTION_GEN_INSTRUCTION_2 = (
    "2. The ultimate goal of such conversion is to make it necessary for "
    "test takers to carefully read through the natural language question in "
    "order to understand all the causal and probabilistic relationships among "
    "entities, instead of easily spotting them at first glance. In light of "
    "this, you should articulate the question under the provided context "
    "**in a highly natural manner like a real piece of narrative**. "
    "Specifically, you should NOT explicitly accompany the real-world "
    "entities with their symbolic node notations (X0, X1, etc.). Instead, implicitly "
    "and naturally embed ALL these causal and probabilistic relationships in "
    "a self-contained narrative, WITHOUT using any bullet points."
)

_QUESTION_GEN_INSTRUCTION_3_BASE = (
    "3. Note that the conversion only alters how the causal question is "
    "expressed, but the underlying causal semantics must be preserved "
    "exactly. More specifically, ALL the provided **causal relationships** between "
    "entities and ALL the listed **probability conditions** MUST still occur in the "
    "converted question, so that it still has **the same final numerical "
    "answer as the original symbolic question** after conversion. You may "
    "express each probability value either as a decimal number (e.g., "
    "0.3635) or as a percentage (e.g., 36.35%), whichever reads more "
    "naturally, but you must NOT change the exact numerical values or "
    "their precision. "
)

_QUESTION_GEN_INSTRUCTION_3_PRUNED_ADDENDUM = (
    "Also note that some graph nodes may be involved in causal "
    "relationships but have no corresponding probability conditions "
    "listed in the original question. This is intentional. You should "
    "still describe their causal relationships faithfully, but "
    "NOT invent or hallucinate any new probability conditions beyond "
    "what is explicitly given in the original question."
)

_QUESTION_GEN_INSTRUCTION_3_SUFFIX = (
    "Moreover, "
    "**for the final question sentence**, you should keep the characteristic "
    "words that indicate the specific causal operations, such as \"force\", "
    "\"set\", \"intervene\", etc., so as to preserve the query's causal "
    "semantics."
)

_QUESTION_GEN_INSTRUCTION_3_ASSOC_SUFFIX = (
    "Moreover, **for the final question sentence**, you should express the "
    "observational or statistical semantics clearly, precisely, and naturally. "
    "It is fine to use terms such as conditional probability, joint "
    "probability, marginal probability, or observed difference when they are "
    "the most accurate wording."
)

_QUESTION_GEN_INSTRUCTION_3_COUNTERFACTUAL_SUFFIX = (
    "Moreover, **for the final question sentence**, you should preserve clear "
    "counterfactual semantics. Keep the characteristic causal-operation words "
    "when relevant, such as \"force\", \"set\", \"fix\", or \"intervene\", "
    "and use counterfactual phrasing such as \"would have\", \"would become\", "
    "\"would have been\", or \"had\" when that is needed to match the symbolic "
    "query."
)

_QUESTION_GEN_INSTRUCTION_3_SELECTOR_ADDENDUM = (
    " When the symbolic question states the exogenous selector-variable "
    "assumption, verbalize that assumption naturally in the real-world "
    "context without presenting the selector variables as ordinary observed "
    "graph nodes."
)

_QUESTION_GEN_INSTRUCTION_4 = (
    "4. You should be moderately concise and NOT verbose. Output ONLY the "
    "converted natural-language question text itself. Do NOT include any "
    "preamble, explanation, commentary, quotation marks, or markdown "
    "formatting around the question."
)


def build_question_gen_prompt(
    symbolic_question_text: str,
    entity_interpretation_json: str,
    variant_name: str = "full",
    query_type: Optional[str] = None,
) -> str:
    """Build the NL question generation prompt (v2 step 2 / v3 step 3).

    Parameters
    ----------
    variant_name : str
        Condition variant name (``"full"`` or ``"surgery_ancestor_prune"``).
        For non-full variants, adds an instruction reminding the LLM not to
        hallucinate probability values for nodes whose CPTs were pruned.
    """
    instruction_3 = _QUESTION_GEN_INSTRUCTION_3_BASE
    if variant_name != "full":
        instruction_3 += _QUESTION_GEN_INSTRUCTION_3_PRUNED_ADDENDUM

    if query_type and is_v2_2_query_type(query_type):
        spec = get_query_spec(query_type)
        if spec.is_association:
            suffix = _QUESTION_GEN_INSTRUCTION_3_ASSOC_SUFFIX
        else:
            suffix = _QUESTION_GEN_INSTRUCTION_3_COUNTERFACTUAL_SUFFIX
            if spec.selector_assumptions_required:
                suffix += _QUESTION_GEN_INSTRUCTION_3_SELECTOR_ADDENDUM
        instruction_3 += suffix
    else:
        instruction_3 += _QUESTION_GEN_INSTRUCTION_3_SUFFIX

    return (
        f"{_QUESTION_GEN_PREAMBLE}\n"
        "\n"
        '"""\n'
        f"{symbolic_question_text}\n"
        '"""\n'
        "\n"
        f"{_QUESTION_GEN_INSTRUCTION_1}\n"
        f"```json\n{entity_interpretation_json}\n```\n"
        "\n"
        f"{_QUESTION_GEN_INSTRUCTION_2}\n"
        "\n"
        f"{instruction_3}\n"
        "\n"
        f"{_QUESTION_GEN_INSTRUCTION_4}"
    )
