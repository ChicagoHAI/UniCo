"""Serialized-triplet adapters for v2.2 NL-domain question generation."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Set

import networkx as nx

from data_generation.att.att_question_assembly import (
    _render_query_text as _render_att_query_text,
)
from data_generation.cf.cf_question_assembly import (
    _render_assumption_text as _render_selector_assumption_text,
    _render_query_text as _render_cf_query_text,
)
from data_generation.conditional_prob.conditional_prob_question_assembly import (
    _render_query_text as _render_conditional_prob_query_text,
)
from data_generation.joint_prob.joint_prob_question_assembly import (
    _render_query_text as _render_joint_prob_query_text,
)
from data_generation.marginal_prob.marginal_prob_question_assembly import (
    _render_query_text as _render_marginal_prob_query_text,
)
from data_generation.nie_nde.nie_nde_question_assembly import (
    _render_query_text as _render_nie_nde_query_text,
)
from data_generation.observed_difference.observed_difference_question_assembly import (
    _render_query_text as _render_observed_difference_query_text,
)
from data_generation.pn_ps.pn_ps_question_assembly import (
    _render_query_text as _render_pn_ps_query_text,
)
from data_generation.dataset_utils.nl_prompt_builder import (
    build_dag,
    render_dag_edges_topo,
    render_node_labels_topo,
    render_scm_text,
)
from data_generation.dataset_utils.nl_query_specs import get_query_spec


def _int_keyed(raw: Dict[Any, Any]) -> Dict[int, Any]:
    return {int(k): v for k, v in raw.items()}


def _sorted_int_keys(raw: Dict[Any, Any]) -> List[int]:
    return sorted(int(k) for k in raw.keys())


def _derive_graph_query_variants(example_seed: int) -> tuple[int, int]:
    rng = random.Random(example_seed)
    return rng.randint(1, 3), rng.randint(1, 3)


def _remove_incoming_edges(dag: nx.DiGraph, nodes: Iterable[int]) -> nx.DiGraph:
    mutilated = dag.copy()
    for node in nodes:
        for parent in list(mutilated.predecessors(node)):
            mutilated.remove_edge(parent, node)
    return mutilated


def _topo_subset(dag: nx.DiGraph, nodes: Set[int]) -> List[int]:
    topo = list(nx.topological_sort(dag))
    return [node for node in topo if node in nodes]


def _ancestor_closure(dag: nx.DiGraph, nodes: Set[int]) -> Set[int]:
    result = set(nodes)
    for node in list(nodes):
        result.update(nx.ancestors(dag, node))
    return result


def get_condition_nodes(
    triplet: Dict[str, Any],
    query_type: str,
    variant_name: str,
) -> List[int] | None:
    """Return CPT nodes rendered for the query/variant.

    ``None`` means all nodes, matching the post-validator convention.
    """
    if variant_name == "full":
        return None

    spec = get_query_spec(query_type)
    if variant_name not in spec.variants:
        raise ValueError(
            f"Unsupported variant {variant_name!r} for {query_type!r}"
        )

    dag = build_dag(triplet["dag_edges"], triplet["num_nodes"])

    if spec.is_association:
        if query_type == "conditional_prob":
            result = triplet[spec.result_field]
            core_nodes = {int(result["target"])} | set(_sorted_int_keys(result["evidence"]))
        elif query_type == "joint_prob":
            core_nodes = {int(n) for n in triplet[spec.result_field]["target_nodes"]}
        elif query_type == "marginal_prob":
            core_nodes = {int(triplet[spec.result_field]["target"])}
        elif query_type == "observed_difference":
            result = triplet[spec.result_field]
            core_nodes = {int(result["source"]), int(result["target"])}
        else:
            raise ValueError(f"Unsupported association query_type: {query_type}")
        return _topo_subset(dag, _ancestor_closure(dag, core_nodes))

    if query_type == "att":
        treatment = int(triplet["treatment"])
        outcome = int(triplet["outcome"])
        factual_nodes = {treatment} | nx.ancestors(dag, treatment)
        mutilated = _remove_incoming_edges(dag, [treatment])
        counterfactual_nodes = {outcome} | nx.ancestors(mutilated, outcome)
        return _topo_subset(dag, factual_nodes | counterfactual_nodes)

    if query_type == "cf":
        treatments = _int_keyed(triplet["treatments"])
        outcome = int(triplet["outcome"])
        evidence = _int_keyed(triplet.get("evidence", {}))
        return _get_cf_prune_nodes(dag, treatments, outcome, evidence)

    if query_type in ("nie", "nde"):
        treatment = int(triplet["treatment"])
        mediator = int(triplet["mediator"])
        outcome = int(triplet["outcome"])
        mutilated = _remove_incoming_edges(dag, [treatment])
        cpd_nodes = _ancestor_closure(mutilated, {mediator, outcome}) - {treatment}
        return _topo_subset(dag, cpd_nodes)

    if query_type in ("pn", "ps"):
        treatment = int(triplet["treatment"])
        outcome = int(triplet["outcome"])
        treatments = {treatment: 1}
        evidence = {treatment: 1, outcome: 1}
        return _get_cf_prune_nodes(dag, treatments, outcome, evidence)

    raise ValueError(f"Unsupported counterfactual query_type: {query_type}")


def _get_cf_prune_nodes(
    dag: nx.DiGraph,
    treatments: Dict[int, int],
    outcome: int,
    evidence: Dict[int, int],
) -> List[int]:
    if evidence:
        factual_nodes = set(evidence.keys())
        for evidence_node in evidence:
            factual_nodes.update(nx.ancestors(dag, evidence_node))
    else:
        factual_nodes = {outcome}

    mutilated = _remove_incoming_edges(dag, treatments.keys())
    counterfactual_nodes = {outcome} | nx.ancestors(mutilated, outcome)
    return _topo_subset(dag, factual_nodes | counterfactual_nodes)


def render_query_text_from_triplet(
    triplet: Dict[str, Any],
    query_type: str,
    query_variant: int,
) -> str:
    spec = get_query_spec(query_type)
    result = triplet[spec.result_field]

    if query_type == "conditional_prob":
        ns = SimpleNamespace(
            target=int(result["target"]),
            evidence=_int_keyed(result["evidence"]),
        )
        return _render_conditional_prob_query_text(ns, query_variant=query_variant)
    if query_type == "joint_prob":
        ns = SimpleNamespace(target_nodes=[int(n) for n in result["target_nodes"]])
        return _render_joint_prob_query_text(ns, query_variant=query_variant)
    if query_type == "marginal_prob":
        ns = SimpleNamespace(target=int(result["target"]))
        return _render_marginal_prob_query_text(ns, query_variant=query_variant)
    if query_type == "observed_difference":
        ns = SimpleNamespace(source=int(result["source"]), target=int(result["target"]))
        return _render_observed_difference_query_text(ns, query_variant=query_variant)
    if query_type == "att":
        return _render_att_query_text(
            int(triplet["treatment"]),
            int(triplet["outcome"]),
            query_variant=query_variant,
        )
    if query_type == "cf":
        return _render_cf_query_text(
            _int_keyed(triplet["treatments"]),
            int(triplet["outcome"]),
            _int_keyed(triplet.get("evidence", {})),
            query_variant=query_variant,
        )
    if query_type in ("nie", "nde"):
        return _render_nie_nde_query_text(
            int(triplet["treatment"]),
            int(triplet["mediator"]),
            int(triplet["outcome"]),
            effect_type=query_type,
            query_variant=query_variant,
        )
    if query_type in ("pn", "ps"):
        return _render_pn_ps_query_text(
            int(triplet["treatment"]),
            int(triplet["outcome"]),
            effect_type=query_type,
            query_variant=query_variant,
        )
    raise ValueError(f"Unsupported query_type: {query_type}")


def build_symbolic_question_text_from_triplet(
    triplet: Dict[str, Any],
    query_type: str,
    variant_name: str,
) -> str:
    dag = build_dag(triplet["dag_edges"], triplet["num_nodes"])
    graph_variant, query_variant = _derive_graph_query_variants(triplet["example_seed"])
    condition_nodes = get_condition_nodes(triplet, query_type, variant_name)
    query_text = render_query_text_from_triplet(triplet, query_type, query_variant)
    conditions_text = render_scm_text(
        triplet["cpts"],
        dag,
        triplet["decimal_places"],
        nodes=condition_nodes,
    )

    text = (
        "Imagine a self-contained, hypothetical world with only the "
        "following conditions, and without any unmentioned factors or "
        "causal relationships:\n"
        f"Given the following causal graph among variables {render_node_labels_topo(dag)}:\n"
        f"{render_dag_edges_topo(dag)}\n"
        "\n"
        "The following probability relationships hold:\n"
        f"{conditions_text}\n"
    )

    if get_query_spec(query_type).selector_assumptions_required:
        text += "\n" + _render_selector_assumption_text(graph_variant) + "\n"

    return text + "\n" + query_text


def extract_answer(triplet: Dict[str, Any], query_type: str) -> Any:
    spec = get_query_spec(query_type)
    result = triplet[spec.result_field]
    if query_type in ("conditional_prob", "joint_prob", "marginal_prob"):
        return f"{float(result['probability']):.4f}"
    if query_type == "observed_difference":
        return f"{float(result['observed_difference']):.4f}"
    if query_type == "att":
        return result["att_value"]
    if query_type == "cf":
        values = result["factor_values"]
        return values.get("(1,)", values.get("1", result.get("cf_answer_probability")))
    if query_type == "nie":
        return result["nie_value"]
    if query_type == "nde":
        return result["nde_value"]
    if query_type == "pn":
        return result["pn_value"]
    if query_type == "ps":
        return result["ps_value"]
    raise ValueError(f"Unsupported query_type: {query_type}")


def build_metadata(
    triplet: Dict[str, Any],
    query_type: str,
    variant_name: str,
    *,
    pipeline_version: str,
    entity_interp: Dict[str, Any],
    question_model: str,
    question_provider: str,
    question_retry_count: int,
) -> Dict[str, Any]:
    spec = get_query_spec(query_type)
    graph_variant, query_variant = _derive_graph_query_variants(triplet["example_seed"])
    condition_nodes = get_condition_nodes(triplet, query_type, variant_name)
    nodes_kept = (
        list(range(triplet["num_nodes"])) if condition_nodes is None
        else sorted(condition_nodes)
    )
    nodes_pruned = sorted(set(range(triplet["num_nodes"])) - set(nodes_kept))
    result = triplet[spec.result_field]

    metadata: Dict[str, Any] = {
        "num_nodes": triplet["num_nodes"],
        "num_edges": triplet["num_edges"],
        "dag_edges": triplet["dag_edges"],
        "query_type": query_type,
        "domain": "nl",
        "variant_name": variant_name,
        "condition_variant": variant_name,
        "pipeline_version": pipeline_version,
        "graph_variant": graph_variant,
        "query_variant": query_variant,
        "nodes_kept": nodes_kept,
        "nodes_pruned": nodes_pruned,
        "passage_idx": entity_interp["passage_idx"],
        "passage_source": entity_interp["passage_source"],
        "entity_mapping": entity_interp["entity_interpretation"],
        "is_abnormal": bool(triplet.get("is_abnormal", result.get("is_abnormal", False))),
        "verification_ok": result.get("verification_ok", True),
        "treatment_outcome_depth": triplet.get("treatment_outcome_depth", 0),
        "effective_prob_conds": triplet.get("effective_prob_conds", 0),
        "has_naive_conditioning": triplet.get("has_naive_conditioning", False),
        "decimal_places": triplet["decimal_places"],
        "model_entity": entity_interp["model"],
        "provider_entity": entity_interp["provider"],
        "model_question": question_model,
        "provider_question": question_provider,
        "entity_retry_count": entity_interp["retry_count"],
        "question_retry_count": question_retry_count,
    }

    if query_type == "conditional_prob":
        metadata.update({
            "target": int(result["target"]),
            "evidence": result["evidence"],
            "probability": result["probability"],
        })
    elif query_type == "joint_prob":
        metadata.update({
            "target_nodes": result["target_nodes"],
            "probability": result["probability"],
        })
    elif query_type == "marginal_prob":
        metadata.update({
            "target": int(result["target"]),
            "probability": result["probability"],
        })
    elif query_type == "observed_difference":
        metadata.update({
            "source": int(result["source"]),
            "target": int(result["target"]),
            "observed_difference": result["observed_difference"],
        })
    elif query_type == "att":
        metadata.update({
            "treatment": int(triplet["treatment"]),
            "outcome": int(triplet["outcome"]),
            "p_y1_do1_given_treated": result["p_y1_do1_given_treated"],
            "p_y1_do0_given_treated": result["p_y1_do0_given_treated"],
            "att_value": result["att_value"],
            "cf_result_do1": result["cf_result_do1"],
            "cf_result_do0": result["cf_result_do0"],
        })
    elif query_type == "cf":
        metadata.update({
            "treatments": triplet["treatments"],
            "outcome": int(triplet["outcome"]),
            "evidence": triplet.get("evidence", {}),
            "cf_result": result,
            "cf_answer_assignment": 1,
            "cf_answer_probability": extract_answer(triplet, query_type),
        })
    elif query_type in ("nie", "nde"):
        metadata.update({
            "treatment": int(triplet["treatment"]),
            "mediator": int(triplet["mediator"]),
            "outcome": int(triplet["outcome"]),
            "effect_type": query_type,
            "nie_value": result["nie_value"],
            "nde_value": result["nde_value"],
            "pm_do0": result["pm_do0"],
            "pm_do1": result["pm_do1"],
            "py_values": result["py_values"],
        })
    elif query_type in ("pn", "ps"):
        evidence = (
            {str(triplet["treatment"]): 1, str(triplet["outcome"]): 1}
            if query_type == "pn"
            else {str(triplet["treatment"]): 0, str(triplet["outcome"]): 0}
        )
        metadata.update({
            "treatment": int(triplet["treatment"]),
            "outcome": int(triplet["outcome"]),
            "effect_type": query_type,
            "pn_value": result["pn_value"],
            "ps_value": result["ps_value"],
            "p_y0_eq0_given_x1_y1": result["p_y0_eq0_given_x1_y1"],
            "p_y1_eq1_given_x0_y0": result["p_y1_eq1_given_x0_y0"],
            "cf_result_pn": result["cf_result_pn"],
            "cf_result_ps": result["cf_result_ps"],
            "conditioning_event": evidence,
        })
    else:
        raise ValueError(f"Unsupported query_type: {query_type}")

    return metadata
