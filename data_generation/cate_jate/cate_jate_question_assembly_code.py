"""
cate_jate_question_assembly_code.py

CATE/JATE Code-Domain Question Assembly: generates Python code representations
of causal structures for CATE and JATE queries, using the shared code renderer.

Given an RLCausalSCM, query specification (treatments, outcome, evidence),
and a pre-computed CATEJATEResult, assembles two code-domain question variants:
  1) full-code: Single-Function, all nodes with full CPDs
  2) surgery_ancestor_prune-code: Multi-Function with stubs for pruned nodes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from data_generation.scm_mechanism_sampling import SCMBase
from data_generation.cate_jate.cate_jate_query_sampling import CATEJATEResult
from data_generation.dataset_utils.code_domain_renderer import (
    sample_axes,
    render_code,
    render_qv_text,
    compose_question_text,
    get_entry_function_name,
)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class AssembledQuestion:
    """A complete assembled code-domain question for CATE/JATE."""
    treatments: List[int]
    outcome: int
    evidence: Dict[int, int]
    query_type: str
    variant_name: str
    effect_ground_truth: float
    is_abnormal: bool
    code_structure: str         # "single" or "multi"
    cf_variant: str             # "A", "B", or "C"
    si_variant: str             # "A", "B", or "C"
    vn_variant: str             # "A" or "B"
    qv_variant: int             # 1, 2, or 3
    full_question_text: str
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


# ============================================================
# Query Type Determination
# ============================================================

def _determine_query_type(
    treatments: List[int],
    evidence: Dict[int, int],
) -> str:
    """Determine query type from treatments and evidence."""
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
    return "cate"


# ============================================================
# Condition Variant Computation (code domain)
# ============================================================

CODE_VARIANT_NAMES = ["full", "surgery_ancestor_prune"]


def _compute_full_nodes(scm: SCMBase) -> tuple:
    """Full variant: all nodes kept, none pruned."""
    all_nodes = sorted(scm.nodes)
    return all_nodes, []


def _compute_surgery_ancestor_prune_nodes(
    scm: SCMBase,
    treatments: List[int],
    outcome: int,
    evidence: Dict[int, int],
) -> tuple:
    """Surgery + ancestor prune for CATE/JATE.

    After do(treatments), keep ancestors of outcome AND evidence nodes
    in the mutilated graph (plus outcome and evidence themselves).
    Treatment(s) are excluded from kept nodes.

    Returns (nodes_kept, nodes_pruned).
    """
    mutilated = scm.pgmpy_model.do(treatments)
    relevant_nodes = [outcome] + sorted(evidence.keys())
    anc_graph = mutilated.get_ancestral_graph(nodes=relevant_nodes)
    anc_nodes = set(anc_graph.nodes())

    cpd_nodes = anc_nodes - set(treatments)
    all_nodes = set(scm.nodes)

    return sorted(cpd_nodes), sorted(all_nodes - cpd_nodes)


# ============================================================
# Question Assembly
# ============================================================

def assemble_all_questions_code(
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
    """Generate two code-domain question variants for a CATE/JATE example.

    Samples (cf, si, vn, qv) axes deterministically from example_seed.
    Returns two AssembledQuestion objects:
      - full-code (single-function)
      - surgery_ancestor_prune-code (multi-function)
    """
    cf, si, vn, qv = sample_axes(example_seed)
    dag = scm.dag
    query_type = _determine_query_type(treatments, evidence)

    # Compute nodes for each variant
    full_kept, full_pruned = _compute_full_nodes(scm)
    sap_kept, sap_pruned = _compute_surgery_ancestor_prune_nodes(
        scm, treatments, outcome, evidence
    )

    results = []

    for variant_name, structure, nodes_kept, nodes_pruned in [
        ("full", "single", full_kept, full_pruned),
        ("surgery_ancestor_prune", "multi", sap_kept, sap_pruned),
    ]:
        code = render_code(
            dag=dag,
            cpts=scm.cpts,
            treatments=treatments,
            outcome=outcome,
            evidence=evidence if evidence else None,
            query_type=query_type,
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            structure=structure,
            cf_variant=cf,
            si_variant=si,
            vn_variant=vn,
            decimal_places=decimal_places,
            example_seed=example_seed,
        )

        func_name = get_entry_function_name(outcome, vn)
        qv_text = render_qv_text(query_type, qv, func_name, num_treatments=len(treatments))
        full_question_text = compose_question_text(code, qv_text)

        metadata = {
            "num_nodes": len(scm.nodes),
            "num_edges": dag.number_of_edges(),
            "dag_edges": sorted(list(dag.edges())),
            "treatments": sorted(treatments),
            "outcome": outcome,
            "evidence": {str(k): v for k, v in sorted(evidence.items())} if evidence else {},
            "query_type": query_type,
            "domain": "code",
            "variant_name": variant_name,
            "code_structure": structure,
            "cf_variant": cf,
            "si_variant": si,
            "vn_variant": vn,
            "qv_variant": qv,
            "is_abnormal": cate_jate_result.is_abnormal,
            "effect_surgery": cate_jate_result.effect_surgery,
            "effect_causal_query": cate_jate_result.effect_causal_query,
            "verification_ok": cate_jate_result.verification_ok,
            "nodes_kept": nodes_kept,
            "nodes_pruned": nodes_pruned,
            "treatment_outcome_depth": treatment_outcome_depth,
            "effective_prob_conds": effective_prob_conds,
            "has_naive_conditioning": has_naive_conditioning,
        }

        results.append(AssembledQuestion(
            treatments=sorted(treatments),
            outcome=outcome,
            evidence=evidence,
            query_type=query_type,
            variant_name=variant_name,
            effect_ground_truth=cate_jate_result.effect_surgery,
            is_abnormal=cate_jate_result.is_abnormal,
            code_structure=structure,
            cf_variant=cf,
            si_variant=si,
            vn_variant=vn,
            qv_variant=qv,
            full_question_text=full_question_text,
            nodes_kept=nodes_kept,
            nodes_pruned=nodes_pruned,
            metadata=metadata,
        ))

    return results
