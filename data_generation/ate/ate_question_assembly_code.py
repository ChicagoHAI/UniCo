"""
ate_question_assembly_code.py

ATE Code-Domain Question Assembly: generates Python code representations
of causal structures for ATE queries, using the shared code renderer.

Given an RLCausalSCM and a pre-computed ATEResult (from ate_query_sampling),
assembles two code-domain question variants:
  1) full-code: Single-Function, all nodes with full CPDs
  2) surgery_ancestor_prune-code: Multi-Function, main function calls all
     helpers; kept node helpers have full CPDs; pruned node helpers are stubs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from data_generation.scm_mechanism_sampling import SCMBase
from data_generation.ate.ate_query_sampling import ATEResult
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
    """A complete assembled code-domain question for one (treatment, outcome, variant) triple."""
    treatment: int
    outcome: int
    variant_name: str
    ate_ground_truth: float
    is_abnormal: bool
    code_structure: str         # "single" or "multi"
    cf_variant: str             # "A", "B", or "C"
    si_variant: str             # "A", "B", or "C"
    vn_variant: str             # "A" or "B"
    qv_variant: int             # 1, 2, or 3
    full_question_text: str     # preamble + code + QV + answer format
    nodes_kept: List[int]
    nodes_pruned: List[int]
    metadata: Dict


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
    treatment: int,
    outcome: int,
) -> tuple:
    """Surgery + ancestor prune: ancestors of outcome in mutilated graph.

    Returns (nodes_kept, nodes_pruned) where:
    - nodes_kept = ancestors of outcome in mutilated graph minus treatment
    - nodes_pruned = all other nodes (including treatment)
    """
    mutilated = scm.pgmpy_model.do([treatment])
    anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])
    anc_nodes = set(anc_graph.nodes())

    cpd_nodes = anc_nodes - {treatment}
    all_nodes = set(scm.nodes)

    return sorted(cpd_nodes), sorted(all_nodes - cpd_nodes)


# ============================================================
# Question Assembly
# ============================================================

def assemble_all_questions_code(
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
    """Generate two code-domain question variants for a single ATE example.

    Samples (cf, si, vn, qv) axes deterministically from example_seed.
    Returns two AssembledQuestion objects:
      - full-code (single-function)
      - surgery_ancestor_prune-code (multi-function)
    """
    cf, si, vn, qv = sample_axes(example_seed)
    dag = scm.dag

    # Compute nodes for each variant
    full_kept, full_pruned = _compute_full_nodes(scm)
    sap_kept, sap_pruned = _compute_surgery_ancestor_prune_nodes(
        scm, treatment, outcome
    )

    results = []

    for variant_name, structure, nodes_kept, nodes_pruned in [
        ("full", "single", full_kept, full_pruned),
        ("surgery_ancestor_prune", "multi", sap_kept, sap_pruned),
    ]:
        code = render_code(
            dag=dag,
            cpts=scm.cpts,
            treatments=[treatment],
            outcome=outcome,
            evidence=None,
            query_type="ate",
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
        qv_text = render_qv_text("ate", qv, func_name)
        full_question_text = compose_question_text(code, qv_text)

        metadata = {
            "num_nodes": len(scm.nodes),
            "num_edges": dag.number_of_edges(),
            "dag_edges": sorted(list(dag.edges())),
            "treatment": treatment,
            "outcome": outcome,
            "query_type": "ate",
            "domain": "code",
            "variant_name": variant_name,
            "code_structure": structure,
            "cf_variant": cf,
            "si_variant": si,
            "vn_variant": vn,
            "qv_variant": qv,
            "is_abnormal": ate_result.is_abnormal,
            "ate_surgery": ate_result.ate_surgery,
            "ate_causal_query": ate_result.ate_causal_query,
            "verification_ok": ate_result.verification_ok,
            "nodes_kept": nodes_kept,
            "nodes_pruned": nodes_pruned,
            "treatment_outcome_depth": treatment_outcome_depth,
            "effective_prob_conds": effective_prob_conds,
            "has_naive_conditioning": has_naive_conditioning,
        }

        results.append(AssembledQuestion(
            treatment=treatment,
            outcome=outcome,
            variant_name=variant_name,
            ate_ground_truth=ate_result.ate_surgery,
            is_abnormal=ate_result.is_abnormal,
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
