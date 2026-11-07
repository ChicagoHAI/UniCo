"""Query registry for v2.2 NL-domain translation.

The existing NL pipeline was originally written for ATE/CATE/JATE.  This
module centralizes the v2.2 association and counterfactual query facts that
need to be shared by prompt construction, question generation, validation,
and integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class NLQuerySpec:
    query_type: str
    family: str
    source_version: str
    source_triplet_dir: str
    nodes_range: Tuple[int, int]
    variants: Tuple[str, ...]
    result_field: str
    selector_assumptions_required: bool = False

    @property
    def is_association(self) -> bool:
        return self.family == "association"

    @property
    def is_counterfactual(self) -> bool:
        return self.family == "counterfactual"


QUERY_SPECS: Dict[str, NLQuerySpec] = {
    "conditional_prob": NLQuerySpec(
        query_type="conditional_prob",
        family="association",
        source_version="v2.0",
        source_triplet_dir="conditional_prob",
        nodes_range=(3, 10),
        variants=("full", "query_ancestor_prune"),
        result_field="conditional_prob_result",
    ),
    "joint_prob": NLQuerySpec(
        query_type="joint_prob",
        family="association",
        source_version="v2.0",
        source_triplet_dir="joint_prob",
        nodes_range=(3, 10),
        variants=("full", "query_ancestor_prune"),
        result_field="joint_prob_result",
    ),
    "marginal_prob": NLQuerySpec(
        query_type="marginal_prob",
        family="association",
        source_version="v2.0",
        source_triplet_dir="marginal_prob",
        nodes_range=(3, 10),
        variants=("full", "query_ancestor_prune"),
        result_field="marginal_prob_result",
    ),
    "observed_difference": NLQuerySpec(
        query_type="observed_difference",
        family="association",
        source_version="v2.0",
        source_triplet_dir="observed_difference",
        nodes_range=(3, 10),
        variants=("full", "query_ancestor_prune"),
        result_field="observed_difference_result",
    ),
    "att": NLQuerySpec(
        query_type="att",
        family="counterfactual",
        source_version="v2.0",
        source_triplet_dir="att",
        nodes_range=(3, 10),
        variants=("full", "surgery_ancestor_prune"),
        result_field="att_result",
    ),
    "cf": NLQuerySpec(
        query_type="cf",
        family="counterfactual",
        source_version="v2.0",
        source_triplet_dir="cf",
        nodes_range=(4, 10),
        variants=("full", "surgery_ancestor_prune"),
        result_field="cf_result",
        selector_assumptions_required=True,
    ),
    "nie": NLQuerySpec(
        query_type="nie",
        family="counterfactual",
        source_version="v2.1",
        source_triplet_dir="nie_nde",
        nodes_range=(4, 10),
        variants=("full", "surgery_ancestor_prune"),
        result_field="nie_nde_result",
    ),
    "nde": NLQuerySpec(
        query_type="nde",
        family="counterfactual",
        source_version="v2.1",
        source_triplet_dir="nie_nde",
        nodes_range=(4, 10),
        variants=("full", "surgery_ancestor_prune"),
        result_field="nie_nde_result",
    ),
    "pn": NLQuerySpec(
        query_type="pn",
        family="counterfactual",
        source_version="v2.0",
        source_triplet_dir="pn_ps",
        nodes_range=(4, 10),
        variants=("full", "surgery_ancestor_prune"),
        result_field="pn_ps_result",
        selector_assumptions_required=True,
    ),
    "ps": NLQuerySpec(
        query_type="ps",
        family="counterfactual",
        source_version="v2.0",
        source_triplet_dir="pn_ps",
        nodes_range=(4, 10),
        variants=("full", "surgery_ancestor_prune"),
        result_field="pn_ps_result",
        selector_assumptions_required=True,
    ),
}

QUERY_TYPES = tuple(QUERY_SPECS.keys())
ASSOCIATION_QUERY_TYPES = tuple(
    qt for qt, spec in QUERY_SPECS.items() if spec.is_association
)
COUNTERFACTUAL_QUERY_TYPES = tuple(
    qt for qt, spec in QUERY_SPECS.items() if spec.is_counterfactual
)
SELECTOR_ASSUMPTION_QUERY_TYPES = tuple(
    qt for qt, spec in QUERY_SPECS.items()
    if spec.selector_assumptions_required
)


def get_query_spec(query_type: str) -> NLQuerySpec:
    try:
        return QUERY_SPECS[query_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported v2.2 NL query_type: {query_type!r}") from exc


def is_v2_2_query_type(query_type: str | None) -> bool:
    return query_type in QUERY_SPECS


def selector_assumptions_required(query_type: str | None) -> bool:
    return query_type in SELECTOR_ASSUMPTION_QUERY_TYPES


def infer_query_type(triplet: dict, explicit_query_type: str | None = None) -> str:
    if explicit_query_type:
        return explicit_query_type
    if "query_type" in triplet:
        return str(triplet["query_type"])
    if "att_result" in triplet:
        return "att"
    if "cf_result" in triplet:
        return "cf"
    if "nie_nde_result" in triplet:
        # Shared corpora need an explicit query type to distinguish NIE/NDE.
        raise ValueError("Cannot infer nie vs nde from shared triplet fields")
    if "pn_ps_result" in triplet:
        # Shared corpora need an explicit query type to distinguish PN/PS.
        raise ValueError("Cannot infer pn vs ps from shared triplet fields")
    for qt, spec in QUERY_SPECS.items():
        if spec.result_field in triplet:
            return qt
    raise ValueError("Cannot infer query_type from triplet")


def query_type_offset(query_type: str, base: int = 1000) -> int:
    """Stable offset for passage sampling across v2.2 query types."""
    if query_type not in QUERY_SPECS:
        return 0
    return QUERY_TYPES.index(query_type) * base
