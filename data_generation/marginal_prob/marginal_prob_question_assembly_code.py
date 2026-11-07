"""marginal_prob_question_assembly_code.py"""

from __future__ import annotations

import networkx as nx

from data_generation.dataset_utils.association_code_assembly_shared import (
    ASSOCIATION_CODE_VARIANT_NAMES as CODE_VARIANT_NAMES,
    AssembledAssociationCodeQuestion as AssembledCodeQuestion,
    assemble_association_code_questions,
)
from data_generation.marginal_prob.marginal_prob_query_sampling import MarginalProbResult, compute_marginal_prob_query
from data_generation.scm_mechanism_sampling import RLCausalSCM, SCMBase


def assemble_all_questions_code(
    scm: SCMBase,
    result: MarginalProbResult,
    *,
    decimal_places: int,
    example_seed: int,
    example_idx: int | None = None,
    is_abnormal: bool | None = None,
    treatment_outcome_depth: float = 0.0,
    effective_prob_conds: int = 0,
    has_naive_conditioning: bool = False,
) -> list[AssembledCodeQuestion]:
    return assemble_association_code_questions(
        scm,
        result,
        query_type="marginal_prob",
        decimal_places=decimal_places,
        example_seed=example_seed,
        example_idx=example_idx,
        is_abnormal=is_abnormal,
        treatment_outcome_depth=treatment_outcome_depth,
        effective_prob_conds=effective_prob_conds,
        has_naive_conditioning=has_naive_conditioning,
    )


if __name__ == "__main__":
    print("Test 1: Marginal Prob code assembly returns association code variants")
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    cpts = {
        0: {"parents": [], "table": {(): 0.3}},
        1: {"parents": [], "table": {(): 0.6}},
        2: {"parents": [0, 1], "table": {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.7, (1, 1): 0.9}},
        3: {"parents": [2], "table": {(0,): 0.2, (1,): 0.8}},
    }
    scm = RLCausalSCM.from_cpts(dag, cpts, decimal_places=1)
    result = compute_marginal_prob_query(scm, target=2)
    questions = assemble_all_questions_code(
        scm,
        result,
        decimal_places=2,
        example_seed=123,
        example_idx=0,
        is_abnormal=False,
    )
    assert [question.variant_name for question in questions] == CODE_VARIANT_NAMES
    assert all(question.answer == "0.3580" for question in questions)
    assert all(question.metadata["domain"] == "code" for question in questions)
    assert all(question.metadata["query_type"] == "marginal_prob" for question in questions)
    assert all("```python" in question.full_question_text for question in questions)
    assert all("\\boxed" in question.full_question_text for question in questions)
    assert any("returns 1" in question.full_question_text for question in questions)
    if "marginal_prob" == "conditional_prob":
        assert all("Optional[int]" in question.full_question_text for question in questions)
        assert all("return None" in question.full_question_text for question in questions)
    print("Marginal Prob code question assembly checks passed.")
