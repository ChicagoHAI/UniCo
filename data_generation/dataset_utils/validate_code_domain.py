"""
validate_code_domain.py

Validation script for code-domain rendering. Executes generated Python code
against controlled RNG streams and verifies that Monte Carlo estimates
match ground-truth causal effects computed by the pipeline.

Covers ATE, CATE, and JATE across all CF variants, plus VN-B identifier
correctness and input validation.
"""

import ast
import random

import networkx as nx

from data_generation.scm_mechanism_sampling import RLCausalSCM
from data_generation.ate.ate_query_sampling import compute_ate
from data_generation.cate_jate.cate_jate_query_sampling import compute_cate_jate
from data_generation.dataset_utils.code_domain_renderer import render_code


def _execute_code_and_get_function(code: str, func_name: str):
    """Execute rendered code and return the entry-point function."""
    namespace = {}
    exec(code, namespace)
    return namespace[func_name]


def _monte_carlo_ate(func, num_samples: int = 100000, seed: int = 0) -> float:
    """Estimate ATE via Monte Carlo: E[Y|do(T=1)] - E[Y|do(T=0)]."""
    rng = random.Random(seed)
    sum_1 = 0
    sum_0 = 0
    for _ in range(num_samples):
        sum_1 += func(1, rng)
        sum_0 += func(0, rng)
    return sum_1 / num_samples - sum_0 / num_samples


def _monte_carlo_cate(func, num_samples: int = 200000, seed: int = 0) -> float:
    """Estimate CATE via Monte Carlo, excluding None returns."""
    rng = random.Random(seed)
    sum_1 = 0
    count_1 = 0
    sum_0 = 0
    count_0 = 0
    for _ in range(num_samples):
        r1 = func(1, rng)
        if r1 is not None:
            sum_1 += r1
            count_1 += 1
        r0 = func(0, rng)
        if r0 is not None:
            sum_0 += r0
            count_0 += 1
    if count_1 == 0 or count_0 == 0:
        return 0.0
    return sum_1 / count_1 - sum_0 / count_0


def _monte_carlo_jate(func, num_samples: int = 100000, seed: int = 0) -> float:
    """Estimate JATE: E[Y|do(T1=1,T2=1)] - E[Y|do(T1=0,T2=0)]."""
    rng = random.Random(seed)
    sum_11 = 0
    sum_00 = 0
    for _ in range(num_samples):
        sum_11 += func(1, 1, rng)
        sum_00 += func(0, 0, rng)
    return sum_11 / num_samples - sum_00 / num_samples


def validate_ate(num_nodes: int = 5, seed: int = 42, atol: float = 0.02):
    """Validate ATE rendering across all CF variants."""
    dag = nx.DiGraph()
    dag.add_edges_from([(3, 0), (0, 1), (1, 2), (0, 2)])
    if num_nodes >= 5:
        dag.add_edges_from([(2, 4)])
    scm = RLCausalSCM(dag, seed=seed, decimal_places=4)

    treatment, outcome = 0, 2
    ate_result = compute_ate(scm, treatment, outcome)
    ground_truth = ate_result.ate_surgery

    print(f"  ATE ground truth: {ground_truth:.6f}")

    for cf in ["A", "B", "C"]:
        for structure in ["single", "multi"]:
            nodes_kept = sorted(scm.nodes)
            nodes_pruned = []
            if structure == "multi":
                # Simulate surgery_ancestor_prune
                mutilated = scm.pgmpy_model.do([treatment])
                anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])
                anc_nodes = set(anc_graph.nodes())
                cpd_nodes = anc_nodes - {treatment}
                nodes_kept = sorted(cpd_nodes)
                nodes_pruned = sorted(set(scm.nodes) - cpd_nodes)

            code = render_code(
                dag, scm.cpts, [treatment], outcome, None, "ate",
                nodes_kept, nodes_pruned, structure,
                cf, "A", "A", 4, seed,
            )

            func_name = f"x{outcome}"
            func = _execute_code_and_get_function(code, func_name)
            mc_estimate = _monte_carlo_ate(func, seed=seed)

            diff = abs(mc_estimate - ground_truth)
            status = "PASS" if diff < atol else "FAIL"
            print(f"    CF-{cf} {structure:6s}: MC={mc_estimate:.6f}  diff={diff:.6f}  [{status}]")
            if status == "FAIL":
                raise AssertionError(
                    f"ATE validation failed for CF-{cf} {structure}: "
                    f"MC={mc_estimate:.6f} vs GT={ground_truth:.6f}"
                )


def validate_jate(seed: int = 42, atol: float = 0.02):
    """Validate JATE rendering."""
    dag = nx.DiGraph()
    dag.add_edges_from([(3, 0), (3, 1), (0, 1), (0, 2), (1, 2)])
    scm = RLCausalSCM(dag, seed=seed, decimal_places=4)

    treatments = [0, 3]
    outcome = 2

    result = compute_cate_jate(scm, treatments, outcome, {})
    ground_truth = result.effect_surgery

    print(f"  JATE ground truth: {ground_truth:.6f}")

    for cf in ["A", "B", "C"]:
        code = render_code(
            dag, scm.cpts, treatments, outcome, None, "jate",
            sorted(scm.nodes), [], "single",
            cf, "A", "A", 4, seed,
        )
        func_name = f"x{outcome}"
        func = _execute_code_and_get_function(code, func_name)
        mc_estimate = _monte_carlo_jate(func, seed=seed)

        diff = abs(mc_estimate - ground_truth)
        status = "PASS" if diff < atol else "FAIL"
        print(f"    CF-{cf} single: MC={mc_estimate:.6f}  diff={diff:.6f}  [{status}]")
        if status == "FAIL":
            raise AssertionError(
                f"JATE validation failed for CF-{cf}: "
                f"MC={mc_estimate:.6f} vs GT={ground_truth:.6f}"
            )


def validate_cate(seed: int = 42, atol: float = 0.03):
    """Validate CATE rendering across all CF variants."""
    dag = nx.DiGraph()
    dag.add_edges_from([(3, 0), (3, 1), (0, 1), (0, 2), (1, 4)])
    scm = RLCausalSCM(dag, seed=seed, decimal_places=4)

    treatment = 0
    outcome = 1
    evidence = {4: 1}

    result = compute_cate_jate(scm, [treatment], outcome, evidence)
    ground_truth = result.effect_surgery

    print(f"  CATE ground truth: {ground_truth:.6f}")

    for cf in ["A", "B", "C"]:
        code = render_code(
            dag, scm.cpts, [treatment], outcome, evidence, "cate",
            sorted(scm.nodes), [], "single",
            cf, "A", "A", 4, seed,
        )
        func_name = f"x{outcome}"
        func = _execute_code_and_get_function(code, func_name)
        mc_estimate = _monte_carlo_cate(func, seed=seed)

        diff = abs(mc_estimate - ground_truth)
        status = "PASS" if diff < atol else "FAIL"
        print(f"    CF-{cf} single: MC={mc_estimate:.6f}  diff={diff:.6f}  [{status}]")
        if status == "FAIL":
            raise AssertionError(
                f"CATE validation failed for CF-{cf}: "
                f"MC={mc_estimate:.6f} vs GT={ground_truth:.6f}"
            )

    # Verify Optional[int] return type and None filtering
    code_check = render_code(
        dag, scm.cpts, [treatment], outcome, evidence, "cate",
        sorted(scm.nodes), [], "single",
        "A", "A", "A", 4, seed,
    )
    assert "Optional[int]" in code_check, "CATE code missing Optional[int]"
    assert "return None" in code_check, "CATE code missing return None"
    assert "from typing import Optional" in code_check, "CATE code missing Optional import"
    print("    Optional/None checks: PASS")


def validate_vn_b_identifiers():
    """Verify all VN-B code uses only var_*, func_*, arg_*, rng identifiers."""
    dag = nx.DiGraph()
    dag.add_edges_from([(0, 1), (1, 2)])
    scm = RLCausalSCM(dag, seed=42, decimal_places=4)

    allowed_builtins = {"random", "Random", "ValueError", "int", "Optional"}

    for cf in ["A", "B", "C"]:
        for structure in ["single", "multi"]:
            code = render_code(
                dag, scm.cpts, [0], 2, None, "ate",
                sorted(scm.nodes), [], structure,
                cf, "A", "B", 4, 42,
            )
            tree = ast.parse(code)
            bad_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    name = node.id
                    if name in allowed_builtins or name == "rng":
                        continue
                    if not (name.startswith("var_") or name.startswith("func_") or name.startswith("arg_")):
                        bad_names.add(name)
                elif isinstance(node, ast.FunctionDef):
                    name = node.name
                    if not (name.startswith("func_") or name.startswith("var_")):
                        bad_names.add(f"def:{name}")
            if bad_names:
                raise AssertionError(
                    f"VN-B leaked non-generic identifiers in CF-{cf} {structure}: {bad_names}"
                )
    print("  VN-B identifier check: PASS (all CF x structure combos clean)")


def validate_jate_multi_treatment():
    """Regression test: JATE with >2 treatments through the assembly path."""
    from data_generation.cate_jate.cate_jate_question_assembly_code import (
        assemble_all_questions_code,
    )
    from data_generation.dataset_utils.difficulty_metrics import (
        compute_treatment_outcome_depth,
        compute_effective_prob_conds,
        has_naive_conditioning,
    )

    dag = nx.DiGraph()
    dag.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
    scm = RLCausalSCM(dag, seed=42, decimal_places=4)

    treatments = [0, 1, 2]
    outcome = 4
    evidence = {}

    # Construct CATEJATEResult manually since compute_cate_jate rejects >2 treatments.
    # Compute ground-truth JATE via direct graph surgery.
    from pgmpy.inference import VariableElimination
    from data_generation.cate_jate.cate_jate_query_sampling import CATEJATEResult

    def _p_of_one(factor, var):
        vals = factor.values
        state_names = factor.state_names[var]
        idx_1 = state_names.index(1)
        return float(vals[idx_1])

    model_do_1 = scm.pgmpy_model.do(treatments)
    ve_1 = VariableElimination(model_do_1)
    ev_1 = {t: 1 for t in treatments}
    p_y_do1 = _p_of_one(ve_1.query([outcome], evidence=ev_1), outcome)

    model_do_0 = scm.pgmpy_model.do(treatments)
    ve_0 = VariableElimination(model_do_0)
    ev_0 = {t: 0 for t in treatments}
    p_y_do0 = _p_of_one(ve_0.query([outcome], evidence=ev_0), outcome)

    effect = p_y_do1 - p_y_do0
    result = CATEJATEResult(
        treatments=treatments,
        outcome=outcome,
        evidence=evidence,
        query_type="jate",
        effect_surgery=effect,
        effect_causal_query=None,
        is_abnormal=False,
        verification_ok=None,
    )

    tod = compute_treatment_outcome_depth(dag, treatments, outcome)
    epc = compute_effective_prob_conds(dag, treatments, outcome, evidence=evidence)
    ns = has_naive_conditioning(dag, treatments, outcome, evidence_nodes=[])

    questions = assemble_all_questions_code(
        scm, treatments, outcome, evidence, result,
        decimal_places=4,
        example_seed=42,
        treatment_outcome_depth=tod,
        effective_prob_conds=epc,
        has_naive_conditioning=ns,
    )

    assert len(questions) == 2, f"Expected 2 questions, got {len(questions)}"

    for q in questions:
        text = q.full_question_text

        # Extract code block from the assembled question text
        code_start = text.find("```python\n") + len("```python\n")
        code_end = text.find("\n```", code_start)
        code_block = text[code_start:code_end]

        # Parse and check function signature has 3 treatment inputs
        tree = ast.parse(code_block)
        func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        entry_func = func_defs[0]
        param_names = [a.arg for a in entry_func.args.args]
        treatment_params = [p for p in param_names if p != "rng"]
        assert len(treatment_params) == 3, (
            f"Expected 3 treatment params in {q.variant_name}, "
            f"got {len(treatment_params)}: {treatment_params}"
        )

        # Check 3 input validation blocks in code
        assert code_block.count("not in (0, 1)") == 3, (
            f"Expected 3 input validations in {q.variant_name}"
        )

        # Check assembled question text wording matches 3-treatment arity
        assert "two inputs" not in text, (
            f"Assembled {q.variant_name} text still says 'two inputs'"
        )
        assert "both inputs" not in text, (
            f"Assembled {q.variant_name} text still says 'both inputs'"
        )

    # Specifically check QV-3 wording by generating with a fixed seed that
    # produces qv=3, or by checking across multiple seeds
    found_qv3 = False
    for seed in range(100):
        questions = assemble_all_questions_code(
            scm, treatments, outcome, evidence, result,
            decimal_places=4, example_seed=seed,
            treatment_outcome_depth=tod, effective_prob_conds=epc,
            has_naive_conditioning=ns,
        )
        q = questions[0]
        if q.qv_variant == 3:
            assert "f(a, b, c)" in q.full_question_text, "QV-3 missing f(a, b, c)"
            assert "f(1, 1, 1)" in q.full_question_text, "QV-3 missing f(1, 1, 1)"
            assert "f(0, 0, 0)" in q.full_question_text, "QV-3 missing f(0, 0, 0)"
            found_qv3 = True
            break
    assert found_qv3, "Could not find a seed producing QV-3 in 100 tries"

    print("  JATE >2 treatments (assembly path): PASS (3 params, QV arity, QV-3 f(a,b,c))")


def validate_cate_none_filter_ordering():
    """Verify that the None-filter block appears after all sampling assignments."""
    dag = nx.DiGraph()
    dag.add_edges_from([(3, 0), (3, 1), (0, 1), (0, 2), (1, 4)])
    scm = RLCausalSCM(dag, seed=42, decimal_places=4)

    treatment = 0
    outcome = 1
    evidence = {4: 1}

    code = render_code(
        dag, scm.cpts, [treatment], outcome, evidence, "cate",
        sorted(scm.nodes), [], "single",
        "A", "A", "A", 4, 42,
    )

    lines = code.split("\n")

    # Find the line index of the last sampling assignment (pattern: "varname = SI_EXPR")
    # Sampling lines contain rng.random() or rng.choices()
    last_sampling_line = -1
    first_return_none_line = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "rng.random()" in stripped or "rng.choices(" in stripped:
            last_sampling_line = i
        if stripped == "return None" and first_return_none_line == -1:
            first_return_none_line = i

    assert last_sampling_line >= 0, "No sampling lines found in CATE code"
    assert first_return_none_line >= 0, "No 'return None' found in CATE code"
    assert first_return_none_line > last_sampling_line, (
        f"None filter at line {first_return_none_line} is before last sampling "
        f"at line {last_sampling_line}. Filter must come after all sampling."
    )

    # Also check multi-function variant
    mutilated = scm.pgmpy_model.do([treatment])
    anc_graph = mutilated.get_ancestral_graph(nodes=[outcome] + sorted(evidence.keys()))
    anc_nodes = set(anc_graph.nodes())
    cpd_nodes = anc_nodes - {treatment}
    nodes_kept = sorted(cpd_nodes)
    nodes_pruned = sorted(set(scm.nodes) - cpd_nodes)

    code_multi = render_code(
        dag, scm.cpts, [treatment], outcome, evidence, "cate",
        nodes_kept, nodes_pruned, "multi",
        "A", "A", "A", 4, 42,
    )

    lines_multi = code_multi.split("\n")
    # In multi-function, the main function calls helpers. Find last helper call
    # (which is the last sampling) and first return None in the main function body.
    last_helper_call = -1
    first_none_multi = -1
    in_main = False
    for i, line in enumerate(lines_multi):
        stripped = line.strip()
        if stripped.startswith("def ") and i == 0 or (stripped.startswith("def func_main") or stripped.startswith(f"def x{outcome}")):
            in_main = True
            continue
        if in_main and stripped.startswith("def "):
            in_main = False
        if in_main:
            # Helper calls are like "x3 = cpd_x3(rng)" or "var_4 = func_0(rng)"
            if "(" in stripped and "rng" in stripped and "=" in stripped and "raise" not in stripped and "not in" not in stripped:
                last_helper_call = i
            if stripped == "return None" and first_none_multi == -1:
                first_none_multi = i

    assert last_helper_call >= 0, "No helper calls found in multi-function CATE main"
    assert first_none_multi >= 0, "No 'return None' found in multi-function CATE main"
    assert first_none_multi > last_helper_call, (
        f"Multi-function: None filter at line {first_none_multi} is before last "
        f"helper call at line {last_helper_call}. Filter must come after all sampling."
    )

    print("  CATE None-filter ordering: PASS (single + multi verified)")


def validate_no_comments():
    """Verify no # characters in generated code across all variants."""
    dag = nx.DiGraph()
    dag.add_edges_from([(0, 1), (1, 2)])
    scm = RLCausalSCM(dag, seed=42, decimal_places=4)

    for cf in ["A", "B", "C"]:
        for si in ["A", "B", "C"]:
            for vn in ["A", "B"]:
                for structure in ["single", "multi"]:
                    code = render_code(
                        dag, scm.cpts, [0], 2, None, "ate",
                        sorted(scm.nodes), [], structure,
                        cf, si, vn, 4, 42,
                    )
                    if "#" in code:
                        raise AssertionError(
                            f"Found # in code: CF-{cf} SI-{si} VN-{vn} {structure}"
                        )
    print("  No-comments check: PASS (all 36 combos clean)")


def validate_input_validation():
    """Verify generated code raises ValueError for invalid inputs."""
    dag = nx.DiGraph()
    dag.add_edges_from([(0, 1)])
    scm = RLCausalSCM(dag, seed=42, decimal_places=4)

    code = render_code(
        dag, scm.cpts, [0], 1, None, "ate",
        sorted(scm.nodes), [], "single",
        "A", "A", "A", 4, 42,
    )
    func = _execute_code_and_get_function(code, "x1")
    rng = random.Random(0)
    try:
        func(2, rng)
        raise AssertionError("Expected ValueError for input=2")
    except ValueError:
        pass
    print("  Input validation: PASS")


def main():
    print("=== Code-Domain Validation ===\n")

    print("--- ATE Validation ---")
    validate_ate()
    print()

    print("--- CATE Validation ---")
    validate_cate()
    print()

    print("--- JATE Validation ---")
    validate_jate()
    print()

    print("--- VN-B Identifier Validation ---")
    validate_vn_b_identifiers()
    print()

    print("--- JATE >2 Treatment Regression ---")
    validate_jate_multi_treatment()
    print()

    print("--- CATE None-Filter Ordering ---")
    validate_cate_none_filter_ordering()
    print()

    print("--- No-Comments Validation ---")
    validate_no_comments()
    print()

    print("--- Input Validation ---")
    validate_input_validation()
    print()

    print("=== All Validations Passed ===")


if __name__ == "__main__":
    main()
