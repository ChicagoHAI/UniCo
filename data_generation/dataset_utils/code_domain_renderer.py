"""
code_domain_renderer.py

Shared code renderer for translating causal DAG + CPT structures into
Python code representations. Used by both ATE and CATE/JATE code-domain
question assembly pipelines.

Supports four independent diversification axes:
  - CF (Control Flow): A=branching, B=table lookup, C=polynomial
  - SI (Sampling Idiom): A=ternary, B=weighted choice, C=cast
  - VN (Variable Naming): A=semantic, B=generic shuffled
  - QV (Query Verbalization): 1/2/3 per query type

And two structural variants:
  - single: all logic inlined in one function (for 'full' condition)
  - multi: main function + per-node helpers (for 'surgery_ancestor_prune')
"""

import itertools
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx

from data_generation.scm_mechanism_sampling import SCMBase


# ============================================================
# Axis Sampling
# ============================================================

CF_VARIANTS = ["A", "B", "C"]
SI_VARIANTS = ["A", "B", "C"]
VN_VARIANTS = ["A", "B"]
QV_VARIANTS = [1, 2, 3]


def sample_axes(example_seed: int) -> Tuple[str, str, str, int]:
    """Sample (cf, si, vn, qv) deterministically from example_seed.

    Returns:
        (cf_variant, si_variant, vn_variant, qv_variant)
    """
    rng = random.Random(example_seed)
    cf = rng.choice(CF_VARIANTS)
    si = rng.choice(SI_VARIANTS)
    vn = rng.choice(VN_VARIANTS)
    qv = rng.choice(QV_VARIANTS)
    return cf, si, vn, qv


# ============================================================
# Topological Ordering
# ============================================================

def _topo_sort_all(dag: nx.DiGraph) -> List[int]:
    """Return all nodes in topological order of the full DAG."""
    return list(nx.topological_sort(dag))


# ============================================================
# CF-C Polynomial Coefficient Computation
# ============================================================

def _compute_poly_coefficients(
    cpt_table: Dict[Tuple, float],
    parents: List[int],
) -> List[float]:
    """Compute polynomial coefficients from CPT entries.

    For 0 parents: [p]
    For 1 parent:  [a, b] where P = a + b*p1
    For 2 parents: [a, b, c, d] where P = a + b*p1 + c*p2 + d*p1*p2

    Coefficient formulas:
        a = P(0,0)
        b = P(1,0) - P(0,0)
        c = P(0,1) - P(0,0)
        d = P(1,1) - P(0,1) - P(1,0) + P(0,0)
    """
    num_parents = len(parents)
    if num_parents == 0:
        return [cpt_table[()]]
    elif num_parents == 1:
        p0 = cpt_table[(0,)]
        p1 = cpt_table[(1,)]
        a = p0
        b = p1 - p0
        return [a, b]
    elif num_parents == 2:
        p00 = cpt_table[(0, 0)]
        p10 = cpt_table[(1, 0)]
        p01 = cpt_table[(0, 1)]
        p11 = cpt_table[(1, 1)]
        a = p00
        b = p10 - p00
        c = p01 - p00
        d = p11 - p01 - p10 + p00
        return [a, b, c, d]
    else:
        raise ValueError(f"CF-C polynomial only supports 0-2 parents, got {num_parents}")


def _format_coeff(val: float, decimal_places: int) -> str:
    """Format a coefficient value, wrapping negatives in parentheses."""
    formatted = f"{val:.{decimal_places}f}"
    if val < 0:
        return f"({formatted})"
    return formatted


# ============================================================
# VN-B Name Mapping
# ============================================================

def _build_vn_b_mapping(
    all_nodes: List[int],
    treatments: List[int],
    outcome: int,
    evidence: Optional[Dict[int, int]],
    structure: str,
    cf_variant: str,
    cpts: Dict,
    example_seed: int,
) -> Dict[str, str]:
    """Build the VN-B name mapping for generic shuffled identifiers.

    Uses a separate RNG stream (example_seed + 1) to avoid interfering
    with axis sampling.

    Returns a mapping from semantic names to generic names:
        x{i} -> var_{k}  (node value variables)
        p_x{i} -> var_{k}  (probability temporaries, CF-A/C only)
        X{i}_TABLE -> var_{k}  (CF-B tables)
        cpd_x{i} -> func_{j}  (helper function names, multi only)
        input_{i} -> arg_{j}  (treatment inputs)
        main function -> func_main
    """
    name_rng = random.Random(example_seed + 1)

    # Collect all semantic identifiers that need var_* names
    var_names_needed = []

    # Node value variables in main function scope
    for node in all_nodes:
        var_names_needed.append(f"x{node}")

    # Probability temporaries (all CF variants use p_x{i} in single-function)
    if structure == "single":
        for node in all_nodes:
            parents = cpts[node]["parents"]
            if len(parents) > 0:
                var_names_needed.append(f"p_x{node}")

    # CF-B tables
    if cf_variant == "B":
        for node in all_nodes:
            parents = cpts[node]["parents"]
            if len(parents) > 0:
                var_names_needed.append(f"X{node}_TABLE")

    # CF-C coefficient lists (multi-function module scope)
    if cf_variant == "C" and structure == "multi":
        for node in all_nodes:
            parents = cpts[node]["parents"]
            if len(parents) > 0:
                var_names_needed.append(f"X{node}_COEFFS")

    # Helper function local probability temporaries (multi-function, all CF variants)
    if structure == "multi":
        for node in all_nodes:
            parents = cpts[node]["parents"]
            if len(parents) > 0:
                var_names_needed.append(f"helper_p_{node}")

    num_vars = len(var_names_needed)
    pool_size = max(num_vars * 2, 20)
    pool = list(range(pool_size))
    name_rng.shuffle(pool)
    selected_indices = pool[:num_vars]

    mapping = {}
    for i, sem_name in enumerate(var_names_needed):
        mapping[sem_name] = f"var_{selected_indices[i]}"

    return mapping


# ============================================================
# CPD Code Rendering (per-node)
# ============================================================

def _render_cpd_probability_code(
    node: int,
    cpts: Dict,
    cf_variant: str,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
    parent_name_fn,
    prob_var_name: str,
    indent: str = "    ",
) -> List[str]:
    """Render the probability computation lines for a single node's CPD.

    Returns lines of code (without trailing newlines) that compute the
    probability variable from parent values.
    """
    cpt_entry = cpts[node]
    parents = cpt_entry["parents"]
    table = cpt_entry["table"]
    lines = []

    if len(parents) == 0:
        # Root node: just a constant
        p = table[()]
        lines.append(f"{indent}{prob_var_name} = {p:.{decimal_places}f}")

    elif cf_variant == "A":
        # Branching: if/elif/else
        configs = list(itertools.product([0, 1], repeat=len(parents)))
        for ci, config in enumerate(configs):
            p = table[config]
            cond_parts = [
                f"{parent_name_fn(parents[j])} == {config[j]}"
                for j in range(len(parents))
            ]
            cond = " and ".join(cond_parts)
            if ci == 0:
                lines.append(f"{indent}if {cond}:")
            elif ci < len(configs) - 1:
                lines.append(f"{indent}elif {cond}:")
            else:
                lines.append(f"{indent}else:")
            lines.append(f"{indent}    {prob_var_name} = {p:.{decimal_places}f}")

    elif cf_variant == "B":
        # Table lookup
        table_name = f"X{node}_TABLE"
        if vn_variant == "B" and vn_mapping:
            table_name = vn_mapping.get(table_name, table_name)

        if len(parents) == 1:
            lines.append(
                f"{indent}{prob_var_name} = {table_name}[{parent_name_fn(parents[0])}]"
            )
        else:
            lines.append(
                f"{indent}{prob_var_name} = {table_name}"
                f"[{parent_name_fn(parents[0])}][{parent_name_fn(parents[1])}]"
            )

    elif cf_variant == "C":
        # Polynomial
        coeffs = _compute_poly_coefficients(table, parents)
        if len(parents) == 1:
            a, b = coeffs
            a_s = f"{a:.{decimal_places}f}"
            b_s = _format_coeff(b, decimal_places)
            p1 = parent_name_fn(parents[0])
            lines.append(
                f"{indent}{prob_var_name} = {a_s} + {b_s} * {p1}"
            )
        else:
            a, b, c, d = coeffs
            a_s = f"{a:.{decimal_places}f}"
            b_s = _format_coeff(b, decimal_places)
            c_s = _format_coeff(c, decimal_places)
            d_s = _format_coeff(d, decimal_places)
            p1 = parent_name_fn(parents[0])
            p2 = parent_name_fn(parents[1])
            lines.append(
                f"{indent}{prob_var_name} = {a_s} + {b_s} * {p1}"
                f" + {c_s} * {p2} + {d_s} * {p1} * {p2}"
            )

    return lines


def _render_sampling_line(
    var_name: str,
    prob_var_name: str,
    si_variant: str,
    indent: str = "    ",
) -> str:
    """Render the Bernoulli sampling line for a given SI variant."""
    if si_variant == "A":
        return f"{indent}{var_name} = 1 if rng.random() < {prob_var_name} else 0"
    elif si_variant == "B":
        return f"{indent}{var_name} = rng.choices([0, 1], weights=[1 - {prob_var_name}, {prob_var_name}])[0]"
    elif si_variant == "C":
        return f"{indent}{var_name} = int(rng.random() < {prob_var_name})"
    else:
        raise ValueError(f"Unknown si_variant: {si_variant}")


def _render_root_sampling_line(
    var_name: str,
    prob_value: float,
    si_variant: str,
    decimal_places: int,
    indent: str = "    ",
) -> str:
    """Render sampling for a root node (inline probability constant)."""
    p_str = f"{prob_value:.{decimal_places}f}"
    if si_variant == "A":
        return f"{indent}{var_name} = 1 if rng.random() < {p_str} else 0"
    elif si_variant == "B":
        return f"{indent}{var_name} = rng.choices([0, 1], weights=[1 - {p_str}, {p_str}])[0]"
    elif si_variant == "C":
        return f"{indent}{var_name} = int(rng.random() < {p_str})"
    else:
        raise ValueError(f"Unknown si_variant: {si_variant}")


# ============================================================
# CF-B Table Rendering
# ============================================================

def _render_cf_b_tables(
    nodes: List[int],
    cpts: Dict,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
) -> List[str]:
    """Render module-scope CF-B table declarations for nodes with parents."""
    lines = []
    for node in nodes:
        parents = cpts[node]["parents"]
        if len(parents) == 0:
            continue
        table = cpts[node]["table"]
        table_name = f"X{node}_TABLE"
        if vn_variant == "B" and vn_mapping:
            table_name = vn_mapping.get(table_name, table_name)

        if len(parents) == 1:
            vals = [f"{table[(0,)]:.{decimal_places}f}", f"{table[(1,)]:.{decimal_places}f}"]
            lines.append(f"{table_name} = [{vals[0]}, {vals[1]}]")
        elif len(parents) == 2:
            lines.append(f"{table_name} = [")
            for p1_val in (0, 1):
                row = [
                    f"{table[(p1_val, 0)]:.{decimal_places}f}",
                    f"{table[(p1_val, 1)]:.{decimal_places}f}",
                ]
                lines.append(f"    [{row[0]}, {row[1]}],")
            lines.append("]")
    return lines


# ============================================================
# CF-C Coefficient List Rendering (module-scope, multi-function)
# ============================================================

def _render_cf_c_coeff_lists(
    nodes: List[int],
    cpts: Dict,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
) -> List[str]:
    """Render module-scope coefficient list declarations for CF-C multi-function."""
    lines = []
    for node in nodes:
        parents = cpts[node]["parents"]
        if len(parents) == 0:
            continue
        coeffs = _compute_poly_coefficients(cpts[node]["table"], parents)
        coeff_name = f"X{node}_COEFFS"
        if vn_variant == "B" and vn_mapping:
            coeff_name = vn_mapping.get(coeff_name, coeff_name)
        vals = [f"{c:.{decimal_places}f}" for c in coeffs]
        lines.append(f"{coeff_name} = [{', '.join(vals)}]")
    return lines


# ============================================================
# Single-Function Renderer
# ============================================================

def render_single_function(
    dag: nx.DiGraph,
    cpts: Dict,
    treatments: List[int],
    outcome: int,
    evidence: Optional[Dict[int, int]],
    query_type: str,
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
) -> str:
    """Render a single-function code block for the 'full' condition variant.

    All graph variables are sampled inline in one function body.
    """
    topo_order = _topo_sort_all(dag)
    all_nodes = list(dag.nodes())
    treatment_set = set(treatments)
    sorted_treatments = sorted(treatments)

    # Determine function name and return type
    if vn_variant == "A":
        func_name = f"x{outcome}"
        input_names = [f"input_{t}" for t in sorted_treatments]
    else:
        func_name = "func_main"
        input_names = [f"arg_{i}" for i in range(len(sorted_treatments))]

    if query_type == "cate":
        ret_type = "Optional[int]"
    else:
        ret_type = "int"

    # Build imports
    import_lines = ["import random"]
    if query_type == "cate":
        import_lines.append("from typing import Optional")

    # Build module-scope declarations
    module_lines = []
    if cf_variant == "B":
        module_lines = _render_cf_b_tables(
            topo_order, cpts, vn_variant, vn_mapping, decimal_places
        )

    # Function signature
    params = []
    for inp_name in input_names:
        params.append(f"{inp_name}: int")
    params.append("rng: random.Random")
    sig = f"def {func_name}({', '.join(params)}) -> {ret_type}:"

    body_lines = []

    # Input validation
    for inp_name in input_names:
        body_lines.append(f'    if {inp_name} not in (0, 1):')
        body_lines.append(f'        raise ValueError("{inp_name} must be 0 or 1")')
    if input_names:
        body_lines.append("")

    # Sample all variables in topological order
    treatment_input_map = {
        t: input_names[i] for i, t in enumerate(sorted_treatments)
    }

    for node in topo_order:
        parents = cpts[node]["parents"]
        table = cpts[node]["table"]

        # Variable name for this node
        if vn_variant == "A":
            node_var = f"x{node}"
        else:
            node_var = vn_mapping[f"x{node}"]

        if len(parents) == 0:
            # Root node: inline constant probability + sampling
            p = table[()]
            body_lines.append(_render_root_sampling_line(
                node_var, p, si_variant, decimal_places
            ))
        else:
            # Non-root: compute probability then sample
            if vn_variant == "A":
                prob_var = f"p_x{node}"
            else:
                prob_var = vn_mapping[f"p_x{node}"]

            def parent_name_fn(p_node):
                if vn_variant == "A":
                    return f"x{p_node}"
                else:
                    return vn_mapping[f"x{p_node}"]

            cpd_lines = _render_cpd_probability_code(
                node, cpts, cf_variant, vn_variant, vn_mapping,
                decimal_places, parent_name_fn, prob_var, indent="    "
            )
            body_lines.extend(cpd_lines)
            body_lines.append(_render_sampling_line(
                node_var, prob_var, si_variant, indent="    "
            ))

        # Intervention overwrite (immediately after sampling)
        if node in treatment_set:
            body_lines.append(f"    {node_var} = {treatment_input_map[node]}")

        body_lines.append("")

    # CATE: condition check after all variables sampled
    if query_type == "cate" and evidence:
        for ev_node, ev_val in sorted(evidence.items()):
            if vn_variant == "A":
                ev_var = f"x{ev_node}"
            else:
                ev_var = vn_mapping[f"x{ev_node}"]
            body_lines.append(f"    if {ev_var} != {ev_val}:")
            body_lines.append(f"        return None")
        body_lines.append("")

    # Return outcome
    if vn_variant == "A":
        outcome_var = f"x{outcome}"
    else:
        outcome_var = vn_mapping[f"x{outcome}"]
    body_lines.append(f"    return {outcome_var}")

    # Compose
    parts = []
    parts.extend(import_lines)
    parts.append("")
    if module_lines:
        parts.extend(module_lines)
        parts.append("")
    parts.append(sig)
    parts.extend(body_lines)

    return "\n".join(parts)


# ============================================================
# Multi-Function Renderer
# ============================================================

def render_multi_function(
    dag: nx.DiGraph,
    cpts: Dict,
    treatments: List[int],
    outcome: int,
    evidence: Optional[Dict[int, int]],
    query_type: str,
    nodes_kept: List[int],
    nodes_pruned: List[int],
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
) -> str:
    """Render a multi-function code block for the 'surgery_ancestor_prune' condition.

    Main function is defined first, calling all helpers in topological order.
    Kept node helpers have full CPD implementations. Pruned node helpers
    are stubs (signature + pass body).
    """
    topo_order = _topo_sort_all(dag)
    treatment_set = set(treatments)
    sorted_treatments = sorted(treatments)
    kept_set = set(nodes_kept)
    pruned_set = set(nodes_pruned)

    # Determine naming
    if vn_variant == "A":
        func_name = f"x{outcome}"
        input_names = [f"input_{t}" for t in sorted_treatments]
    else:
        func_name = "func_main"
        input_names = [f"arg_{i}" for i in range(len(sorted_treatments))]

    if query_type == "cate":
        ret_type = "Optional[int]"
    else:
        ret_type = "int"

    # Build imports
    import_lines = ["import random"]
    if query_type == "cate":
        import_lines.append("from typing import Optional")

    # Module-scope declarations
    module_lines = []
    if cf_variant == "B":
        # Only render tables for kept nodes (pruned have stub bodies)
        kept_nodes_in_order = [n for n in topo_order if n in kept_set]
        module_lines = _render_cf_b_tables(
            kept_nodes_in_order, cpts, vn_variant, vn_mapping, decimal_places
        )
    elif cf_variant == "C":
        kept_nodes_in_order = [n for n in topo_order if n in kept_set]
        module_lines = _render_cf_c_coeff_lists(
            kept_nodes_in_order, cpts, vn_variant, vn_mapping, decimal_places
        )

    # Assign helper function names
    helper_names = {}
    helper_counter = 0
    for node in topo_order:
        if vn_variant == "A":
            helper_names[node] = f"cpd_x{node}"
        else:
            helper_names[node] = f"func_{helper_counter}"
            helper_counter += 1

    # === Main function ===
    params = []
    for inp_name in input_names:
        params.append(f"{inp_name}: int")
    params.append("rng: random.Random")
    main_sig = f"def {func_name}({', '.join(params)}) -> {ret_type}:"

    main_body = []

    # Input validation
    for inp_name in input_names:
        main_body.append(f'    if {inp_name} not in (0, 1):')
        main_body.append(f'        raise ValueError("{inp_name} must be 0 or 1")')
    if input_names:
        main_body.append("")

    treatment_input_map = {
        t: input_names[i] for i, t in enumerate(sorted_treatments)
    }

    # Call helpers in topological order (ALL nodes)
    for node in topo_order:
        parents = cpts[node]["parents"]
        h_name = helper_names[node]

        # Build call arguments
        if vn_variant == "A":
            node_var = f"x{node}"
            call_args = [f"x{p}" for p in parents] + ["rng"]
        else:
            node_var = vn_mapping[f"x{node}"]
            call_args = [vn_mapping[f"x{p}"] for p in parents] + ["rng"]

        main_body.append(f"    {node_var} = {h_name}({', '.join(call_args)})")

        # Intervention overwrite
        if node in treatment_set:
            main_body.append(f"    {node_var} = {treatment_input_map[node]}")

    main_body.append("")

    # CATE: condition check
    if query_type == "cate" and evidence:
        for ev_node, ev_val in sorted(evidence.items()):
            if vn_variant == "A":
                ev_var = f"x{ev_node}"
            else:
                ev_var = vn_mapping[f"x{ev_node}"]
            main_body.append(f"    if {ev_var} != {ev_val}:")
            main_body.append(f"        return None")
        main_body.append("")

    # Return outcome
    if vn_variant == "A":
        outcome_var = f"x{outcome}"
    else:
        outcome_var = vn_mapping[f"x{outcome}"]
    main_body.append(f"    return {outcome_var}")

    # === Helper functions ===
    helper_funcs = []
    for node in topo_order:
        parents = cpts[node]["parents"]
        table = cpts[node]["table"]
        h_name = helper_names[node]

        # Build parameter list
        if vn_variant == "A":
            h_params = [f"x{p}: int" for p in parents] + ["rng: random.Random"]
        else:
            h_params = [f"arg_{i}: int" for i in range(len(parents))] + ["rng: random.Random"]

        h_sig = f"def {h_name}({', '.join(h_params)}) -> int:"

        if node in pruned_set:
            # Stub: signature + pass
            helper_funcs.append(h_sig)
            helper_funcs.append("    pass")
            helper_funcs.append("")
            continue

        h_body = []

        if len(parents) == 0:
            # Root node
            p = table[()]
            p_str = f"{p:.{decimal_places}f}"
            if si_variant == "A":
                h_body.append(f"    return 1 if rng.random() < {p_str} else 0")
            elif si_variant == "B":
                h_body.append(f"    return rng.choices([0, 1], weights=[1 - {p_str}, {p_str}])[0]")
            elif si_variant == "C":
                h_body.append(f"    return int(rng.random() < {p_str})")
        else:
            # Non-root: compute probability then return sampled value
            if vn_variant == "A":
                if cf_variant in ("A", "C"):
                    prob_var = f"p_x{node}"
                else:
                    prob_var = "p"
            else:
                prob_var = vn_mapping[f"helper_p_{node}"]

            def helper_parent_name_fn(p_node, _parents=parents):
                if vn_variant == "A":
                    return f"x{p_node}"
                else:
                    idx = _parents.index(p_node)
                    return f"arg_{idx}"

            if cf_variant == "B":
                # Table lookup using module-scope table
                table_name = f"X{node}_TABLE"
                if vn_variant == "B" and vn_mapping:
                    table_name = vn_mapping.get(table_name, table_name)

                if len(parents) == 1:
                    h_body.append(
                        f"    {prob_var} = {table_name}[{helper_parent_name_fn(parents[0])}]"
                    )
                else:
                    h_body.append(
                        f"    {prob_var} = {table_name}"
                        f"[{helper_parent_name_fn(parents[0])}][{helper_parent_name_fn(parents[1])}]"
                    )
            elif cf_variant == "C":
                # Polynomial using module-scope coefficient list
                coeff_name = f"X{node}_COEFFS"
                if vn_variant == "B" and vn_mapping:
                    coeff_name = vn_mapping.get(coeff_name, coeff_name)

                if len(parents) == 1:
                    p1 = helper_parent_name_fn(parents[0])
                    h_body.append(
                        f"    {prob_var} = {coeff_name}[0] + {coeff_name}[1] * {p1}"
                    )
                else:
                    p1 = helper_parent_name_fn(parents[0])
                    p2 = helper_parent_name_fn(parents[1])
                    h_body.append(
                        f"    {prob_var} = {coeff_name}[0] + {coeff_name}[1] * {p1}"
                        f" + {coeff_name}[2] * {p2} + {coeff_name}[3] * {p1} * {p2}"
                    )
            else:
                # CF-A: branching inside helper
                cpd_lines = _render_cpd_probability_code(
                    node, cpts, cf_variant, vn_variant, vn_mapping,
                    decimal_places, helper_parent_name_fn, prob_var, indent="    "
                )
                h_body.extend(cpd_lines)

            # Sampling return
            if si_variant == "A":
                h_body.append(f"    return 1 if rng.random() < {prob_var} else 0")
            elif si_variant == "B":
                h_body.append(f"    return rng.choices([0, 1], weights=[1 - {prob_var}, {prob_var}])[0]")
            elif si_variant == "C":
                h_body.append(f"    return int(rng.random() < {prob_var})")

        helper_funcs.append(h_sig)
        helper_funcs.extend(h_body)
        helper_funcs.append("")

    # Compose
    parts = []
    parts.extend(import_lines)
    parts.append("")
    if module_lines:
        parts.extend(module_lines)
        parts.append("")
    parts.append(main_sig)
    parts.extend(main_body)
    parts.append("")
    parts.extend(helper_funcs)

    # Strip trailing empty lines
    while parts and parts[-1] == "":
        parts.pop()

    return "\n".join(parts)


# ============================================================
# Top-Level Render Function
# ============================================================

def render_code(
    dag: nx.DiGraph,
    cpts: Dict,
    treatments: List[int],
    outcome: int,
    evidence: Optional[Dict[int, int]],
    query_type: str,
    nodes_kept: List[int],
    nodes_pruned: List[int],
    structure: str,
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
    decimal_places: int,
    example_seed: int,
) -> str:
    """Top-level rendering function.

    Args:
        dag: The full causal DAG.
        cpts: CPT dictionary from SCM.
        treatments: List of treatment node IDs.
        outcome: Outcome node ID.
        evidence: Evidence dict (CATE only, else None or empty).
        query_type: "ate", "cate", or "jate".
        nodes_kept: Nodes with full CPD implementations (for multi-function).
        nodes_pruned: Nodes that become stubs (for multi-function).
        structure: "single" or "multi".
        cf_variant: "A", "B", or "C".
        si_variant: "A", "B", or "C".
        vn_variant: "A" or "B".
        decimal_places: Number of decimal places for CPT values.
        example_seed: Seed for VN-B name shuffling.

    Returns:
        Complete Python code string (no comments, includes imports).
    """
    all_nodes = _topo_sort_all(dag)

    # Build VN-B mapping if needed
    vn_mapping = None
    if vn_variant == "B":
        vn_mapping = _build_vn_b_mapping(
            all_nodes, treatments, outcome, evidence, structure,
            cf_variant, cpts, example_seed,
        )

    if structure == "single":
        return render_single_function(
            dag, cpts, treatments, outcome, evidence, query_type,
            cf_variant, si_variant, vn_variant, vn_mapping, decimal_places,
        )
    elif structure == "multi":
        return render_multi_function(
            dag, cpts, treatments, outcome, evidence, query_type,
            nodes_kept, nodes_pruned,
            cf_variant, si_variant, vn_variant, vn_mapping, decimal_places,
        )
    else:
        raise ValueError(f"Unknown structure: {structure}")


# ============================================================
# QV Text Templates
# ============================================================

def render_qv_text(
    query_type: str,
    qv_variant: int,
    func_name: str,
    num_treatments: int = 1,
) -> str:
    """Render the query verbalization text for a given query type and variant.

    Args:
        query_type: "ate", "cate", or "jate".
        qv_variant: 1, 2, or 3.
        func_name: Name of the entry-point function (e.g., "x1" or "func_main").
        num_treatments: Number of treatment variables (used for JATE wording).

    Returns:
        Query text string.
    """
    if query_type == "ate":
        return _render_ate_qv(qv_variant, func_name)
    elif query_type == "cate":
        return _render_cate_qv(qv_variant, func_name)
    elif query_type == "jate":
        return _render_jate_qv(qv_variant, func_name, num_treatments)
    else:
        raise ValueError(f"Unknown query_type: {query_type}")


def _render_ate_qv(qv: int, fn: str) -> str:
    if qv == 1:
        return (
            f"How does the input affect the expected return value of `{fn}`? "
            f"In other words, what is the change in the probability that "
            f"`{fn}` returns 1 as the input changes from 0 to 1?"
        )
    elif qv == 2:
        return (
            f"Suppose you call `{fn}` a large number of times with the input "
            f"set to 1 and, separately, a large number of times with the input "
            f"set to 0. What is the difference in the fraction of calls that "
            f"return 1 between the two groups?"
        )
    elif qv == 3:
        return (
            f"Let p(x) denote the probability that `{fn}` returns 1 when the "
            f"input is fixed to x. Compute p(1) - p(0)."
        )
    else:
        raise ValueError(f"Unknown ATE qv_variant: {qv}")


def _render_cate_qv(qv: int, fn: str) -> str:
    if qv == 1:
        return (
            f"Considering only the calls where `{fn}` does not return None, "
            f"how does the input affect the expected return value? In other "
            f"words, what is the change in the probability that `{fn}` returns "
            f"1, among non-None results, as the input changes from 0 to 1?"
        )
    elif qv == 2:
        return (
            f"Run `{fn}` many times with the input set to 1, and discard every "
            f"call that returns None. Separately, do the same with the input "
            f"set to 0. Among the kept results in each group, what is the "
            f"difference in the rate of returning 1?"
        )
    elif qv == 3:
        return (
            f"Focus on the non-None outputs of `{fn}`. Define p(x) as the "
            f"probability that `{fn}` returns 1 (excluding calls where it "
            f"returns None) when the input is fixed to x. What is p(1) - p(0)?"
        )
    else:
        raise ValueError(f"Unknown CATE qv_variant: {qv}")


def _render_jate_qv(qv: int, fn: str, num_treatments: int = 2) -> str:
    n = num_treatments
    inputs_word = "two inputs" if n == 2 else f"{n} inputs"
    all_word = "both" if n == 2 else "all"

    # Build formal parameter names for QV-3
    param_names = [chr(ord("a") + i) for i in range(n)]
    param_list = ", ".join(param_names)
    ones = ", ".join(["1"] * n)
    zeros = ", ".join(["0"] * n)

    if qv == 1:
        return (
            f"How do the {inputs_word} jointly affect the expected return value "
            f"of `{fn}`? In other words, what is the change in the probability "
            f"that `{fn}` returns 1 when {all_word} inputs are set to 1, compared to "
            f"when {all_word} are set to 0?"
        )
    elif qv == 2:
        return (
            f"Imagine running `{fn}` many times under two configurations: once "
            f"where {all_word} inputs are 1, and once where {all_word} inputs are 0. What "
            f"is the difference in the average output between these two "
            f"configurations?"
        )
    elif qv == 3:
        return (
            f"Define f({param_list}) as the expected output of `{fn}` when the "
            f"inputs are ({param_list}). What is f({ones}) - f({zeros})?"
        )
    else:
        raise ValueError(f"Unknown JATE qv_variant: {qv}")



# ============================================================
# Observational Association Renderer
# ============================================================

def _build_observational_vn_mapping(nodes: List[int], example_seed: int) -> Dict[str, str]:
    rng = random.Random(example_seed + 17)
    semantic_names = []
    for node in nodes:
        semantic_names.append(f"x{node}")
        semantic_names.append(f"p_x{node}")
        semantic_names.append(f"X{node}_TABLE")
    pool = list(range(10_000))
    rng.shuffle(pool)
    return {name: f"var_{pool[idx]}" for idx, name in enumerate(semantic_names)}


def _obs_node_name(node: int, vn_variant: str, vn_mapping: Optional[Dict[str, str]]) -> str:
    if vn_variant == "A":
        return f"x{node}"
    return vn_mapping[f"x{node}"]


def _obs_prob_name(node: int, vn_variant: str, vn_mapping: Optional[Dict[str, str]]) -> str:
    if vn_variant == "A":
        return f"p_x{node}"
    return vn_mapping[f"p_x{node}"]


def get_observational_entry_function_name(
    return_nodes: List[int],
    vn_variant: str,
    *,
    query_type: str,
) -> str:
    if vn_variant == "B":
        return "func_main"
    if query_type == "marginal_prob":
        return f"x{return_nodes[0]}"
    if query_type == "conditional_prob":
        return f"x{return_nodes[0]}_given"
    return "x" + "_".join(str(node) for node in return_nodes)


def _observational_return_type(query_type: str, return_nodes: List[int]) -> str:
    if query_type == "conditional_prob":
        return "Optional[int]"
    if len(return_nodes) > 1:
        return "tuple[int, ...]"
    return "int"


def _render_observational_single_code(
    dag: nx.DiGraph,
    cpts: Dict,
    *,
    query_type: str,
    return_nodes: List[int],
    evidence: Dict[int, int],
    topo_nodes: List[int],
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
) -> str:
    func_name = get_observational_entry_function_name(return_nodes, vn_variant, query_type=query_type)
    return_type = _observational_return_type(query_type, return_nodes)
    lines = ["import random"]
    if query_type == "conditional_prob":
        lines.append("from typing import Optional")
    lines.append("")
    if cf_variant == "B":
        table_lines = _render_cf_b_tables(topo_nodes, cpts, vn_variant, vn_mapping, decimal_places)
        if table_lines:
            lines.extend(table_lines)
            lines.append("")

    lines.append(f"def {func_name}(rng: random.Random) -> {return_type}:")
    if not topo_nodes:
        lines.append("    raise ValueError(\"No nodes are available to sample\")")
        return "\n".join(lines)

    for node in topo_nodes:
        parents = cpts[node]["parents"]
        node_var = _obs_node_name(node, vn_variant, vn_mapping)
        if len(parents) == 0:
            lines.append(_render_root_sampling_line(node_var, cpts[node]["table"][()], si_variant, decimal_places))
        else:
            prob_var = _obs_prob_name(node, vn_variant, vn_mapping)

            def parent_name_fn(parent_node, _vn_variant=vn_variant, _vn_mapping=vn_mapping):
                return _obs_node_name(parent_node, _vn_variant, _vn_mapping)

            lines.extend(_render_cpd_probability_code(
                node,
                cpts,
                cf_variant,
                vn_variant,
                vn_mapping,
                decimal_places,
                parent_name_fn,
                prob_var,
                indent="    ",
            ))
            lines.append(_render_sampling_line(node_var, prob_var, si_variant, indent="    "))
        lines.append("")

    if evidence:
        cond = " and ".join(
            f"{_obs_node_name(node, vn_variant, vn_mapping)} == {value}"
            for node, value in sorted(evidence.items())
        )
        lines.append(f"    if not ({cond}):")
        lines.append("        return None")
        lines.append("")

    return_exprs = [_obs_node_name(node, vn_variant, vn_mapping) for node in return_nodes]
    if len(return_exprs) == 1:
        lines.append(f"    return {return_exprs[0]}")
    else:
        lines.append(f"    return ({', '.join(return_exprs)})")
    return "\n".join(lines)


def _render_observational_multi_code(
    dag: nx.DiGraph,
    cpts: Dict,
    *,
    query_type: str,
    return_nodes: List[int],
    evidence: Dict[int, int],
    topo_nodes: List[int],
    nodes_pruned: List[int],
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
    vn_mapping: Optional[Dict[str, str]],
    decimal_places: int,
) -> str:
    func_name = get_observational_entry_function_name(return_nodes, vn_variant, query_type=query_type)
    return_type = _observational_return_type(query_type, return_nodes)
    kept_set = set(topo_nodes)
    pruned_set = set(nodes_pruned)
    full_topo_order = _topo_sort_all(dag)
    relevant_topo_order = [node for node in full_topo_order if node in kept_set or node in pruned_set]

    import_lines = ["import random"]
    if query_type == "conditional_prob":
        import_lines.append("from typing import Optional")

    module_lines = []
    if cf_variant == "B":
        module_lines = _render_cf_b_tables(topo_nodes, cpts, vn_variant, vn_mapping, decimal_places)
    elif cf_variant == "C":
        module_lines = _render_cf_c_coeff_lists(topo_nodes, cpts, vn_variant, vn_mapping, decimal_places)

    helper_names = {}
    helper_counter = 0
    for node in relevant_topo_order:
        if vn_variant == "A":
            helper_names[node] = f"cpd_x{node}"
        else:
            helper_names[node] = f"func_{helper_counter}"
            helper_counter += 1

    main_body = []
    for node in relevant_topo_order:
        parents = cpts[node]["parents"]
        if vn_variant == "A":
            node_var = f"x{node}"
            call_args = [f"x{parent}" for parent in parents] + ["rng"]
        else:
            node_var = vn_mapping[f"x{node}"]
            call_args = [vn_mapping[f"x{parent}"] for parent in parents] + ["rng"]
        main_body.append(f"    {node_var} = {helper_names[node]}({', '.join(call_args)})")
    main_body.append("")

    if evidence:
        cond = " and ".join(
            f"{_obs_node_name(node, vn_variant, vn_mapping)} == {value}"
            for node, value in sorted(evidence.items())
        )
        main_body.append(f"    if not ({cond}):")
        main_body.append("        return None")
        main_body.append("")

    return_exprs = [_obs_node_name(node, vn_variant, vn_mapping) for node in return_nodes]
    if len(return_exprs) == 1:
        main_body.append(f"    return {return_exprs[0]}")
    else:
        main_body.append(f"    return ({', '.join(return_exprs)})")

    helper_funcs = []
    for node in relevant_topo_order:
        parents = cpts[node]["parents"]
        helper_name = helper_names[node]
        if vn_variant == "A":
            params = [f"x{parent}: int" for parent in parents] + ["rng: random.Random"]
        else:
            params = [f"arg_{idx}: int" for idx in range(len(parents))] + ["rng: random.Random"]
        helper_funcs.append(f"def {helper_name}({', '.join(params)}) -> int:")
        if node in pruned_set:
            helper_funcs.append("    pass")
            helper_funcs.append("")
            continue
        if len(parents) == 0:
            helper_funcs.append(_render_root_sampling_line("return", cpts[node]["table"][()], si_variant, decimal_places).replace("return = ", "return "))
        else:
            if vn_variant == "A":
                prob_var = f"p_x{node}" if cf_variant in ("A", "C") else "p"
            else:
                prob_var = vn_mapping[f"p_x{node}"]

            def helper_parent_name_fn(parent_node, _parents=parents):
                if vn_variant == "A":
                    return f"x{parent_node}"
                idx = _parents.index(parent_node)
                return f"arg_{idx}"

            if cf_variant == "B":
                table_name = f"X{node}_TABLE"
                if vn_variant == "B" and vn_mapping:
                    table_name = vn_mapping.get(table_name, table_name)
                if len(parents) == 1:
                    helper_funcs.append(f"    {prob_var} = {table_name}[{helper_parent_name_fn(parents[0])}]")
                else:
                    helper_funcs.append(
                        f"    {prob_var} = {table_name}[{helper_parent_name_fn(parents[0])}][{helper_parent_name_fn(parents[1])}]"
                    )
            elif cf_variant == "C":
                coeff_name = f"X{node}_COEFFS"
                if vn_variant == "B" and vn_mapping:
                    coeff_name = vn_mapping.get(coeff_name, coeff_name)
                if len(parents) == 1:
                    p1 = helper_parent_name_fn(parents[0])
                    helper_funcs.append(f"    {prob_var} = {coeff_name}[0] + {coeff_name}[1] * {p1}")
                else:
                    p1 = helper_parent_name_fn(parents[0])
                    p2 = helper_parent_name_fn(parents[1])
                    helper_funcs.append(
                        f"    {prob_var} = {coeff_name}[0] + {coeff_name}[1] * {p1} + {coeff_name}[2] * {p2} + {coeff_name}[3] * {p1} * {p2}"
                    )
            else:
                helper_funcs.extend(_render_cpd_probability_code(
                    node,
                    cpts,
                    cf_variant,
                    vn_variant,
                    vn_mapping,
                    decimal_places,
                    helper_parent_name_fn,
                    prob_var,
                    indent="    ",
                ))
            if si_variant == "A":
                helper_funcs.append(f"    return 1 if rng.random() < {prob_var} else 0")
            elif si_variant == "B":
                helper_funcs.append(f"    return rng.choices([0, 1], weights=[1 - {prob_var}, {prob_var}])[0]")
            elif si_variant == "C":
                helper_funcs.append(f"    return int(rng.random() < {prob_var})")
            else:
                raise ValueError(f"Unknown si_variant: {si_variant}")
        helper_funcs.append("")

    parts = []
    parts.extend(import_lines)
    parts.append("")
    if module_lines:
        parts.extend(module_lines)
        parts.append("")
    parts.append(f"def {func_name}(rng: random.Random) -> {return_type}:")
    parts.extend(main_body)
    parts.append("")
    parts.extend(helper_funcs)
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def render_observational_code(
    dag: nx.DiGraph,
    cpts: Dict,
    *,
    query_type: str,
    return_nodes: List[int],
    evidence: Optional[Dict[int, int]],
    nodes_kept: List[int],
    nodes_pruned: Optional[List[int]] = None,
    structure: str = "single",
    cf_variant: str,
    si_variant: str,
    vn_variant: str,
    decimal_places: int,
    example_seed: int,
) -> str:
    """Render observational SCM sampling code for association-style queries."""
    evidence = evidence or {}
    nodes_pruned = nodes_pruned or []
    kept_set = set(nodes_kept)
    topo_nodes = [node for node in nx.topological_sort(dag) if node in kept_set]
    vn_mapping = _build_observational_vn_mapping(
        sorted(set(topo_nodes) | set(nodes_pruned)), example_seed
    ) if vn_variant == "B" else None
    if structure == "single":
        return _render_observational_single_code(
            dag,
            cpts,
            query_type=query_type,
            return_nodes=return_nodes,
            evidence=evidence,
            topo_nodes=topo_nodes,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            vn_mapping=vn_mapping,
            decimal_places=decimal_places,
        )
    if structure == "multi":
        return _render_observational_multi_code(
            dag,
            cpts,
            query_type=query_type,
            return_nodes=return_nodes,
            evidence=evidence,
            topo_nodes=topo_nodes,
            nodes_pruned=nodes_pruned,
            cf_variant=cf_variant,
            si_variant=si_variant,
            vn_variant=vn_variant,
            vn_mapping=vn_mapping,
            decimal_places=decimal_places,
        )
    raise ValueError(f"Unknown observational structure: {structure}")

# ============================================================
# Question Text Composition
# ============================================================

CODE_PREAMBLE = "Consider the following python code without using any external execution environment."

CODE_ANSWER_FORMAT = (
    "Think step-by-step to solve this code reasoning question. "
    "Your final answer should be rounded to four decimal places "
    "and included inside \\boxed{}, as below: \\boxed{<your final answer>}."
)


def compose_question_text(code: str, qv_text: str) -> str:
    """Compose the full question text from code and QV text.

    Structure:
        <preamble>
        <blank line>
        ```python
        <code>
        ```
        <blank line>
        <qv_text>
        <blank line>
        <answer_format>
    """
    return (
        f"{CODE_PREAMBLE}\n"
        f"\n"
        f"```python\n"
        f"{code}\n"
        f"```\n"
        f"\n"
        f"{qv_text}\n"
        f"\n"
        f"{CODE_ANSWER_FORMAT}"
    )


def get_entry_function_name(
    outcome: int,
    vn_variant: str,
) -> str:
    """Get the entry-point function name for QV text."""
    if vn_variant == "A":
        return f"x{outcome}"
    else:
        return "func_main"


# ============================================================
# Shared-Latent Counterfactual Helpers
# ============================================================

def compute_code_full_nodes(scm: SCMBase) -> Tuple[List[int], List[int], List[int], List[int]]:
    all_nodes = sorted(scm.nodes)
    return all_nodes, [], all_nodes, all_nodes


def factual_relevant_nodes(dag: nx.DiGraph, evidence: Dict[int, int], outcome: int) -> List[int]:
    if not evidence:
        return [outcome]
    relevant = set(evidence)
    for node in evidence:
        relevant.update(nx.ancestors(dag, node))
    return sorted(relevant)


def counterfactual_relevant_nodes(scm: SCMBase, treatments: Dict[int, int], outcome: int) -> List[int]:
    mutilated = scm.pgmpy_model.do(sorted(treatments))
    anc_graph = mutilated.get_ancestral_graph(nodes=[outcome])
    return sorted(set(anc_graph.nodes()))


def latent_map_name(vn_variant: str) -> str:
    return 'latents' if vn_variant == 'A' else 'store'


def build_state_name(vn_variant: str) -> str:
    return 'build_state' if vn_variant == 'A' else 'make_state'


def build_factual_state_name(vn_variant: str) -> str:
    return 'build_factual_state' if vn_variant == 'A' else 'make_factual'


def build_counterfactual_state_name(vn_variant: str) -> str:
    return 'build_counterfactual_state' if vn_variant == 'A' else 'make_shifted'


def draw_latents_name(vn_variant: str) -> str:
    return 'draw_latents' if vn_variant == 'A' else 'draw_store'


def selector_key_map_name(node: int, vn_variant: str) -> str:
    return f'SELECTOR_KEYS_{node}' if vn_variant == 'A' else f'MAP_{node}'


def selector_list_name(node: int, vn_variant: str) -> str:
    return f'SELECTOR_LIST_{node}' if vn_variant == 'A' else f'LIST_{node}'


def build_selector_name_map(
    scm: SCMBase,
    nodes_used: List[int],
    vn_variant: str,
) -> Dict[Tuple[int, Tuple[int, ...]], str]:
    selector_name_map: Dict[Tuple[int, Tuple[int, ...]], str] = {}
    counter = 0
    for node in nodes_used:
        for config in sorted(scm.cpts[node]['table']):
            selector_name_map[(node, config)] = (
                f'latent_{counter}' if vn_variant == 'A' else f'slot_{counter}'
            )
            counter += 1
    return selector_name_map


def selector_var_name(
    selector_name_map: Dict[Tuple[int, Tuple[int, ...]], str],
    node: int,
    config: Tuple[int, ...],
) -> str:
    return selector_name_map[(node, config)]


def render_sampling_line(var_name: str, p: float, si_variant: str, indent: str = '    ') -> str:
    p_str = f"{p:.4f}"
    if si_variant == 'A':
        return f"{indent}{var_name} = 1 if rng.random() < {p_str} else 0"
    if si_variant == 'B':
        return f"{indent}{var_name} = rng.choices([0, 1], weights=[1 - {p_str}, {p_str}])[0]"
    if si_variant == 'C':
        return f"{indent}{var_name} = int(rng.random() < {p_str})"
    raise ValueError(f"Unknown si_variant: {si_variant}")


def render_selector_lookup_declarations(
    scm: SCMBase,
    nodes_used: List[int],
    cf_variant: str,
    vn_variant: str,
    selector_name_map: Dict[Tuple[int, Tuple[int, ...]], str],
) -> List[str]:
    lines = []
    if cf_variant == 'B':
        for node in nodes_used:
            parents = scm.cpts[node]['parents']
            if not parents:
                continue
            items = []
            for config in itertools.product([0, 1], repeat=len(parents)):
                items.append(f'{config}: "{selector_var_name(selector_name_map, node, config)}"')
            lines.append(f"{selector_key_map_name(node, vn_variant)} = {{{', '.join(items)}}}")
        if lines:
            lines.append('')
    elif cf_variant == 'C':
        for node in nodes_used:
            parents = scm.cpts[node]['parents']
            if not parents:
                continue
            selector_names = [f'"{selector_var_name(selector_name_map, node, config)}"' for config in itertools.product([0, 1], repeat=len(parents))]
            lines.append(f"{selector_list_name(node, vn_variant)} = [{', '.join(selector_names)}]")
        if lines:
            lines.append('')
    return lines


def render_build_state_lines(
    scm: SCMBase,
    nodes_used: List[int],
    treatments: Dict[int, int],
    *,
    cf_variant: str,
    vn_variant: str,
    selector_name_map: Dict[Tuple[int, Tuple[int, ...]], str],
) -> List[str]:
    lines = []
    latent_name = latent_map_name(vn_variant)
    topo_order = list(nx.topological_sort(scm.dag))
    ordered_nodes = [node for node in topo_order if node in set(nodes_used)]
    for node in ordered_nodes:
        node_key = f"x{node}"
        lines.append(f"    if \"{node_key}\" in overrides:")
        lines.append(f"        state[\"{node_key}\"] = overrides[\"{node_key}\"]")
        lines.append("    else:")
        parents = scm.cpts[node]['parents']
        if not parents:
            selector_name = selector_var_name(selector_name_map, node, ())
            lines.append(f"        state[\"{node_key}\"] = {latent_name}[\"{selector_name}\"]")
            continue
        if cf_variant == 'A':
            for case_index, config in enumerate(itertools.product([0, 1], repeat=len(parents))):
                cond = ' and '.join(
                    f"state['x{parent}'] == {value}" for parent, value in zip(parents, config)
                )
                keyword = 'if' if case_index == 0 else 'elif'
                selector_name = selector_var_name(selector_name_map, node, config)
                lines.append(f"        {keyword} {cond}:")
                lines.append(f"            state[\"{node_key}\"] = {latent_name}[\"{selector_name}\"]")
        elif cf_variant == 'B':
            tuple_expr = ', '.join(f"state['x{parent}']" for parent in parents)
            if len(parents) == 1:
                tuple_expr = f"({tuple_expr},)"
            else:
                tuple_expr = f"({tuple_expr})"
            map_name = selector_key_map_name(node, vn_variant)
            lines.append(f"        selector_name = {map_name}[{tuple_expr}]")
            lines.append(f"        state[\"{node_key}\"] = {latent_name}[selector_name]")
        else:
            bits = [f"state['x{parent}'] * {2 ** i}" for i, parent in enumerate(reversed(parents))]
            idx_expr = ' + '.join(bits) if bits else '0'
            list_name = selector_list_name(node, vn_variant)
            lines.append(f"        selector_name = {list_name}[{idx_expr}]")
            lines.append(f"        state[\"{node_key}\"] = {latent_name}[selector_name]")
    lines.append('    return state')
    return lines


def render_selector_block(
    scm: SCMBase,
    nodes_used: List[int],
    si_variant: str,
    vn_variant: str,
    selector_name_map: Dict[Tuple[int, Tuple[int, ...]], str],
) -> List[str]:
    lines = []
    for node in nodes_used:
        for config, prob in sorted(scm.cpts[node]['table'].items()):
            selector_name = selector_var_name(selector_name_map, node, config)
            lines.append(render_sampling_line(selector_name, prob, si_variant))
    return lines
