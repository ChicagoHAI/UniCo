"""
scm_mechanism_sampling.py

Implements SCM Mechanism Sampling pipelines based on:
1. CaLM (arXiv:2405): Parametric generation + Data Simulation
2. RLCausal (arXiv:2512): Non-parametric (Tabular) CPT generation + Exact Inference
"""

import networkx as nx
import numpy as np
import pandas as pd
import itertools
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Literal, Union
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination
from data_generation.rlcausal_dag_sampling import CausalGraphSampler


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


class SCMBase(ABC):
    """
    Abstract base class for Structural Causal Model generators.
    """
    def __init__(self, dag: nx.DiGraph, seed: int = None):
        """
        Args:
            dag: A networkx.DiGraph representing the causal structure.
            seed: Random seed for reproducibility.
        """
        self.dag = dag
        # Ensure consistent node ordering for matrix operations
        self.nodes = sorted(list(dag.nodes()))
        self.rng = np.random.RandomState(seed)
        # CPT cache: {node: {'parents': list, 'table': {tuple: float}}}
        # Populated by subclass __init__ before _build_pgmpy_model() is called.
        self.cpts: Dict = {}
        self.pgmpy_model: DiscreteBayesianNetwork = None

    @abstractmethod
    def get_node_cpt(self, node: int) -> Dict[Tuple[int, ...], float]:
        """
        Returns the Conditional Probability Table for a specific node.
        
        Returns:
            Dict where keys are tuples of parent values (e.g., (0, 1))
            and values are P(node=1 | parents).
            Parent values in key must be sorted by parent node ID.
        """
        pass

    def _get_sorted_parents(self, node: int) -> List[int]:
        """Helper to get parents sorted by ID."""
        return sorted(list(self.dag.predecessors(node)))

    def _build_pgmpy_model(self) -> DiscreteBayesianNetwork:
        """
        Build and cache a pgmpy DiscreteBayesianNetwork from the DAG and CPTs.

        Constructs TabularCPDs from get_node_cpt() for each node. The model
        is validated via check_model() and cached for reuse.

        Returns:
            A validated pgmpy DiscreteBayesianNetwork ready for exact inference.
        """
        if self.pgmpy_model is not None:
            return self.pgmpy_model

        edges = list(self.dag.edges())
        model = DiscreteBayesianNetwork(edges)

        # Ensure isolated nodes (no edges) are included in the model
        for node in self.nodes:
            if node not in model.nodes():
                model.add_node(node)

        cpds = []
        for node in self.nodes:
            parents = self._get_sorted_parents(node)
            cpt = self.get_node_cpt(node)

            if not parents:
                p_1 = cpt[()]
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[1.0 - p_1], [p_1]],
                )
            else:
                # Column ordering: itertools.product([0,1], repeat=len(parents))
                # with parents sorted by node ID. This matches pgmpy's convention
                # where the last evidence variable cycles fastest.
                configs = list(itertools.product([0, 1], repeat=len(parents)))
                p_1_list = [cpt[config] for config in configs]
                p_0_list = [1.0 - p for p in p_1_list]

                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[p_0_list, p_1_list],
                    evidence=parents,
                    evidence_card=[2] * len(parents),
                )

            cpds.append(cpd)

        model.add_cpds(*cpds)

        if not model.check_model():
            raise ValueError(
                "pgmpy model validation failed: CPDs are inconsistent "
                "with DAG structure"
            )

        self.pgmpy_model = model
        return model

    def query(
        self,
        variables: List[int],
        evidence: Dict[int, int] = None,
        show_progress: bool = False,
    ) -> DiscreteFactor:
        """
        Compute exact probabilities via Variable Elimination.

        Args:
            variables: List of node IDs to compute probability for.
            evidence: Dict mapping observed node IDs to their observed values
                (0 or 1). None means no evidence (marginal query).
            show_progress: Whether to display pgmpy's elimination progress bar.

        Returns:
            A pgmpy DiscreteFactor representing P(variables | evidence).
            Access probabilities via factor.values numpy array, where indices
            correspond to variable states in the order of factor.variables.

        Example:
            >>> factor = scm.query([3], evidence={0: 1})
            >>> p_3_eq_1 = float(factor.values[1])  # P(node 3 = 1 | node 0 = 1)
        """
        ve = VariableElimination(self.pgmpy_model)
        result = ve.query(
            variables=variables,
            evidence=evidence,
            joint=True,
            show_progress=show_progress,
        )
        return result


class CaLMSCM(SCMBase):
    """
    Implements the CaLM SCM generation strategy (arXiv:2405 Sec 4.2.2).
    
    Logic:
    1. Mechanisms are Single-Layer Perceptrons (Sigmoid of linear sum).
    2. Generates N samples (default 50k) by simulation.
    3. CPTs and Queries are approximated empirically from the generated data.
    """
    def __init__(self, dag: nx.DiGraph, num_samples: int = 50000, seed: int = None):
        super().__init__(dag, seed)
        self.num_samples = num_samples
        
        # Structure: {node: {'bias': float, 'weights': {parent: float}, 'parents': list}}
        self.mechanisms = {} 
        self.data: pd.DataFrame = None

        # Execute Pipeline
        self._sample_parameters()
        self._generate_data()
        self._derive_cpts()
        self._build_pgmpy_model()

    def _sample_parameters(self):
        """
        Samples bias and weights from Normal(0, 1).
        """
        for node in self.nodes:
            parents = self._get_sorted_parents(node)
            
            # Sample parameters
            # "Both b_X and w_X are randomly generated"
            bias = self.rng.normal(0, 1)
            weights = {p: self.rng.normal(0, 1) for p in parents}
            
            self.mechanisms[node] = {
                'bias': bias,
                'weights': weights,
                'parents': parents
            }

    def _generate_data(self):
        """
        Generates data by iterating through the graph in topological order.
        """
        # Data matrix: Rows=Samples, Cols=Nodes
        data_matrix = np.zeros((self.num_samples, len(self.nodes)), dtype=int)
        
        # Mapping for fast column access
        node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        
        # Topological sort ensures we compute parents before children
        try:
            topo_order = list(nx.topological_sort(self.dag))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph is not a DAG! Cannot perform topological generation.")

        for node in topo_order:
            mech = self.mechanisms[node]
            col_idx = node_to_idx[node]
            
            # Start with bias
            # Shape: (num_samples,)
            logits = np.full(self.num_samples, mech['bias'])
            
            # Add weighted contribution from parents
            for parent in mech['parents']:
                p_idx = node_to_idx[parent]
                weight = mech['weights'][parent]
                # Vectorized addition: weight * column of parent values
                logits += weight * data_matrix[:, p_idx]
            
            # Compute Prob(Node=1)
            probs = sigmoid(logits)
            
            # Sample Bernoullis
            # Generate random uniform [0,1] and compare against probs
            random_draws = self.rng.uniform(0, 1, size=self.num_samples)
            data_matrix[:, col_idx] = (random_draws < probs).astype(int)
            
        self.data = pd.DataFrame(data_matrix, columns=self.nodes)

    def _derive_cpts(self):
        """
        Derive and cache empirical CPTs for all nodes from the simulated data.
        """
        for node in self.nodes:
            parents = self.mechanisms[node]['parents']

            # Case 1: Root Node (No parents)
            if not parents:
                # Marginal probability of being 1
                prob_1 = self.data[node].mean()
                table = {(): float(prob_1)}
            # Case 2: Dependent Node
            # Group data by parent values
            else:
                grouped = self.data.groupby(parents)[node].mean()

                def _groupby_key(config, _parents=parents):
                    return config[0] if len(_parents) == 1 else config

                table = {}
                # Iterate through all possible parent configurations to ensure completeness
                # (Though with 50k samples and low in-degree, most should be present)
                for config in itertools.product([0, 1], repeat=len(parents)):
                    # config is a tuple like (0, 1, 0)
                    key = _groupby_key(config)
                    if key in grouped.index:
                        table[config] = float(grouped.loc[key])
                    else:
                        # Fallback if configuration never appeared in samples
                        table[config] = 0.5

            self.cpts[node] = {'parents': parents, 'table': table}

    def get_node_cpt(self, node: int) -> Dict[Tuple[int, ...], float]:
        """
        Returns the empirically-derived CPT cached at construction time.
        """
        return self.cpts[node]['table']

    def query(
        self,
        variables: List[int],
        evidence: Dict[int, int] = None,
        mode: Literal["approximate", "exact"] = "exact",
        show_progress: bool = False,
    ) -> DiscreteFactor:
        """
        Compute P(variables | evidence).

        Args:
            variables: List of node IDs to query.
            evidence: Dict mapping observed node IDs to their values (0 or 1).
                None means no evidence (marginal query).
            mode: "exact" (default) uses pgmpy Variable Elimination on CPTs
                  derived from the simulated data.
                  "approximate" uses empirical counts from the simulated data.
            show_progress: Passed to pgmpy VE (exact mode only).

        Returns:
            A pgmpy DiscreteFactor representing P(variables | evidence).
        """
        if mode == "exact":
            return super().query(variables, evidence, show_progress)

        elif mode == "approximate":
            # Filter data by evidence
            if evidence:
                mask = pd.Series(True, index=self.data.index)
                for node, val in evidence.items():
                    mask &= (self.data[node] == val)
                filtered_data = self.data[mask]
            else:
                filtered_data = self.data

            num_filtered = len(filtered_data)

            # Compute joint frequency distribution over query variables
            var_configs = list(itertools.product([0, 1], repeat=len(variables)))
            counts = []
            for config in var_configs:
                config_mask = pd.Series(True, index=filtered_data.index)
                for var, val in zip(variables, config):
                    config_mask &= (filtered_data[var] == val)
                counts.append(config_mask.sum())

            total = sum(counts)
            if total == 0:
                # Edge case: evidence combination never appeared in samples
                probs = [1.0 / len(var_configs)] * len(var_configs)
            else:
                probs = [c / total for c in counts]

            return DiscreteFactor(variables, [2] * len(variables), probs)

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'approximate' or 'exact'.")


class RLCausalSCM(SCMBase):
    """
    Implements the RLCausal SCM generation strategy (arXiv:2512).
    """
    def __init__(self, dag: nx.DiGraph, seed: int = None, decimal_places: int = 2):
        super().__init__(dag, seed)
        self.decimal_places = decimal_places
        self._sample_cpts()
        self._build_pgmpy_model()

    def _sample_cpts(self):
        """
        Directly samples probability values for every parent configuration.
        """
        for node in self.nodes:
            parents = self._get_sorted_parents(node)
            
            # Generate Truth Table (Cartesian product of {0,1})
            # e.g. for 2 parents: [(0,0), (0,1), (1,0), (1,1)]
            # If 0 parents: [()]
            parent_configs = list(itertools.product([0, 1], repeat=len(parents)))
            
            node_cpt_table = {}
            for config in parent_configs:
                # RLCausal: "sample a binary distribution q_v uniformly from the simplex"
                # For binary outcome, this simplifies to p ~ Uniform(0, 1)
                # Round at sampling time to match display precision
                while True:
                    p_val = round(self.rng.uniform(0, 1), self.decimal_places)
                    if 0.0 < p_val < 1.0:
                        break
                node_cpt_table[config] = p_val
                
            self.cpts[node] = {
                'parents': parents,
                'table': node_cpt_table
            }

    def get_node_cpt(self, node: int) -> Dict[Tuple[int, ...], float]:
        """
        Returns the EXACT CPT stored in memory.
        """
        return self.cpts[node]['table']

    @classmethod
    def from_cpts(cls, dag: nx.DiGraph, cpts: Dict, decimal_places: int = 2):
        """Construct an RLCausalSCM from pre-existing CPTs without re-sampling.

        Used by downstream pipelines (e.g., question assembly) to reconstruct
        an SCM from serialized triplet data.

        Args:
            dag: A networkx.DiGraph representing the causal structure.
            cpts: Pre-computed CPT dict with structure
                  {node: {'parents': list, 'table': {tuple: float}}}.
            decimal_places: The decimal precision used when the CPTs were sampled.
        """
        instance = cls.__new__(cls)
        instance.dag = dag
        instance.nodes = sorted(list(dag.nodes()))
        instance.rng = None
        instance.decimal_places = decimal_places
        instance.cpts = cpts
        instance.pgmpy_model = None
        instance._build_pgmpy_model()
        return instance

    # Inherits query() from SCMBase — exact inference via pgmpy VE on stored CPTs.


# ==========================================
# TEST SUITE
# ==========================================
if __name__ == "__main__":
    print("=== Testing SCM Sampling Pipeline ===\n")

    # ------------------------------------------------------------------
    # 1. Setup: Generate a DAG
    # ------------------------------------------------------------------
    print("1. Generating DAG using CausalGraphSampler...")
    dag_sampler = CausalGraphSampler(num_nodes=5)
    dag = dag_sampler.sample()
    print(f"   DAG Generated. Nodes: {dag.nodes()}")
    print(f"   Edges: {dag.edges()}\n")

    # ------------------------------------------------------------------
    # 2. Test CaLMSCM
    # ------------------------------------------------------------------
    print("2. Testing CaLM SCM (Parametric + Simulation)...")
    calm_scm = CaLMSCM(dag, num_samples=50000, seed=42)

    # 2a. Data generation
    print(f"   Data Shape: {calm_scm.data.shape} (Expected (50000, 5))")
    assert calm_scm.data.shape == (50000, 5)
    assert set(calm_scm.data.columns) == set(range(5))
    assert calm_scm.data.isin([0, 1]).all().all(), "Data contains non-binary values"
    print("   -> Data structure valid.")

    # 2a2. Cached CPTs and pgmpy_model
    print("   Checking self.cpts and self.pgmpy_model...")
    assert isinstance(calm_scm.cpts, dict)
    assert set(calm_scm.cpts.keys()) == set(range(5))
    for node_id in range(5):
        assert 'parents' in calm_scm.cpts[node_id]
        assert 'table' in calm_scm.cpts[node_id]
    assert isinstance(calm_scm.pgmpy_model, DiscreteBayesianNetwork)
    assert calm_scm.pgmpy_model.check_model()
    print("   -> self.cpts and self.pgmpy_model valid.")

    # 2b. CPT extraction
    print("   Checking CPT extraction for a sample node...")
    test_node = list(dag.nodes)[-1]
    cpt = calm_scm.get_node_cpt(test_node)
    parents = calm_scm.mechanisms[test_node]['parents']
    expected_rows = 2 ** len(parents)
    print(f"   Node {test_node} (Parents: {parents}) has {len(cpt)} CPT rows.")
    assert len(cpt) == expected_rows
    for k, v in cpt.items():
        assert 0.0 <= v <= 1.0
    print("   -> CPT structure valid.")

    # 2c. Approximate marginal query
    print("   Testing approximate marginal query...")
    approx_marginal_factor = calm_scm.query([test_node], mode="approximate")
    approx_marginal_p1 = float(approx_marginal_factor.values[1])
    print(f"   P(Node_{test_node}=1) approx = {approx_marginal_p1:.4f}")
    assert 0.0 <= approx_marginal_p1 <= 1.0
    assert isinstance(approx_marginal_factor, DiscreteFactor)
    print("   -> Approximate marginal query passed.")

    # 2d. Approximate conditional query (if parents exist)
    if parents:
        print("   Testing approximate conditional query...")
        evidence = {parents[0]: 1}
        approx_cond_factor = calm_scm.query([test_node], evidence, mode="approximate")
        approx_cond_p1 = float(approx_cond_factor.values[1])
        print(f"   P(Node_{test_node}=1 | Node_{parents[0]}=1) approx = {approx_cond_p1:.4f}")
        assert 0.0 <= approx_cond_p1 <= 1.0
        print("   -> Approximate conditional query passed.")

    # 2e. Exact marginal query
    print("   Testing exact marginal query (pgmpy VE)...")
    exact_marginal_factor = calm_scm.query([test_node], mode="exact")
    exact_marginal_p1 = float(exact_marginal_factor.values[1])
    print(f"   P(Node_{test_node}=1) exact = {exact_marginal_p1:.4f}")
    assert 0.0 <= exact_marginal_p1 <= 1.0
    assert isinstance(exact_marginal_factor, DiscreteFactor)
    print("   -> Exact marginal query passed.")

    # 2f. Exact conditional query (if parents exist)
    if parents:
        print("   Testing exact conditional query...")
        evidence = {parents[0]: 1}
        exact_cond_factor = calm_scm.query([test_node], evidence, mode="exact")
        exact_cond_p1 = float(exact_cond_factor.values[1])
        print(f"   P(Node_{test_node}=1 | Node_{parents[0]}=1) exact = {exact_cond_p1:.4f}")
        assert 0.0 <= exact_cond_p1 <= 1.0
        print("   -> Exact conditional query passed.")

    # 2g. Exact vs approximate consistency
    print("   Testing exact vs approximate consistency...")
    atol = 0.05  # tolerance for 50k samples
    diff = abs(exact_marginal_p1 - approx_marginal_p1)
    print(f"   |exact - approx| marginal = {diff:.6f} (tolerance {atol})")
    assert diff < atol, f"Exact/approx marginal mismatch: {exact_marginal_p1} vs {approx_marginal_p1}"
    if parents:
        diff_cond = abs(exact_cond_p1 - approx_cond_p1)
        print(f"   |exact - approx| conditional = {diff_cond:.6f} (tolerance {atol})")
        assert diff_cond < atol, f"Exact/approx conditional mismatch: {exact_cond_p1} vs {approx_cond_p1}"
    print("   -> Exact vs approximate consistency passed.")

    # 2h. Multi-variable joint query
    print("   Testing multi-variable joint query...")
    all_nodes = sorted(dag.nodes())
    if len(all_nodes) >= 2:
        joint_vars = all_nodes[:2]
        joint_factor = calm_scm.query(joint_vars, mode="exact")
        joint_sum = float(joint_factor.values.sum())
        print(f"   Joint query on {joint_vars}: sum = {joint_sum:.6f} (expected 1.0)")
        assert abs(joint_sum - 1.0) < 1e-8, f"Joint distribution does not sum to 1: {joint_sum}"
        assert joint_factor.values.shape == (2, 2), f"Unexpected shape: {joint_factor.values.shape}"
        print("   -> Multi-variable joint query passed.")

    # 2i. Invalid mode raises ValueError
    print("   Testing invalid mode raises ValueError...")
    try:
        calm_scm.query([test_node], mode="invalid")
        assert False, "Did not raise ValueError"
    except ValueError:
        print("   -> ValueError raised as expected.")

    # 2j. Regression test for single-parent groupby indexing
    print("   Testing single-parent CPT extraction (groupby index)...")
    single_parent_dag = nx.DiGraph()
    single_parent_dag.add_nodes_from([0, 1])
    single_parent_dag.add_edge(0, 1)
    single_parent_scm = CaLMSCM(single_parent_dag, num_samples=5000, seed=123)
    single_parent_cpt = single_parent_scm.get_node_cpt(1)
    assert set(single_parent_cpt.keys()) == {(0,), (1,)}
    print("   -> Single-parent CPT extraction passed.")

    print("   -> All CaLM tests passed.\n")

    # ------------------------------------------------------------------
    # 3. Test RLCausalSCM
    # ------------------------------------------------------------------
    print("3. Testing RLCausal SCM...")
    rl_scm = RLCausalSCM(dag, seed=42)

    # 3a. CPT structure
    print("   Checking CPT correctness...")
    for node in dag.nodes():
        cpt = rl_scm.get_node_cpt(node)
        node_parents = sorted(list(dag.predecessors(node)))
        assert len(cpt) == 2 ** len(node_parents)
        for key in cpt.keys():
            assert len(key) == len(node_parents)
        for prob in cpt.values():
            assert 0.0 <= prob <= 1.0
    print("   -> All CPTs structurally correct and bounded [0, 1].")

    # 3a2. pgmpy_model available
    print("   Checking self.pgmpy_model...")
    assert isinstance(rl_scm.pgmpy_model, DiscreteBayesianNetwork)
    assert rl_scm.pgmpy_model.check_model()
    print("   -> self.pgmpy_model valid.")

    # 3b. Marginal query (no evidence)
    print("   Testing marginal query (exact via VE)...")
    rl_marginal_factor = rl_scm.query([test_node])
    rl_marginal_p1 = float(rl_marginal_factor.values[1])
    print(f"   P(Node_{test_node}=1) = {rl_marginal_p1:.6f}")
    assert 0.0 <= rl_marginal_p1 <= 1.0
    assert isinstance(rl_marginal_factor, DiscreteFactor)
    print("   -> Marginal query passed.")

    # 3c. Conditional query (with evidence, if parents exist)
    rl_test_parents = sorted(list(dag.predecessors(test_node)))
    if rl_test_parents:
        print("   Testing conditional query...")
        rl_evidence = {rl_test_parents[0]: 1}
        rl_cond_factor = rl_scm.query([test_node], rl_evidence)
        rl_cond_p1 = float(rl_cond_factor.values[1])
        print(f"   P(Node_{test_node}=1 | Node_{rl_test_parents[0]}=1) = {rl_cond_p1:.6f}")
        assert 0.0 <= rl_cond_p1 <= 1.0
        print("   -> Conditional query passed.")

    # 3d. Root node marginal matches CPT entry exactly
    print("   Testing root node marginal matches CPT exactly...")
    root_nodes = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    for root in root_nodes:
        root_cpt = rl_scm.get_node_cpt(root)
        root_factor = rl_scm.query([root])
        root_p1_ve = float(root_factor.values[1])
        root_p1_cpt = root_cpt[()]
        assert abs(root_p1_ve - root_p1_cpt) < 1e-10, \
            f"Root {root}: VE={root_p1_ve} vs CPT={root_p1_cpt}"
        print(f"   Root node {root}: VE P(=1) = {root_p1_ve:.6f}, CPT = {root_p1_cpt:.6f} -> match")
    print("   -> Root node marginal test passed.")

    # 3e. Manual computation on a 3-node chain: verify P(2=1|0=1)
    print("   Testing manual computation on 3-node chain...")
    chain_dag = nx.DiGraph()
    chain_dag.add_edges_from([(0, 1), (1, 2)])
    chain_scm = RLCausalSCM(chain_dag, seed=99)

    cpt_0 = chain_scm.get_node_cpt(0)  # {(): p0}
    cpt_1 = chain_scm.get_node_cpt(1)  # {(0,): p10, (1,): p11}
    cpt_2 = chain_scm.get_node_cpt(2)  # {(0,): p20, (1,): p21}

    # P(2=1|0=1) = P(2=1|1=0)*P(1=0|0=1) + P(2=1|1=1)*P(1=1|0=1)
    p_1_eq_1_given_0_eq_1 = cpt_1[(1,)]
    p_1_eq_0_given_0_eq_1 = 1.0 - p_1_eq_1_given_0_eq_1
    expected_p2_given_0 = (
        cpt_2[(0,)] * p_1_eq_0_given_0_eq_1
        + cpt_2[(1,)] * p_1_eq_1_given_0_eq_1
    )

    chain_factor = chain_scm.query([2], evidence={0: 1})
    ve_p2_given_0 = float(chain_factor.values[1])
    diff_chain = abs(ve_p2_given_0 - expected_p2_given_0)
    print(f"   P(2=1|0=1): VE = {ve_p2_given_0:.10f}, manual = {expected_p2_given_0:.10f}, diff = {diff_chain:.2e}")
    assert diff_chain < 1e-10, f"Chain manual vs VE mismatch: {ve_p2_given_0} vs {expected_p2_given_0}"
    print("   -> Manual computation on chain passed.")

    # 3f. Conditional on direct parent matches CPT exactly (2-node chain)
    print("   Testing direct parent conditional matches CPT...")
    two_node_dag = nx.DiGraph()
    two_node_dag.add_edge(0, 1)
    two_node_scm = RLCausalSCM(two_node_dag, seed=77)
    cpt_child = two_node_scm.get_node_cpt(1)

    for parent_val in [0, 1]:
        factor = two_node_scm.query([1], evidence={0: parent_val})
        ve_p1 = float(factor.values[1])
        cpt_p1 = cpt_child[(parent_val,)]
        assert abs(ve_p1 - cpt_p1) < 1e-10, \
            f"Parent={parent_val}: VE={ve_p1} vs CPT={cpt_p1}"
        print(f"   P(1=1|0={parent_val}): VE = {ve_p1:.6f}, CPT = {cpt_p1:.6f} -> match")
    print("   -> Direct parent conditional test passed.")

    # 3g. Multi-variable joint query
    print("   Testing multi-variable joint query...")
    if len(all_nodes) >= 2:
        rl_joint_vars = all_nodes[:2]
        rl_joint_factor = rl_scm.query(rl_joint_vars)
        rl_joint_sum = float(rl_joint_factor.values.sum())
        print(f"   Joint query on {rl_joint_vars}: sum = {rl_joint_sum:.6f} (expected 1.0)")
        assert abs(rl_joint_sum - 1.0) < 1e-8, f"Joint distribution does not sum to 1: {rl_joint_sum}"
        assert rl_joint_factor.values.shape == (2, 2)
        print("   -> Multi-variable joint query passed.")

    print("   -> All RLCausal tests passed.\n")

    # ------------------------------------------------------------------
    # 4. Edge case: single-node DAG (isolated node, no edges)
    # ------------------------------------------------------------------
    print("4. Testing edge case: single-node DAG...")
    single_node_dag = nx.DiGraph()
    single_node_dag.add_node(0)
    single_node_rl = RLCausalSCM(single_node_dag, seed=55)
    single_factor = single_node_rl.query([0])
    single_p1 = float(single_factor.values[1])
    single_cpt_p1 = single_node_rl.get_node_cpt(0)[()]
    assert abs(single_p1 - single_cpt_p1) < 1e-10
    print(f"   Single-node DAG: P(0=1) = {single_p1:.6f}, CPT = {single_cpt_p1:.6f} -> match")
    print("   -> Single-node DAG test passed.\n")

    print("=== All Tests Passed Successfully ===")
