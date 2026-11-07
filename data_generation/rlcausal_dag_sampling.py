import networkx as nx
import random
import itertools
from typing import List

class CausalGraphSampler:
    """
    A class to sample random Causal DAGs based on the pipeline described in 
    RLCausal (arXiv:2512.20760) and Lampinen et al. (arXiv:2305.16183).
    
    Pipeline:
    1. Define topological order.
    2. Determine number of independent (root) nodes.
    3. Iteratively add children with 1 or 2 parents from existing nodes.
    4. Constraint Check: Weak Connectivity.
    5. Constraint Check: Uniqueness (Non-Isomorphism).
    6. Post-processing: Random node permutation (Anonymization).
    """

    def __init__(self, num_nodes: int = 10, max_attempts: int = 1000, seed: int = None):
        """
        Args:
            num_nodes (int): Total number of variables in the graph.
            max_attempts (int): Max rejection sampling attempts before failure.
            seed (int): Random seed for reproducibility. None uses unseeded RNG.
        """
        self.num_nodes = num_nodes
        self.max_attempts = max_attempts
        self.rng = random.Random(seed)
        # Stores previously sampled graphs to ensure uniqueness (isomorphism check)
        self.history: List[nx.DiGraph] = []

    def reset_history(self):
        """Clears the history of generated graphs."""
        self.history = []

    def _generate_candidate_topology(self) -> nx.DiGraph:
        """
        Generates a candidate DAG using the topological construction method.
        Nodes 0..N-1 are created in topological order.
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))

        # Step 1: Sample number of independent nodes (roots)
        # "samples a random number of independent nodes between 1 and N"
        n_indep = self.rng.randint(1, self.num_nodes)

        # Step 2: Iteratively add edges for the remaining dependent nodes
        # Nodes 0 to n_indep-1 are roots (no parents).
        # We iterate through the rest.
        for current_node in range(n_indep, self.num_nodes):
            # "Each added node has either one or two parents"
            num_desired_parents = self.rng.choice([1, 2])

            # "Chosen uniformly from the existing nodes"
            # Existing nodes are 0 to current_node-1
            potential_parents = list(range(0, current_node))

            # Safety check: can't sample 2 parents if only 1 exists
            num_actual_parents = min(num_desired_parents, len(potential_parents))
            
            if num_actual_parents > 0:
                selected_parents = self.rng.sample(potential_parents, num_actual_parents)
                for parent in selected_parents:
                    G.add_edge(parent, current_node)

        return G

    def _check_connectivity(self, G: nx.DiGraph) -> bool:
        """
        Constraint 1: The graph must be weakly connected.
        (All nodes must be reachable if edges were undirected).
        """
        # If N > 1 and there are no edges, it's not connected.
        # nx.is_weakly_connected handles single nodes as connected.
        return nx.is_weakly_connected(G)

    def _check_isomorphism_uniqueness(self, G: nx.DiGraph) -> bool:
        """
        Constraint 2: The graph must not be isomorphic to any previously 
        sampled graph in self.history.
        """
        for history_G in self.history:
            if nx.is_isomorphic(G, history_G):
                return False  # Duplicate found
        return True

    def _permute_labels(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Applies a random permutation to node labels to hide the 
        generative topological order (i.e., ensure 0 isn't always a root).
        """
        old_labels = list(G.nodes())
        new_labels = list(G.nodes())
        self.rng.shuffle(new_labels)
        
        mapping = dict(zip(old_labels, new_labels))
        return nx.relabel_nodes(G, mapping)

    def sample(self) -> nx.DiGraph:
        """
        Main driver for sampling a single valid DAG.
        
        Returns:
            nx.DiGraph: A valid, unique, permuted causal graph.
        
        Raises:
            RuntimeError: If a valid graph cannot be found within max_attempts.
        """
        for _ in range(self.max_attempts):
            # 1. Generate Topology
            candidate_G = self._generate_candidate_topology()

            # 2. Check Connectivity
            if not self._check_connectivity(candidate_G):
                continue

            # 3. Check Uniqueness (Isomorphism)
            if not self._check_isomorphism_uniqueness(candidate_G):
                continue

            # 4. Valid found: Permute labels
            final_G = self._permute_labels(candidate_G)

            # 5. Add to history
            self.history.append(final_G)
            
            return final_G

        raise RuntimeError(
            f"Failed to generate a valid unique DAG after {self.max_attempts} attempts. "
            "Try increasing node count or reducing constraints."
        )


if __name__ == "__main__":
    print("=== Starting Holistic Verification of CausalGraphSampler ===")
    
    # Configuration for tests
    N = 10
    NUM_SAMPLES = 20
    sampler = CausalGraphSampler(num_nodes=N)
    
    print(f"Generating {NUM_SAMPLES} samples for verification...")
    try:
        graphs = [sampler.sample() for _ in range(NUM_SAMPLES)]
    except RuntimeError as e:
        print(f"Error during sampling: {e}")
        exit(1)

    print("Sampling complete. Running checks...\n")

    # --- Test 1: Basic Structural Validity ---
    print(f"[Test 1] Checking DAG properties (N={N})...")
    for i, G in enumerate(graphs):
        assert isinstance(G, nx.DiGraph), f"Sample {i} is not a DiGraph"
        assert len(G.nodes) == N, f"Sample {i} has {len(G.nodes)} nodes, expected {N}"
        assert nx.is_directed_acyclic_graph(G), f"Sample {i} contains cycles!"
    print("-> PASSED: All samples are DAGs of correct size.\n")

    # --- Test 2: Connectivity Constraint ---
    print("[Test 2] Checking Weak Connectivity...")
    for i, G in enumerate(graphs):
        assert nx.is_weakly_connected(G), f"Sample {i} is not weakly connected"
    print("-> PASSED: All samples are weakly connected.\n")

    # --- Test 3: Uniqueness (Isomorphism) Constraint ---
    print("[Test 3] Checking Pairwise Isomorphism (Uniqueness)...")
    isomorphism_conflicts = 0
    combinations = list(itertools.combinations(enumerate(graphs), 2))
    
    for (idx1, g1), (idx2, g2) in combinations:
        if nx.is_isomorphic(g1, g2):
            print(f"  Conflict found between Sample {idx1} and Sample {idx2}")
            isomorphism_conflicts += 1
    
    if isomorphism_conflicts == 0:
        print(f"-> PASSED: All {NUM_SAMPLES} samples are structurally unique.")
    else:
        print(f"-> FAILED: Found {isomorphism_conflicts} isomorphic pairs.")
    print("")

    # --- Test 4: In-Degree Logic Verification ---
    # RLCausal logic dictates: Nodes are either roots (in-degree 0) 
    # or have 1 or 2 parents. No node should have > 2 parents.
    print("[Test 4] Checking Parent Count Constraints...")
    for i, G in enumerate(graphs):
        in_degrees = [d for n, d in G.in_degree()]
        
        # Check max parents
        max_parents = max(in_degrees)
        assert max_parents <= 2, f"Sample {i} has a node with {max_parents} parents (Max allowed: 2)"
        
        # Check that independent nodes exist
        num_roots = in_degrees.count(0)
        assert num_roots > 0, f"Sample {i} has no independent nodes (roots)"
        
    print("-> PASSED: All nodes have 0, 1, or 2 parents.\n")

    # --- Test 5: Permutation Verification (Probabilistic) ---
    # In the raw topological generation, if edge u->v exists, then u < v always.
    # After permutation, there should be edges where u > v.
    print("[Test 5] Checking Permutation (Anonymization)...")
    edges_violating_order = 0
    total_edges = 0
    
    for G in graphs:
        for u, v in G.edges():
            total_edges += 1
            if u > v:
                edges_violating_order += 1
    
    print(f"  Total Edges: {total_edges}")
    print(f"  Edges pointing 'backwards' (u > v) relative to ID: {edges_violating_order}")
    
    # In a random permutation, roughly 50% of edges might point u > v. 
    # We just need to ensure it's not 0 (which would imply no permutation happened).
    assert edges_violating_order > 0, "No edges point from High ID to Low ID. Permutation likely failed."
    print("-> PASSED: Node labels appear successfully permuted.\n")

    print("=== All Holistic Checks Passed Successfully ===")
