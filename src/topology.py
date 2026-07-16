"""Communication topologies as mixing matrices, plus spectral-gap computation.

Each topology is represented as a symmetric doubly-stochastic mixing matrix W,
where W[i][j] is the weight client i places on client j when averaging with its
neighbours. Symmetry and double-stochasticity guarantee that repeated neighbour
averaging converges to the true average model (Xiao, Boyd & Lall, 2006), the
condition the decentralized-SGD analysis (Lian et al., 2017) assumes.

The spectral gap of W is a scalar summary of how well-connected a topology is
and governs how quickly information propagates through the network.
"""

import numpy as np
import networkx as nx


def metropolis_weights(graph):
    """Return a symmetric doubly-stochastic mixing matrix W from a graph.
    
    Uses Metropolis weights (Xiao, Boyd & Lall, 2006): each edge (i, j) gets
    weight 1 / (1 + max(deg i, deg j)), and each diagonal entry absorbs the
    remainder so every row (and, by symmetry, every column) sums to 1. Only
    local neighbour-degree information is required to compute the weights.
    """
    n = graph.number_of_nodes()
    W = np.zeros((n, n))
    degrees = dict(graph.degree())

    for i, j in graph.edges():
        W[i, j] = W[j, i] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W


def spectral_gap(W):
    """Return the spectral gap 1 - rho of a symmetric doubly-stochastic mixing matrix W.
    
    rho = max(|lambda_2|, |lambda_n|) is the second-largest eigenvalue magnitude;
    the largest eigenvalue is always 1 (the consensus direction) and is excluded.
    A larger gap means faster convergence to consensus.
    """
    eigenvalues = np.sort(np.linalg.eigvalsh(W))    # ascending; eigvalsh for symmetric W
    rho = max(abs(eigenvalues[-2]), abs(eigenvalues[0]))
    return 1.0 - rho


def make_topology(name, num_clients):
    """Return the mixing matrix W for a named topology over num_clients nodes.
    
    Supported topologies span the connectivity spectrum:
        line    - open chain, each node linked to its 1-2 neighbours (least connected)
        ring    - chain closed into a loop
        star    - one hub connected to all spokes; spokes not linked to each other
        hybrid  - two equal rings joined by a single bridge edge (Yuan et al., Fig. 4g)
        mesh    - complete graph, every node linked to every other node (most connected)
    """
    if name == "line":
        graph = nx.path_graph(num_clients)
    elif name == "ring":
        graph = nx.cycle_graph(num_clients)
    elif name == "star":
        # star_graph(k) yields k+1 nodes, so passed num_clients - 1 for num_clients total.
        graph = nx.star_graph(num_clients - 1)
    elif name == "mesh":
        graph = nx.complete_graph(num_clients)
    elif name == "hybrid":
        half = num_clients // 2
        graph = nx.Graph()
        graph.add_nodes_from(range(num_clients))
        # First ring: nodes 0..half-1
        for i in range(half):
            ring_1 = (i + 1) % half
            graph.add_edge(i, ring_1)
        # Second ring: nodes half..num_clients
        for i in range(half, num_clients):
            ring_2 = half + ((i - half + 1) % (num_clients - half))
            graph.add_edge(i, ring_2)
        # Bridge joining the two rings
        graph.add_edge(0, half)
    else:
        raise ValueError(f"Unknown topology: {name}")
    return metropolis_weights(graph)