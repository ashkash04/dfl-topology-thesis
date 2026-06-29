"""Communication topologies as mixing matrices, plus spectral-gap computation.

Each topology is expressed as a symmetric doubly-stochastic mixing matrix W:
W[i][j] is the weight client i places on client j when averaging with neighbours.
The spectral gap of W summarizes how well-connected the topology is and, per the
decentralized-SGD analysis, governs how fast information propagates.
"""

import numpy as np
import networkx as nx


def metropolis_hastings_weights(graph):
    """Return a symmetric doubly-stochastic W from an undirected graph.
    
    Metropolis-Hastings weights: an edge (i, j) gets 1 / (1 + max(deg i, deg j)),
    and each diagonal entry absorbs the remainder so every row sums to 1.
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
    """Return the spectral gap 1 - rho of a symmetric mixing matrix W.
    
    rho = max(|lambda_2|, |lambda_n|) os the second-largest eigenvalue magnitude;
    the largest eigenvalue is always 1 (the consensus direction) and is excluded.
    A larger gap means faster convergence to consensus.
    """
    eigenvalues = np.sort(np.linalg.eigvalsh(W))    # ascending; eigvalsh for symmetric W
    rho = max(abs(eigenvalues[-2]), abs(eigenvalues[0]))
    return 1.0 - rho


def make_topology(name, num_clients):
    """Return the mixing matrix W for a named topology over num_clients nodes."""
    if name == "line":
        graph = nx.path_graph(num_clients)
    elif name == "ring":
        graph = nx.cycle_graph(num_clients)
    elif name == "mesh":
        graph = nx.complete_graph(num_clients)
    elif name == "star":
        # star_graph(k) yields k+1 nodes, so passed num_clients - 1 for num_clients total.
        graph = nx.star_graph(num_clients - 1)
    elif name == "hybrid":
        # Will refine
        graph = nx.cycle_graph(num_clients)
        for i in range(num_clients // 2):
            graph.add_edge(i, i + num_clients // 2)
    else:
        raise ValueError(f"Unknown topology: {name}")
    return metropolis_hastings_weights(graph)