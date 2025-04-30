def generate_G_hyperedge_from_H(H):
    """
    The below code generates a hyperedge-level propagation matrix:
    G = D_e^{-1/2} * H^T * D_v^{-1} * H * D_e^{-1/2}
    H: hypergraph incidence matrix
    And the one we have in hypergraph_utils.py generates a node-level propagation matrix.	
    """
    H = np.array(H)
    n_edge = H.shape[1]

    W = np.ones(n_edge)  # if you want weighted edges
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    invDV = np.asmatrix(np.diag(np.power(DV, -1)))
    invDE_half = np.asmatrix(np.diag(np.power(DE, -0.5)))

    H = np.asmatrix(H)
    HT = H.T

    G_hyperedge = invDE_half * HT * invDV * H * invDE_half
    return G_hyperedge



def generate_G_hyperedge_from_H(H, node_weights=None):
    """
    The below code generates a hyperedge-level propagation matrix with optional node weights:
    G = D_e^{-1/2} * H^T * W_v * D_v^{-1} * H * D_e^{-1/2}
    H: incidence matrix (n_nodes x n_hyperedges)
    node_weights: np.array of shape (n_nodes,) or None
    And the one we have in hypergraph_utils.py generates a node-level propagation matrix.
    """
    H = np.array(H)
    n_nodes, n_edges = H.shape

    # Hyperedge degree
    DE = np.sum(H, axis=0)
    invDE_half = np.asmatrix(np.diag(np.power(DE, -0.5)))

    # Node degree
    if node_weights is None:
        node_weights = np.ones(n_nodes)
    Wv = np.asmatrix(np.diag(node_weights))

    DV = np.sum(H * np.reshape(node_weights, (-1, 1)), axis=1)
    invDV = np.asmatrix(np.diag(np.power(DV, -1)))

    H = np.asmatrix(H)
    HT = H.T

    G_hyperedge = invDE_half * HT * Wv * invDV * H * invDE_half
    return G_hyperedge
