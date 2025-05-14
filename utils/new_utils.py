import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import warnings


def _generate_W_sparse(H, E_weights):
    """
    Calculate W matrix (sparse implementation).
    :param H: hypergraph incidence matrix H (V x E) - scipy.sparse.csr_matrix
    :param E_weights: weights for each hyperedge (1 x E)
    :return: W - scipy.sparse.csr_matrix
    """
    # Convert H to CSR if it's not already
    if not sp.isspmatrix_csr(H):
        H = sp.csr_matrix(H)

    # Create a diagonal matrix from E_weights
    E_diag = sp.diags(E_weights)

    # Multiply H by the diagonal matrix of edge weights
    W = H.dot(E_diag)

    return W


def generate_P_sparse(H, R, E_weights):
    """
    Calculate transition matrix P (sparse implementation).
    :param H: hypergraph incidence matrix H (V x E) - scipy.sparse.csr_matrix
    :param R: edge-dependent vertex weights for each vertex (E x V) - scipy.sparse.csr_matrix
    :param E_weights: weights for each hyperedge (1 x E)
    :return: P (V x V) - scipy.sparse.csr_matrix
    """
    # Convert to sparse if not already
    if not sp.isspmatrix_csr(H):
        H = sp.csr_matrix(H)
    if not sp.isspmatrix_csr(R):
        R = sp.csr_matrix(R)

    # Compute vertex degrees (sum of each row of H)
    d_v = np.array(H.sum(axis=1)).flatten()
    # Compute edge degrees (sum of each column of H)
    d_e = np.array(H.sum(axis=0)).flatten()

    # Create diagonal matrices
    D_V_inv = sp.diags(1.0 / np.maximum(d_v, 1e-10))  # Avoid division by zero
    D_E_inv = sp.diags(1.0 / np.maximum(d_e, 1e-10))  # Avoid division by zero

    # Generate W
    W = _generate_W_sparse(H, E_weights)

    # Calculate P = D_V^-1 * W * D_E^-1 * R
    P = D_V_inv.dot(W.dot(D_E_inv.dot(R)))

    return P


def generate_Pi_from_P_sparse(P, max_iter=100, tol=1e-6):
    """
    Calculate stationary distribution Pi from transition matrix P using power iteration.
    This is more efficient than eigendecomposition for large sparse matrices.

    :param P: transition matrix P (V x V) - scipy.sparse.csr_matrix
    :param max_iter: maximum number of iterations for power iteration
    :param tol: convergence tolerance
    :return: Pi (V x V) diagonal matrix - scipy.sparse.csr_matrix
    """
    n = P.shape[0]

    # Initialize with uniform distribution
    pi = np.ones(n) / n

    # Power iteration
    for _ in range(max_iter):
        pi_next = P.T.dot(pi)

        # Normalize
        pi_next = pi_next / np.sum(pi_next)

        # Check convergence
        if np.linalg.norm(pi_next - pi) < tol:
            pi = pi_next
            break

        pi = pi_next

    # Ensure non-negative values and normalize
    pi = np.abs(pi)
    pi = pi / np.sum(pi)

    # Create diagonal matrix
    Pi = sp.diags(pi)

    return Pi


def generate_Pi_from_A_sparse(H, R, E_weights, max_iter=100, tol=1e-6):
    """
    Calculate stationary distribution Pi from square matrix A using power iteration.

    :param H: hypergraph incidence matrix H (V x E) - scipy.sparse.csr_matrix
    :param R: edge-dependent vertex weights for each vertex (E x V) - scipy.sparse.csr_matrix
    :param E_weights: weights for each hyperedge (1 x E)
    :return: Pi (V x V) - scipy.sparse.csr_matrix
    """
    # Convert to sparse if not already
    if not sp.isspmatrix_csr(H):
        H = sp.csr_matrix(H)
    if not sp.isspmatrix_csr(R):
        R = sp.csr_matrix(R)

    # Compute vertex degrees (sum of each row of H)
    d_v = np.array(H.sum(axis=1)).flatten()

    # Create diagonal matrix D_V^-1
    D_V_inv = sp.diags(1.0 / np.maximum(d_v, 1e-10))  # Avoid division by zero

    # Generate W
    W = _generate_W_sparse(H, E_weights)

    # Calculate A = W^T * D_V^-1 * R^T
    A = W.T.dot(D_V_inv.dot(R.T))

    # Initialize with uniform distribution
    m = A.shape[0]  # Number of edges
    rho = np.ones(m) / m

    # Power iteration to find the eigenvector with eigenvalue closest to 1
    for _ in range(max_iter):
        rho_next = A.dot(rho)

        # Normalize
        rho_next = rho_next / np.sum(np.abs(rho_next))

        # Check convergence
        if np.linalg.norm(rho_next - rho) < tol:
            rho = rho_next
            break

        rho = rho_next

    # Calculate pi_v values using the formula from the paper
    # pi_v = diagonal((W * rho) @ R)
    W_rho = W.multiply(sp.csr_matrix(rho.reshape(-1, 1)))
    pi_v = np.array((W_rho.dot(R)).diagonal()).flatten()

    # Ensure non-negative values and normalize
    pi_v = np.abs(pi_v)
    pi_v = pi_v / np.sum(pi_v)

    # Create diagonal matrix
    Pi = sp.diags(pi_v)

    return Pi


def generate_G_from_H_sparse(H, R, E_weights, Pi_version='from_P'):
    """
    Calculate spectral convolution G from hypergraph incidence matrix H (sparse implementation).

    :param H: hypergraph incidence matrix H (V x E)
    :param R: edge-dependent vertex weights for each vertex (E x V)
    :param E_weights: weights for each hyperedge (1 x E)
    :param Pi_version: which method to use for calculating Pi ('from_P' or 'from_A')
    :return: G (V x V) - scipy.sparse.csr_matrix
    """
    # Convert to sparse if not already
    if not sp.isspmatrix_csr(H):
        H = sp.csr_matrix(H)
    if not sp.isspmatrix_csr(R):
        R = sp.csr_matrix(R)

    # Generate P (transition matrix)
    P = generate_P_sparse(H, R, E_weights)

    # Generate Pi (stationary distribution)
    if Pi_version == 'from_P':
        Pi = generate_Pi_from_P_sparse(P)
    elif Pi_version == 'from_A':
        Pi = generate_Pi_from_A_sparse(H, R, E_weights)
    else:
        raise ValueError(f"Unknown Pi_version: {Pi_version}")

    # Calculate G = Pi - (Pi @ P + P.T @ Pi) / 2
    G = Pi - (Pi.dot(P) + P.T.dot(Pi)) / 2

    return G


def generate_G_from_H(H, R, E_weights, Pi_version='from_P'):
    """
    Calculate spectral convolution G from hypergraph incidence matrix H.
    This is a wrapper that selects between sparse and dense implementations.

    :param H: hypergraph incidence matrix H (V x E)
    :param R: edge-dependent vertex weights for each vertex (E x V)
    :param E_weights: weights for each hyperedge (1 x E)
    :param Pi_version: which method to use for calculating Pi ('from_P' or 'from_A')
    :return: G (V x V)
    """
    # Check matrix dimensions
    v_size = H.shape[0]
    e_size = H.shape[1]

    # Use sparse implementation for large matrices
    if v_size > 1000 or e_size > 1000:
        print(
            f"Using sparse implementation for {v_size}x{e_size} hypergraph...")

        # Convert to sparse if not already
        H_sparse = sp.csr_matrix(H)
        R_sparse = sp.csr_matrix(R)

        # Use sparse implementation
        G_sparse = generate_G_from_H_sparse(
            H_sparse, R_sparse, E_weights, Pi_version)

        # Convert back to numpy array
        G = G_sparse.toarray()
        return G
    else:
        print(
            f"Using dense implementation for {v_size}x{e_size} hypergraph...")

        # Use original dense implementation
        P = generate_P(H, R, E_weights)

        if Pi_version == 'from_P':
            Pi = generate_Pi_from_P(P)
        elif Pi_version == 'from_A':
            Pi = generate_Pi_from_A(H, R, E_weights)
        else:
            raise ValueError(f"Unknown Pi_version: {Pi_version}")

        G = Pi - (Pi @ P + P.T @ Pi) / 2
        return G

# Original dense functions kept for backward compatibility and small matrices


def _generate_W(H, E_weights):
    '''
    Calculate W matrix.
    :param H: hypergraph incidence matrix H (V x E).
    :param E_weights: weights for each hyperedge (1 x E).
    :return: W
    '''
    W = np.zeros(H.shape)
    for e in range(H.shape[1]):
        W[:, e] = E_weights[e] * H[:, e]
    return W


def generate_P(H, R, E_weights):
    '''
    Calculate transition matrix P.
    :param H: hypergraph incidence matrix H (V x E).
    :param R: edge-dependent vertex weights for each vertex (E x V).
    :param E_weights: weights for each hyperedge (1 x E).
    :return: P (V x V).
    '''
    D_V = np.diag(np.count_nonzero(H, axis=1))
    D_E = np.diag(np.count_nonzero(H, axis=0))
    W = _generate_W(H, E_weights)

    # Handle potential singularity in D_V and D_E
    D_V_inv = np.linalg.inv(D_V + np.eye(D_V.shape[0]) * 1e-10)
    D_E_inv = np.linalg.inv(D_E + np.eye(D_E.shape[0]) * 1e-10)

    P = D_V_inv @ W @ D_E_inv @ R
    return P


def generate_Pi_from_P(P):
    '''
    Calculate stationary distribution Pi from transition matrix P.
    :param P: transition matrix P (V x V).
    :return: Pi (V x V).
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues, eigenvectors = np.linalg.eig(P)

    max_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_index]
    norm_max_eigenvector = np.abs(max_eigenvector) / \
        np.sum(np.abs(max_eigenvector))
    Pi = np.diag(norm_max_eigenvector)
    return Pi


def generate_Pi_from_A(H, R, E_weights):
    '''
    Calculate stationary distribution Pi from square matrix A.
    :param H: hypergraph incidence matrix H (V x E).
    :param R: edge-dependent vertex weights for each vertex (E x V).
    :param E_weights: weights for each hyperedge (1 x E).
    :return: Pi (V x V).
    '''
    D_V = np.diag(np.count_nonzero(H, axis=1))
    W = _generate_W(H, E_weights)

    # Handle potential singularity in D_V
    D_V_inv = np.linalg.inv(D_V + np.eye(D_V.shape[0]) * 1e-10)

    A = W.T @ D_V_inv @ R.T

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigenvalues, eigenvectors = np.linalg.eig(A)

    # Select the eigenvector with eigenvalue closest to 1
    rho_e = eigenvectors[:, (np.abs(eigenvalues - 1)).argmin()]
    pi_v = np.diagonal((W * rho_e) @ R)  # We don't need nondiagonal values

    # Ensure non-negative values and normalize
    pi_v = np.abs(pi_v)
    pi_v = pi_v / np.sum(pi_v)

    Pi = np.diag(pi_v)
    return Pi
