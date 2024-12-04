# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Tamanna Urmi, Xiyu Yang
# Date: December 2024
# --------------------------------------------------------
import numpy as np
import numpy.linalg as LA


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


def _generate_R(H, V_weights):
    pass


def generate_P(H, E_weights, V_weights):
    '''
    Calculate transition matrix P.
    :param H: hypergraph incidence matrix H (V x E).
    :param E_weights: weights for each hyperedge (1 x E).
    :param V_weights: vertex weights for each vertex (1 x ?).
    :return: P (V x V).
    '''
    D_V = np.diag(np.count_nonzero(H, axis=1))
    D_E = np.diag(np.count_nonzero(H, axis=0))

    W = _generate_W(H, E_weights)
    R = _generate_R(H, V_weights)

    P = LA.inv(D_V) @ W @ LA.inv(D_E) @ R

    return P


def generate_Pi_from_P(P):
    '''
    Calculate stationary distribution Pi from transition matrix P.
    :param P: transition matrix P (V x V).
    :return: Pi (V x V).
    '''
    eigenvalues, eigenvectors = LA.eig(P)

    max_index = np.argmax(np.abs(eigenvalues))
    max_eigenvector = eigenvectors[:, max_index]
    norm_max_eigenvector = max_eigenvector / np.sum(max_eigenvector)

    Pi = np.diag(norm_max_eigenvector)

    return Pi


def generate_Pi_from_A(H, E_weights, V_weights):
    '''
    Calculate stationary distribution Pi from square matrix A.
    :param H: hypergraph incidence matrix H (V x E).
    :param E_weights: weights for each hyperedge (1 x E).
    :param V_weights: vertex weights for each vertex (1 x ?).
    :return: Pi (V x V).
    '''
    D_V = np.diag(np.count_nonzero(H, axis=1))
    W = _generate_W(H, E_weights)
    R = _generate_R(H, V_weights)
    A = W.T @ LA.inv(D_V) @ R.T

    eigenvalues, eigenvectors = LA.eig(A)

    # select the eigenvector with eigenvalue closest to 1.
    rho_e = eigenvectors[:, (np.abs(eigenvalues - 1)).argmin()]

    pi_v = np.diagonal((W * rho_e) @ R)  # we don't need non-diagonal values
    Pi = np.diag(pi_v)

    return Pi


def generate_G_from_H(H, E_weights, V_weights, Pi_version='from_P'):
    '''
    Calculate spectral convolution G from hypergraph incidence matrix H.
    :param H: hypergraph incidence matrix H (V x E).
    :param E_weights: weights for each hyperedge (1 x E).
    :param V_weights: vertex weights for each vertex (1 x ?).
    :return: G (V x V).
    '''
    P = generate_P(H, E_weights, V_weights)
    if Pi_version == 'from_P':
        Pi = generate_Pi_from_P(P)
    elif Pi_version == 'from_A':
        Pi = generate_Pi_from_A(H, E_weights, V_weights)

    G = Pi - (Pi @ P + P.T @ Pi) / 2

    return G
