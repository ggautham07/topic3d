"""
Methods for computations on tracing data and topic decomposition (:mod:`topic3d.compute`)
=====================================================

.. authors:: Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann

Function reference
------------------
"""


import numpy as np
from scipy.linalg import issymmetric

########################


def symmetric(matrix:np.ndarray) -> np.ndarray:
    """ Build a symmetric version of the input matrix.
    If the lower triangle matrix is filled with zeros, the upper triangle matrix is copied to the lower whilst retaining the original diagonal elements.
    Otherwise, it adds the matrix to its transpose and divides the new matrix by two.
    
    Input
    ----------
    matrix : two-dimensional square array

    Returns
    ----------
    a symmetric matrix
    """

    assert matrix.ndim == 2

    if issymmetric(matrix):
        return matrix
    return matrix + matrix.T - np.diag(np.diag(matrix)) if (np.tril(matrix, k=-1) == 0).all() else (matrix + matrix.T) / 2

########################


def compute_cov_matrix(matrix:np.ndarray) -> np.ndarray:
    """ Compute the covariance matrix from a symmetric mean-squared distance matrix assuming a Gaussian distribution
    
    Input
    ----------
    matrix : two-dimensional square symmetric array

    Returns
    ----------
    cov : covariance matrix computed using Daniel Jost's method
    """

    assert matrix.ndim == 2
    assert issymmetric(matrix)
    N = matrix.shape[0]

    cov = np.zeros((N-1, N-1))
    np.fill_diagonal(cov, matrix[0,1:])
    for i in range(N-1):
        for j in range(i+1, N-1):
            cov[i,j] = (cov[i,i] + cov[j,j] - matrix[i+1, j+1]) / 2
    cov = symmetric(cov)
    
    return cov





