import numpy as np
import pandas as pd
from itertools import chain, combinations

"""
Partially cited from: https://github.com/dhbrookes/FitnessSparsity/
"""

def powerset(iterable):
    """Returns the powerset of a given set"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_all_interactions(L, index_1=False):
    """
    Returns a list of all epistatic interactions for a given sequence length.
    This sets of the order used for beta coefficients throughout the code.
    If index_1=True, then returns epistatic interactions corresponding to 
    1-indexing.
    """
    if index_1:
        pos = range(1, L+1)
    else:
        pos = range(L)
    all_U = list(powerset(pos))
    return all_U


def convert_01_bin_seqs(bin_seqs):
    """Converts a numpy array of {0, 1} binary sequences to {-1, 1} sequences"""
    assert type(bin_seqs) == np.ndarray
    bin_seqs[bin_seqs == 0] = -1
    return bin_seqs


def walsh_hadamard_from_seqs(bin_seqs):
    """
    Returns an N x 2^L array containing the Walsh-Hadamard encodings of
    a given list of N binary ({0,1}) sequences. This will return the 
    same array as fourier_from_seqs(bin_seqs, [2]*L), but is much
    faster.
    """
    bin_seqs_ = convert_01_bin_seqs(np.array(bin_seqs))
    L = len(bin_seqs_[0])
    all_U = get_all_interactions(L)
    M = 2**L
    N = len(bin_seqs)
    X = np.zeros((N, len(all_U)))
    for i, U in enumerate(all_U):
        if len(U) == 0:
            X[:, i] = 1
        else:
            X[:, i] = np.prod(bin_seqs_[:, U], axis=1)
    X = X / np.sqrt(M)
    return X


def walsh_hadamard_matrix(L=13, normalize=False):
    '''
    Compute the WHT matrix for domain of dimension L
    '''
    H1 = np.asarray([[1.,1.], [1.,-1.]])
    H = np.asarray([1.])
    for i in range(L):
        H = np.kron(H, H1)
    if normalize:
        H = (1 / np.sqrt(2**L)) * H
    return H