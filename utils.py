import numpy as np
import pandas as pd
from itertools import chain, combinations, product
from scipy.special import binom
from math import factorial
from tqdm import tqdm

"""
Partially cited from: https://github.com/dhbrookes/FitnessSparsity/
"""

####################### Encodings ##########################################

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


def get_group_assignments(L, q):
    """
    Return a list of indices that indicate which group of interaction a coefficient belongs to
    size is q^L, with 2^L unique elements (i.e. number of groups)
    """
    # number of coeffs for each interaction term
    all_u = get_all_interactions(L)
    num_coeffs_per_u = [(q-1)**len(u) for u in all_u] 
    groups = []
    for i in range(len(num_coeffs_per_u)):
        # iterate over the groups
        groups.append(np.repeat(i+1, num_coeffs_per_u[i]))
    groups = np.concatenate(groups)
    return groups


def complete_graph_evs(q):
    """
    Returns a set of eigenvectors of complete graph of size q as column vectors of a matrix
    """
    x = np.ones(q)
    y = np.eye(q)[0]
    v = x - np.linalg.norm(x, ord=2) * y
    w = v / np.linalg.norm(v, ord=2)
    w = w.reshape(q, 1)
    P = np.eye(q) - 2*np.dot(w, w.T)
    return P


def fourier_basis_recursive(L, q):
    """
    Recursively constructs the Fourier basis corresponding to the H(L, q) hamming graph 
    (i.e. graphs of sequences of length L with alphabet size q). Hamming graphs are the 
    Cartesian product H(L+1, q) = H(L, q) x Kq, and the eigevectors of a cartesian product 
    (A x B) are the Kronecker product of the eigenvectors of A and B. This method is much 
    faster than constructing the the Fourier representation for each sequence individually
    (as with fourier_for_seqs, below) but the resulting basis less interpretable
    (i.e. it is difficult to associate rows with particular sequences and columns with
    particular epistatic interactions). In all of this code, we enforce that beta is ordered
    according to epistatic interactions, so DO NOT multiply the basis resulting from this
    method by a sample of beta.
    """
    # bear: for q = 2, this gives the same output as `walsh_hadamard_matrix`
    # i.e. the rows and columns are ordered as the canonical WH matrix
    # NOT by the order of interactions (by the order of interactions means: all the 1st order, then 2nd orders, etc)
    # but the canonical order "makes less sense" for larger alphabets
    Pq = complete_graph_evs(q)
    phi = np.copy(Pq)
    for i in range(L-1):
        phi = np.kron(Pq, phi)
    return phi


def get_encodings(qs):
    """
    Returns a length L list of arrays containing the encoding vectors corresponding 
    to each alphabet element at each position in sequence, given the alphabet size 
    at each position.
    bear: for alphabet size q, `encoding` is a q by q-1 matrix, where the ith row encodes
    the ith element in the alphabet with a q-1 vector 
    """
    encodings = []
    Pqs = []
    L = len(qs)
    for i in range(L):
        qi = qs[i]
        Pq = complete_graph_evs(qi) * np.sqrt(qi)
        Pqs.append(Pq)
        enc_i = Pq[:, 1:]
        encodings.append(enc_i)
    return encodings


def fourier_for_seq(int_seq, encodings):
    """
    Returns an M x 1 array containing the Fourier encoding of a sequence, 
    given the integer representation of the sequence and the encodings returned 
    by get_encodings, where M = prod(qs) and qs is the alphabet size at each position.
    """
    L = len(int_seq)
    all_U = get_all_interactions(L)
    all_U = [list(U) for U in all_U]
    epi_encs = []
    enc_1 = encodings[0][int_seq[0]] # the encoding of the first position of the sequence
    for U in all_U:
        if len(U) > 0 and 0 == U[0]:
        # bear: 0 == U[0] means the first alphabet is one of the positions of interaction in U
        # since its encoding is always the all-1s vector, we can ignore it  
            U_enc = enc_1
            U.pop(0)
        else:
            U_enc = np.array([1])
        epi_encs.append(U_enc)
    
    for l in range(1, L):
        # bear: for the lth position, get its element and look up its encoding (q-1 vector)
        enc_l = encodings[l][int_seq[l]]
        for k, U in enumerate(all_U):
            U_enc = epi_encs[k]
            if len(U) > 0 and l == U[0]:
                U_enc = np.kron(U_enc, enc_l)
                U.pop(0)
            epi_encs[k] = U_enc
    all_enc = np.concatenate(epi_encs)
    return all_enc


def fourier_from_seqs(int_seqs, qs):
    """
    Returns an N x M array containing the Fourier encodings of a given list of 
    N sequences with alphabet sizes qs.
    """
    if type(qs) == int:
        qs = [qs]*len(int_seqs[0])
    M = np.prod(qs)
    N = len(int_seqs)
    encodings = get_encodings(qs)
    phi = np.zeros((N, M))
    for i, seq in enumerate(tqdm(int_seqs)):
        phi[i] = fourier_for_seq(seq, encodings) / np.sqrt(M)
    return phi


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


#################################### GNK #####################################

def get_neighborhood_powerset(V):
    """Returns the union of powersets of a set of neighborhoods"""
    Vs = [sorted(Vk) for Vk in V]
    powersets = [tuple(utils.powerset(Vs[i])) for i in range(len(Vs))]
    T = set().union(*powersets)
    return T


def calculate_sparsity(L, q, V):
    """Calculates sparsity given any neighborhoods V=[V1,V2,...,VL]"""
    T = get_neighborhood_powerset(V)
    sparsity = 0
    for U in T:
        sparsity += (q-1)**len(U)
    return sparsity


def calc_bn_sparsity(L, q, K):
    """Calculates sparsity of Block Beighborhood scheme"""
    sparsity = (L/K)*(q**K - 1) +1
    return sparsity


def calc_an_sparsity(L, q, K):
    """Calculates sparsity of Adjacent Neighborhood scheme"""
    return 1 + L*(q-1)*q**(K-1)


def _calc_set_prob(r, L, K):
    """Calculates p(r) for a set of size r, for use with 'calc_mean_rn_sparsity'"""
    if r == 0 or r == 1:
        return 1
    else:
        ar = (factorial(K-1) / factorial(L-1)) * (factorial(L-r) / factorial(K-r))
        br = ar * ((K-r) / (L-r))
        term1 = (1-ar)**r
        term2 = (1-br)**(L-r)
        return 1-term1*term2
  

def calc_mean_rn_sparsity(L, q, K):
    """Calculates expected sparsity of Random Neighborhood scheme"""
    sparsity = 0
    for r in range(K+1):
        pr = _calc_set_prob(r, L, K)
        sparsity += binom(L, r)*pr *(q-1)**r
    return sparsity


def calc_max_rn_sparsity(L, q, K):
    """Calculates an upper bound on the sparsity of the Random Neighborhood scheme"""
    bd = 1+L*(q-1)
    for r in range(2, K+1):
        bd += L * binom(K, r) * (q-1)**r
    return bd


def build_adj_neighborhoods(L, K, symmetric=True):
    """Build Adjacent Neighborhoods with periodic boundary conditions"""
    V = []
    M = (K-1)/2
    for i in range(L):
        if symmetric:
            start = np.floor(i-M)
        else:
            start = i
        Vi = [int(((start + j) % L)+1) for j in range(K)]
        V.append(Vi)
    return V


def build_block_neighborhoods(L, K):
    """Build neighborhoods according to the Block Neighborhood scheme"""
    assert L % K == 0
    V = []
    block_size = int(L/K)
    for j in range(L):
        val = int(K*np.floor(j / K))
        Vj = list(range(val+1, val+K+1))
        V.append(Vj)
    return V


def sample_random_neighborhoods(L, K):
    """Sample neighborhoods according to the Random Neighborhood scheme"""
    V = []
    for i in range(L):
        indices = [j+1 for j in range(L) if j != i]
        Vi = list(np.random.choice(indices, size=K-1, replace=False))
        Vi.append(i+1)
        V.append(sorted(Vi))
    return V


def calc_beta_var(L, qs, V):
    """
    Calculates the variance of beta coefficients for a given sequence length, L, 
    list of alphabet sizes, and neighborhoods V. The returned coefficients are ordered
    by degree of epistatic interaction.
    """
    if type(qs) is int:
        qs = [qs]*L
    all_U = get_all_interactions(L, index_1=True) # index by 1 to match neighborhoods
    z = np.prod(qs)
    beta_var_U = []
    facs = []
    for j, Vj in enumerate(V):
        fac = 1
        for k in Vj:
            fac *= 1/qs[k-1]
        facs.append(fac)
    
    for i, U in enumerate(all_U):
        sz = np.prod([qs[k-1]-1 for k in U])
        bv = 0
        for j, Vj in enumerate(V):
            Uset = set(U)
            Vj_set = set(Vj)
            if Uset.issubset(Vj_set):
                bv += facs[j]
        bv *= z
        bv_expand = bv*np.ones(int(sz))
        beta_var_U.append(bv_expand)
    beta_var = np.concatenate(beta_var_U)
    return beta_var


def sample_gnk_fitness_function(L, qs, V='random', K=None):
    """
    Sample a GNK fitness function given the sequence length, alphabet sizes
    and neighborhoods. If V='random', V='block', or V='adjacent', then
    the neighborhoods will be set to the corresponding standard neighborhood
    scheme. Otherwise, V must be a list of neighborhoods. 
    """
    if type(V) is str:
        assert K is not None
    if V == 'random':
        V = sample_random_neighborhoods(L, K)
    elif V == 'adjacent':
        V = build_adj_neighborhoods(L, K)
    elif V == 'block':
        V = build_block_neighborhoods(L, K)

    beta_var = calc_beta_var(L, qs, V)
    use_wh = False
    if type(qs) is int:
        if qs == 2:
            use_wh = True
        qs = [qs]*L
    alphs = [list(range(q)) for q in qs]
    seqs = list(product(*alphs))
    if use_wh:
        phi = walsh_hadamard_from_seqs(seqs)
    else:
        phi = fourier_from_seqs(seqs, qs)
    beta = np.random.randn(len(beta_var))*np.sqrt(beta_var)
    f = np.dot(phi, beta)
    return (f, phi, beta, V)