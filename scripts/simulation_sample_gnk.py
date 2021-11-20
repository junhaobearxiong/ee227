from utils import sample_gnk_fitness_function
import pickle

L = 6 # length of sequence
q = 3 # alphabet size
K = 4 # neighborhood size
Vstr = 'random' # neighborhood type 

y, X, beta, V = sample_gnk_fitness_function(L=L, qs=L*[q], V=Vstr, K=K)

gnk_model = {
    'y': y,
    'X': X,
    'beta': beta,
    'V': V
}

with open('data/gnk_L{}_q{}_V{}_K{}.pkl'.format(L, q, Vstr, K), 'wb') as f:
    pickle.dump(gnk_model, f)