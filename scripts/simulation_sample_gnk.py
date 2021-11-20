from utils import sample_gnk_fitness_function
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('L', type=int)
parser.add_argument('q', type=int)
parser.add_argument('Vstr', type=str)
parser.add_argument('K', type=int)
args = parser.parse_args()

L = args.L # length of sequence
q = args.q # alphabet size
K = args.K # neighborhood size
Vstr = args.Vstr # neighborhood type 

y, X, beta, V = sample_gnk_fitness_function(L=L, qs=L*[q], V=Vstr, K=K)

gnk_model = {
    'y': y,
    'X': X,
    'beta': beta,
    'V': V
}

with open('data/gnk_L{}_q{}_V{}_K{}.pkl'.format(L, q, Vstr, K), 'wb') as f:
    pickle.dump(gnk_model, f)
