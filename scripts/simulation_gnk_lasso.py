from experiment_utils import run_model_across_sample_sizes
import argparse
import pickle
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('L', type=int)
parser.add_argument('q', type=int)
parser.add_argument('Vstr', type=str)
parser.add_argument('K', type=int)
parser.add_argument('n', type=int, nargs='?', default=2)
parser.add_argument('num_replicates', type=int, nargs='?', default=1)

with open('data/gnk_L{}_q{}_V{}_K{}.pkl'.format(args.L, args.q, args.Vstr, args.K), 'rb') as f:
	gnk_model = pickle.load(f)

start_time = datetime.now()
X = gnk_model['X']
y = gnk_model['y']
beta = gnk_model['beta']
num_samples_arr = np.linspace(100, X.shape[0], args.n, dtype=int)
results = run_model_across_sample_sizes(X, y, beta=beta, 
	model_name='lasso', 
	num_samples_arr=num_samples_arr,
	num_replicates=args.num_replicates, 
	savefile='results/gnk_L{}_q{}_V{}_K{}_n{}_r{}'.format(args.L, args.q, args.V, args.K, args.n, args.num_replicates))
print('Time elapsed: {}'.format(datetime.now() - start_time))