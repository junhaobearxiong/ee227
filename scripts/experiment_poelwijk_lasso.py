import numpy as np
from experiment_utils import *
from utils import *
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num_replicates', type=int, nargs='?', default=1)
parser.add_argument('max_num_samples', type=int, nargs='?', default=1000)
parser.add_argument('num_samples_step', type=int, nargs='?', default=100)
parser.add_argument('alpha', type=float, nargs='?', default=None)
args = parser.parse_args()

start_time = datetime.now()
X, y = poelwijk_construct_Xy()
num_samples_arr = np.arange(100, args.max_num_samples, args.num_samples_step)
run_lasso_across_sample_sizes(
	X, y,
	alpha=args.alpha,
	num_samples_arr=num_samples_arr,
	num_replicates=args.num_replicates,
	savefile='results/poelwijk_lasso_sample_sizes_r{}_n{}s{}_alpha{}.pkl'.format(args.num_replicates, args.max_num_samples, args.num_samples_step, args.alpha)
)
print('Time elapsed: {}'.format(datetime.now() - start_time))