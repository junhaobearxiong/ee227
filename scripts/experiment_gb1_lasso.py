import numpy as np
from experiment_utils import *
from utils import *
from datetime import datetime
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('min_num_samples', type=int)
parser.add_argument('max_num_samples', type=int)
parser.add_argument('num_samples_step', type=int)
parser.add_argument('num_replicates', type=int)
parser.add_argument('cv', type=int)
parser.add_argument('data_dir', type=str)
parser.add_argument('train_size', type=int)
args = parser.parse_args()
start_time = datetime.now()

params_dict = {}
if args.cv == 0:
    if args.model_name == 'lasso':
        params_dict['alpha'] = 1e-6
    elif args.model_name == 'group_lasso':
        params_dict['group_reg'] = 1e-4
        params_dict['l1_reg'] = 1e-4
    print('`cv == 0`, no cross validation is performed, using hyperparams'.format(params_dict))
elif args.cv == 1:
    if args.model_name == 'lasso':
        params_dict['alpha'] =  [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    elif args.model_name == 'group_lasso':
        params_dict['group_reg'] = [1e-5, 1e-4, 1e-3, 1e-2] # [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        params_dict['l1_reg'] = [1e-5, 1e-4, 1e-3, 1e-2] # [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    print('`cv == 1`, using a coarse grid for cross validation')
elif args.cv == 2:
    if args.model_name == 'lasso':
        params_dict['alpha'] = 10**np.linspace(-7, -5, 20) 
    elif args.model_name == 'group_lasso':
        pass
    print('`cv == 2`, using a fine grid for cross validation')
else:
    raise ValueError('cv == {} is not valid'.format(args.cv))


with open(args.data_dir + 'gb1_Xtrain_{}.pkl'.format(args.train_size), 'rb') as f:
    X = pickle.load(f)
with open(args.data_dir + 'gb1_ytrain_{}.pkl'.format(args.train_size), 'rb') as f:
    y = pickle.load(f)

num_samples_arr = np.arange(args.min_num_samples, args.max_num_samples + 1, args.num_samples_step)
savefile = 'results/gb1_{}_{}_r{}_n{}-{}_s{}_cv{}.pkl'.format(args.train_size, args.model_name, args.num_replicates, args.min_num_samples, args.max_num_samples, 
        args.num_samples_step, args.cv)
print('-----------results will be saved at: {}-------------------'.format(savefile))

results = run_model_across_sample_sizes(X, y,
    beta=None,
    model_name=args.model_name,
    num_samples_arr=num_samples_arr,
    num_replicates=args.num_replicates,
    savefile=savefile,
    params_dict=params_dict
)
print('Time elapsed: {}'.format(datetime.now() - start_time))