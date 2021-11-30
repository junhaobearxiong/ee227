import numpy as np
from experiment_utils import *
from utils import *
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str)
parser.add_argument('min_num_samples', type=int)
parser.add_argument('max_num_samples', type=int)
parser.add_argument('num_samples_step', type=int)
parser.add_argument('num_replicates', type=int)
parser.add_argument('cv', type=int)
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

X, y = poelwijk_construct_Xy()
num_samples_arr = np.arange(args.min_num_samples, args.max_num_samples, args.num_samples_step)
savefile = 'results/poelwijk_{}_r{}_n{}-{}_s{}_cv{}.pkl'.format(args.model_name, args.num_replicates, args.min_num_samples, 
    args.max_num_samples, args.num_samples_step, args.cv)
print('-----------results will be saved at: {}-------------------'.format(savefile))

run_model_across_sample_sizes(
    X, y,
    beta= X @ y,
    model_name=args.model,
    num_samples_arr=num_samples_arr,
    num_replicates=args.num_replicates,
    savefile=savefile,
    params_dict=params_dict,
    cv=args.cv
)
print('Time elapsed: {}'.format(datetime.now() - start_time))