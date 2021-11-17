import numpy as np
import pandas as pd
from experiment_utils import *
from utils import *
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('num_replicates', type=int, nargs='?', default=1)
parser.add_argument('max_num_samples', type=int, nargs='?', default=1000)
args = parser.parse_args()

start_time = datetime.now()
df = pd.read_excel("data/poelwijk_supp3.xlsx")
X, y = poelwijk_construct_Xy(df)
num_samples_arr = np.arange(100, args.max_num_samples, 200)
run_lasso_across_sample_sizes(
	X, y,
	num_samples_arr=num_samples_arr,
	savefile='results/poelwijk_lasso_sample_sizes_r{}_n{}.pkl'.format(args.num_replicates, args.max_num_samples)
)
print('Time elapsed: {}'.format(datetime.now() - start_time))