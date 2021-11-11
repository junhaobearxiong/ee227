import numpy as np
import pandas as pd
from experiment_utils import *
from utils import *

from datetime import datetime


start_time = datetime.now()
df = pd.read_excel("data/poelwijk_supp3.xlsx")
X, y = poelwijk_construct_Xy(df)
num_samples_arr = np.arange(100, 1000, 100)
run_lasso_across_samples(X, y, num_samples_arr, 1e-5, 'results/poelwijk_lasso_samples.pkl')
print('Time elapsed: {}'.format(datetime.now() - start_time))