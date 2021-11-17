import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pickle
from collections import Counter
from utils import *


def convert_binary_str_to_arr(df):
    # convert the binary strings from poelwijk into np.array
    bin_seqs_ = list(df['binary'][1:])
    bin_seqs = []
    for s_ in bin_seqs_:
        s_ = s_[1:-1]
        s = []
        for si in s_:
            if si == '0':
                s.append(0)
            else:
                s.append(1)
        bin_seqs.append(s)
    bin_seqs = np.array(bin_seqs)
    return bin_seqs


def get_idx_by_epistasis_order(bin_seqs):
    # order the indices of binary sequences by the order of epistasis 
    # i.e. number of 1's in each sequence
    return np.argsort(np.sum(bin_seqs==1, axis=1))


def poelwijk_construct_Xy(df):
    y = np.array(df['brightness.2'][1:]).astype(float)
    '''
    TODO: i don't understand why this works. X is not ordered the same way y is 
    y is ordered the same way W-H coefficients are
    X is ordered by orders of interactions
    bin_seqs = convert_binary_str_to_arr(df)
    L = len(bin_seqs[0])
    X = walsh_hadamard_from_seqs(bin_seqs)
    '''
    X = walsh_hadamard_matrix(L=13, normalize=True)
    return X, y


def run_lasso_across_sample_sizes(X, y, num_samples_arr, savefile, alpha=None, num_replicates=1, train_size=5000):
    """
    num_replicates: number of replicates of model to train on a given number of samples
    """
    beta = X @ y # true WHT
    # metrics to return
    y_mse = np.zeros((num_replicates, num_samples_arr.size))
    y_pearson_r = np.zeros((num_replicates, num_samples_arr.size))
    y_r2 = np.zeros((num_replicates, num_samples_arr.size))
    beta_mse = np.zeros((num_replicates, num_samples_arr.size))
    beta_pearson_r = np.zeros((num_replicates, num_samples_arr.size))
    alphas = np.zeros((num_replicates, num_samples_arr.size))

    for i in range(num_replicates):
        # each replicate has an independent train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        for j, n in enumerate(num_samples_arr):
            print('replicate: {}, n: {}'.format(i+1, n))
            # using 10 independent samples of size n to select alpha before sampling for training
            if alpha is None:
                alpha = determine_alpha(X_train, y_train, n, 1)
            alphas[i, j] = alpha
            # randomly subsample n samples from training set for actual training
            samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
            model = Lasso(alpha=alpha)
            model.fit(X_train[samples_idx, :], y_train[samples_idx])

            # evaluating on test set
            pred = model.predict(X_test)
            y_mse[i, j] = np.sum(np.square(y_test - pred))
            y_pearson_r[i, j] = pearsonr(y_test, pred)[0]
            y_r2[i, j] = r2_score(y_test, pred)
            beta_hat = model.coef_
            beta_mse[i, j] = np.sum(np.square(beta - beta_hat))
            beta_pearson_r[i, j] = pearsonr(beta, beta_hat)[0]

    results_dict = {'num_samples': num_samples_arr, 'y_mse': y_mse, 'y_pearson_r': y_pearson_r, 'y_r2': y_r2,
        'beta_mse': beta_mse, 'beta_pearson_r': beta_pearson_r, 'alpha': alphas}
    with open(savefile, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict


def determine_alpha(X_train, y_train, n, replicates=10):
    """
    Determines the optimal regularization parameter for n data points randomly subsampled from
    a given training set (X_train, y_train)
    """
    alphas = [5e-8, 1e-8, 5e-7, 1e-7, 5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
    opt_vals = np.zeros(replicates)
    for j in range(replicates):
        model = LassoCV(alphas=alphas, n_jobs=10)
        model.fit(X_train, y_train)
        opt_vals[j] = model.alpha_
    cts = Counter(opt_vals)
    opt_alpha = cts.most_common(1)[0][0]
    return opt_alpha

