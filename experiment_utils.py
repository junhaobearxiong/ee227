import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pickle
from collections import Counter
from utils import *


def get_idx_by_epistasis_order(bin_seqs):
    # order the indices of binary sequences by the order of epistasis 
    # i.e. number of 1's in each sequence
    return np.argsort(np.sum(bin_seqs==1, axis=1))


def poelwijk_convert_binary_str_to_arr(readfile=None):
    # convert the binary strings from poelwijk into np.array
    if readfile is None:
        readfile = 'data/poelwijk_supp3.xlsx'
    df = pd.read_excel(readfile)
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


def poelwijk_construct_Xy(readfile=None):
    if readfile is None:
        readfile = 'data/poelwijk_supp3.xlsx'
    df = pd.read_excel(readfile)
    y = np.array(df['brightness.2'][1:]).astype(float)
    '''
    # bear: this works fine too. but the ordering of beta is by epistatic order
    # and is not the same as X @ y
    bin_seqs = convert_binary_str_to_arr(df)
    L = len(bin_seqs[0])
    X = walsh_hadamard_from_seqs(bin_seqs)
    '''
    X = walsh_hadamard_matrix(L=13, normalize=True)
    return X, y


def run_model_across_sample_sizes(X, y, model_name, num_samples_arr, savefile, num_replicates=1, beta=None, alpha=None):
    """
    num_replicates: number of replicates of model to train on a given number of samples
    """
    print('------------{} for max number of samples: {}, number of replicates: {}-----------'.format(model_name, num_samples_arr.max(), num_replicates))
    if beta is None:
        beta = X @ y # true WHT
        print('------------- true beta is computed by X dot y ---------------------')
    # metrics to return
    y_mse = np.zeros((num_replicates, num_samples_arr.size))
    y_pearson_r = np.zeros((num_replicates, num_samples_arr.size))
    y_r2 = np.zeros((num_replicates, num_samples_arr.size))
    beta_mse = np.zeros((num_replicates, num_samples_arr.size))
    beta_pearson_r = np.zeros((num_replicates, num_samples_arr.size))
    alphas = np.zeros(num_replicates)

    for i in range(num_replicates):
        print('replicate: {}'.format(i+1))
        # each replicate has an independent train test split
        # the actual training set consists of subsamples from `(X_train, y_train)`
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_samples_arr.max())
        # select alpha using the whole training set
        if alpha is None:
            if model_name == 'lasso':
                alphas_list = [5e-7, 1e-7, 5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
                model_cv = LassoCV(alphas=alphas_list, n_jobs=10).fit(X_train, y_train)
            elif model_name == 'ridge':
                model_cv = RidgeCV().fit(X_train, y_train)
            elif model_name == 'elastic_net':
                model_cv = Elastic

            # elif model_name == 'group_lasso':
            alpha = model_cv.alpha_
        alphas[i] = alpha

        for j, n in enumerate(num_samples_arr):
            print('{} out of {} sampling pts'.format(j+1, num_samples_arr.size))
            samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
            # train model
            if model_name == 'lasso':
                model = Lasso(alpha=alpha).fit(X_train[samples_idx, :], y_train[samples_idx])
            elif model_name == 'ols':
                model = LinearRegression().fit(X_train[samples_idx, :], y_train[samples_idx])
            elif model_name == 'ridge':
                model = Ridge(alpha=alpha).fit(X_train[samples_idx, :], y_train[samples_idx])
            else:
                raise ValueError('model {} not implemented'.format(model_name))
            
            # evaluating model on test set
            pred = model.predict(X_test)
            y_mse[i, j] = np.sum(np.square(y_test - pred))
            y_pearson_r[i, j] = pearsonr(y_test, pred)[0]
            y_r2[i, j] = r2_score(y_test, pred)

            # TODO: figure out how best to deal with intercept: the first element of beta corresponds to intercept
            # but `model.intercept_` is not in the same scale as sum(y) / sqrt(M)
            beta_hat = model.coef_[1:]
            beta = beta[1:]
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
    alphas = [5e-7, 1e-7, 5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3]
    opt_vals = np.zeros(replicates)
    for j in range(replicates):
        samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
        model = LassoCV(alphas=alphas, n_jobs=10).fit(X_train[samples_idx, :], y_train[samples_idx])
        opt_vals[j] = model.alpha_
    cts = Counter(opt_vals)
    opt_alpha = cts.most_common(1)[0][0]
    return opt_alpha

