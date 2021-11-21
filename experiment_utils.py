import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from group_lasso import GroupLasso
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


def run_model_across_sample_sizes(X, y, model_name, num_samples_arr, savefile, num_replicates=1, beta=None, hyperparams=None, groups=None):
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
    hyperparams_list = [None] * num_replicates

    for i in range(num_replicates):
        print('replicate: {}'.format(i+1))
        # each replicate has an independent train test split
        # the actual training set consists of subsamples from `(X_train, y_train)`
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_samples_arr.max())
        # select hyperparams by cross-validation using the whole training set
        if hyperparams is None:
            hyperparams, model_cv = select_hyperparams(X_train, y_train, model_name)
        hyperparams_list[i] = hyperparams

        for j, n in enumerate(num_samples_arr):
            print('{} out of {} sampling pts'.format(j+1, num_samples_arr.size))
            # subsample n samples from the training set for actual training 
            # each subsamples trained with the same (already chosen) hyperparameters
            samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
            if model_name == 'lasso':
                model = Lasso(alpha=hyperparams['alpha']).fit(X_train[samples_idx, :], y_train[samples_idx])
            elif model_name == 'ols':
                model = LinearRegression().fit(X_train[samples_idx, :], y_train[samples_idx])
            elif model_name == 'ridge':
                model = Ridge(alpha=hyperparams['alpha']).fit(X_train[samples_idx, :], y_train[samples_idx])
            elif model_name == 'group_lasso':
                if groups is None:
                    raise ValueError('`groups` cannot be `None` for group_lasso')
                model = GroupLasso(groups=groups, group_reg=hyperparams['group_reg'], l1_reg=hyperparams['l1_reg'], supress_warning=True, n_iter=1000, tol=1e-3)
                model.fit(X_train[samples_idx, :], y_train[samples_idx])
            elif model_name == 'elastic_net':
                model = ElasticNet(alpha=hyperparams['alpha'], l1_ratio=hyperparams['l1_ratio']).fit(X_train[samples_idx, :], y_train[samples_idx])
            else:
                raise ValueError('model {} not implemented'.format(model_name))
            
            # evaluating model on test set
            pred = model.predict(X_test)
            y_mse[i, j] = np.sum(np.square(y_test - pred))
            y_pearson_r[i, j] = pearsonr(y_test, pred)[0]
            y_r2[i, j] = r2_score(y_test, pred)

            # TODO: figure out how best to deal with intercept: the first element of beta corresponds to intercept
            # but `model.intercept_` is not in the same scale as sum(y) / sqrt(M)
            beta_hat = model.coef_
            beta_hat[0] = model.intercept_
            beta_mse[i, j] = np.sum(np.square(beta - beta_hat))
            beta_pearson_r[i, j] = pearsonr(beta, beta_hat)[0]

    results_dict = {'num_samples': num_samples_arr, 'y_mse': y_mse, 'y_pearson_r': y_pearson_r, 'y_r2': y_r2,
        'beta_mse': beta_mse, 'beta_pearson_r': beta_pearson_r, 'hyperparams': hyperparams_list}
    with open(savefile, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict



def select_hyperparams(X_train, y_train, model_name, groups=None):
    hyperparams = {}
    alphas_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    print('------------ running cv for {} with sample size {} -------------'.format(model_name, X_train.shape[0]))
    if model_name == 'lasso':
        model_cv = LassoCV(alphas=alphas_list, n_jobs=10, max_iter=2000, tol=1e-3, verbose=1).fit(X_train, y_train)
        hyperparams['alpha'] = model_cv.alpha_
    elif model_name == 'ridge':
        model_cv = RidgeCV().fit(X_train, y_train)
        hyperparams['alpha'] = model_cv.alpha_
    elif model_name == 'elastic_net':
        l1_ratio_list = [.1, .5, .7, .9, .95, .99, 1]
        model_cv = ElasticNetCV(l1_ratio=l1_ratio_list, alphas=alphas_list, n_jobs=10).fit(X_train, y_train)
        hyperparams['alpha'] = model_cv.alpha_
        hyperparams['l1_ratio'] = model_cv.l1_ratio_
    elif model_name == 'group_lasso':
        params_dict = {}
        params_dict['group_reg'] =  [0, 1e-7, 1e-5, 1e-3, 1e-1, 1]
        params_dict['l1_reg'] = [0, 1e-7, 1e-5, 1e-3, 1e-1, 1]
        model = GroupLasso(groups=groups, supress_warning=True, n_iter=2000, tol=1e-3, warm_start=True)
        model_cv = GridSearchCV(model, params_dict, n_jobs=10, refit=False, verbose=1)
        model_cv.fit(X_train, y_train)
        hyperparams = model_cv.best_params_
    return (hyperparams, model_cv)



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

