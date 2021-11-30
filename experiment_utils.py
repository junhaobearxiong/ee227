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


def gb1_construct_Xy(data_dir=None, train_size=5000, test_size=2000):
    # only need to run once
    np.random.seed(0)

    def str_seqs_to_char_list(str_seqs):
        return str_seqs.apply(lambda x: [char for char in x]).values

    def get_int_seqs(str_seqs, alphabet):
        # process each variant sequence (pd series) into an integer encoding
        char_arr = str_seqs_to_char_list(str_seqs)

        dictionary = {}
        for i, a in enumerate(alphabet):
            dictionary[a] = i

        int_seqs = [None] * char_arr.size
        for i, s in enumerate(char_arr):
            seq = [dictionary[c] for c in s]
            int_seqs[i] = seq
        return int_seqs

    if data_dir is None:
        data_dir = 'data/'
    gb1 = pd.read_csv(data_dir + 'gb1.csv', index_col=0)
    str_seqs = gb1['Variants']
    char_arr = str_seqs_to_char_list(str_seqs)
    alphabet = np.unique(np.concatenate(char_arr))

    sample_idx = np.random.choice(np.arange(gb1.shape[0]), train_size + test_size, replace=False)
    train_sample_idx = sample_idx[:train_size]
    test_sample_idx = sample_idx[train_size:]
    y_train = gb1['Fitness'][train_sample_idx].values
    y_test = gb1['Fitness'][test_sample_idx].values

    train_seqs = get_int_seqs(gb1['Variants'][train_sample_idx], alphabet)
    test_seqs = get_int_seqs(gb1['Variants'][test_sample_idx], alphabet)
    X_train = fourier_from_seqs(train_seqs, 20)
    X_test = fourier_from_seqs(test_seqs, 20)

    with open(data_dir + 'gb1_Xtrain_{}.pkl'.format(train_size), 'wb') as f:
        pickle.dump(X_train, f)
    with open(data_dir + 'gb1_ytrain_{}.pkl'.format(train_size), 'wb') as f:
        pickle.dump(y_train, f)
    with open(data_dir + 'gb1_Xtest_{}.pkl'.format(test_size), 'wb') as f:
        pickle.dump(X_test, f)
    with open(data_dir + 'gb1_ytest_{}.pkl'.format(test_size), 'wb') as f:
        pickle.dump(y_test, f)


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


def run_model_across_sample_sizes(X, y, model_name, num_samples_arr, savefile, num_replicates=1, 
    beta=None, cv=1, params_dict=None, groups=None, ignore_intercept=False):
    """
    num_replicates: number of replicates of model to train on a given number of samples
    cv: whether to select hyperparams by cross validation with the function `select_hyperparams`
    params_dict: if `cv == 1` (coarse grid) or `cv == 2` (fine grid), then `params_dict` should give the range of hyperparameters for `GridSearchCV`
        if `cv == 0`, then `params_dict` should provide the hyperparameter used in training, no cross validation is performed
    """
    print('------------{} for max number of samples: {}, number of replicates: {}-----------'.format(model_name, num_samples_arr.max(), num_replicates))
    if beta is not None:
        beta_pearson_r = np.zeros((num_replicates, num_samples_arr.size))
        beta_mse = np.zeros((num_replicates, num_samples_arr.size))
        if ignore_intercept:
            beta = beta[1:]
    # metrics to return
    y_mse = np.zeros((num_replicates, num_samples_arr.size))
    y_pearson_r = np.zeros((num_replicates, num_samples_arr.size))
    hyperparams_list = [None] * num_replicates
    model_cv_list = [None] * num_replicates
    # model_list = [None] * num_replicates

    for i in range(num_replicates):
        print('replicate: {}'.format(i+1))
        # each replicate has an independent train test split
        # the actual training set consists of subsamples from `(X_train, y_train)`
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=num_samples_arr.max())
        X_train = np.asfortranarray(X_train)
        if cv > 0:
            model_cv = select_hyperparams(X_train, y_train, model_name, params_dict)
            model_cv_list[i] = pd.DataFrame(model_cv.cv_results_)
            hyperparams = model_cv.best_params_
            hyperparams_list[i] = hyperparams
        else:
            hyperparams = params_dict

        for j, n in enumerate(num_samples_arr):
            print('{} out of {} sampling pts'.format(j+1, num_samples_arr.size))
            # subsample n samples from the training set for actual training 
            # each subsamples trained with the same (already chosen) hyperparameters
            samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
            X_train_samples = X_train[samples_idx, :]
            X_train_samples = np.asfortranarray(X_train_samples)
            y_train_samples = y_train[samples_idx]
            if model_name == 'lasso':
                model = Lasso(alpha=hyperparams['alpha'], max_iter=5000, tol=1e-3).fit(X_train_samples, y_train_samples)
            elif model_name == 'ols':
                model = LinearRegression().fit(X_train_samples, y_train_samples)
            elif model_name == 'ridge':
                model = Ridge(alpha=hyperparams['alpha']).fit(X_train_samples, y_train_samples)
            elif model_name == 'group_lasso':
                if groups is None:
                    raise ValueError('`groups` cannot be `None` for group_lasso')
                model = GroupLasso(groups=groups, group_reg=hyperparams['group_reg'], l1_reg=hyperparams['l1_reg'], supress_warning=True, n_iter=5000, tol=1e-3)
                model.fit(X_train_samples, y_train_samples)
            elif model_name == 'elastic_net':
                model = ElasticNet(alpha=hyperparams['alpha'], l1_ratio=hyperparams['l1_ratio']).fit(X_train_samples, y_train_samples)
            else:
                raise ValueError('model {} not implemented'.format(model_name))
            
            # evaluating model on test set
            pred = model.predict(X_test)
            y_mse[i, j] = np.sum(np.square(y_test - pred))
            y_pearson_r[i, j] = pearsonr(y_test, pred)[0]

            if beta is not None:
                # TODO: figure out how best to deal with intercept: the first element of beta corresponds to intercept
                # but `model.intercept_` is not in the same scale as sum(y) / sqrt(M)
                beta_hat = model.coef_
                beta_hat[0] = model.intercept_
                if ignore_intercept:
                    beta_hat = beta_hat[1:]
                beta_mse[i, j] = np.sum(np.square(beta - beta_hat))
                beta_pearson_r[i, j] = pearsonr(beta, beta_hat)[0]

    results_dict = {'num_samples': num_samples_arr, 'y_mse': y_mse, 'y_pearson_r': y_pearson_r,
            'hyperparams': hyperparams_list, 'model_cv': model_cv_list}
    if beta is not None:
        results_dict['beta_pearson_r'] = beta_pearson_r
        results_dict['beta_mse'] = beta_mse

    with open(savefile, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict



def select_hyperparams(X_train, y_train, model_name, params_dict, groups=None):
    print('------------ running cv for {} with sample size {} -------------'.format(model_name, X_train.shape[0]))
    if model_name == 'lasso':
        model = Lasso(max_iter=5000, tol=1e-3)
        model_cv = GridSearchCV(model, params_dict, n_jobs=-1, refit=False, verbose=1)
        model_cv.fit(X_train, y_train)
    elif model_name == 'group_lasso':
        model = GroupLasso(groups=groups, supress_warning=True, n_iter=5000, tol=1e-3)
        model_cv = GridSearchCV(model, params_dict, n_jobs=1, refit=False, verbose=1)
        model_cv.fit(X_train, y_train)
    elif model_name == 'ridge':
        model = Ridge()
        model_cv = GridSearchCV(model, params_dict, n_jobs=-1, refit=False, verbose=1)
        model_cv.fit(X_train, y_train)
    else:
        raise ValueError('{} not implemented for cv'.format(model_name))
    return model_cv
