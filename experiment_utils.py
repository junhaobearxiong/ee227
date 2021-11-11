import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pickle
from utils import *


def poelwijk_construct_Xy(df):
    y = np.array(df['brightness.2'][1:]).astype(float)
    '''
    TODO: i don't understand why this works. X is not ordered the same way y is 
    y is ordered the same way W-H coefficients are
    X is ordered by orders of interactions
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
    L = len(bin_seqs[0])
    X = walsh_hadamard_from_seqs(bin_seqs)
    '''
    X = walsh_hadamard_matrix(L=13, normalize=True)
    return X, y


def run_lasso_across_samples(X, y, num_samples_arr, alpha, savefile, num_replicates=1, train_size=5000):
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

    for i, r in enumerate(range(num_replicates)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        for j, n in enumerate(num_samples_arr):
            print('n: {}'.format(n))
            # randomly subsample n samples from training set
            samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
            # TODO: add CV to select alpha
            model = Lasso(alpha=alpha)
            model.fit(X_train[samples_idx, :], y_train[samples_idx])
            pred = model.predict(X_test)
            y_mse[i, j] = np.sum(np.square(y_test - pred))
            y_pearson_r[i, j] = pearsonr(y_test, pred)[0]
            y_r2[i, j] = r2_score(y_test, pred)
            beta_hat = model.coef_
            beta_mse[i, j] = np.sum(np.square(beta - beta_hat))
            beta_pearson_r[i, j] = pearsonr(beta, beta_hat)[0]
            print(model.intercept_)

    results_dict = {'num_samples': num_samples_arr, 'y_mse': y_mse, 'y_pearson_r': y_pearson_r, 'y_r2': y_r2,
        'beta_mse': beta_mse, 'beta_pearson_r': beta_pearson_r}
    with open(savefile, 'wb') as f:
        pickle.dump(results_dict, f)
    return results_dict

