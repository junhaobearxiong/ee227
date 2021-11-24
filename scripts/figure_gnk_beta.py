import numpy as np
import matplotlib.pyplot as plt
import pickle
from experiment_utils import *
from plot_utils import *
from utils import *
import os.path

np.random.seed(0)
train_size = 600
L = 4
q = 10
K = 2
Vstr = 'random'


def fit_model(X_train, y_train, model_name):
    if model_name == 'group_lasso':
        model = GroupLasso(groups=groups, group_reg=1e-4, l1_reg=1e-4, supress_warning=True, n_iter=5000, tol=1e-3)
        model.fit(X_train, y_train)
    elif model_name == 'lasso':
        model = Lasso(alpha=1.5e-6, max_iter=5000, tol=1e-3).fit(X_train, y_train)

    betahat = model.coef_.flatten()
    betahat[0] = model.intercept_
    return betahat


'''
Load data
'''
with open('data/gnk_L{}_q{}_V{}_K{}.pkl'.format(L, q, Vstr, K), 'rb') as f:
    gnk_model = pickle.load(f)

X = gnk_model['X']
y = gnk_model['y']
beta = gnk_model['beta']
groups = get_group_assignments(L, q)
group_idx = np.unique(groups, return_index=True)[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

ax = plot_neighborhoods(gnk_model['V'], L, range(1, L+1))
plt.savefig('figures/gnk_neighborhoods_L{}_q{}_V{}_K{}.png'.format(L, q, Vstr, K))


'''
Fit models 
'''
lasso_filename = 'results/gnk_beta_lasso_L{}_q{}_Vstr{}_K{}_trainsize{}'.format(L, q, Vstr, K, train_size)
glasso_filename = 'results/gnk_beta_group_lasso_L{}_q{}_Vstr{}_K{}_trainsize{}'.format(L, q, Vstr, K, train_size)

if os.path.exists(lasso_filename):
    with open(lasso_filename, 'rb') as f:
        betahat_lasso = pickle.load(f)
else:
    betahat_lasso = fit_model(X_train, y_train, 'lasso')
    with open(lasso_filename, 'wb') as f:
        pickle.dump(betahat_lasso, f)

if os.path.exists(glasso_filename):
    with open(glasso_filename, 'rb') as f:
        betahat_glasso = pickle.load(f)
else:
    betahat_glasso = fit_model(X_train, y_train, 'group_lasso')
    with open(glasso_filename, 'wb') as f:
        pickle.dump(betahat_glasso, f)


'''
Plot betas
'''
for num_coeffs in [500, 3000]:
    if num_coeffs == 500:
        width = 1
    elif num_coeffs == 3000:
        width = 3
    fig, axs = plt.subplots(3, 1, figsize=(25, 12), sharex=True, sharey=True)
    plot_beta(axs[0], beta, group_idx, 'True', num_coeffs, width)
    plot_beta(axs[1], betahat_lasso, group_idx, 'Lasso', num_coeffs, width)
    plot_beta(axs[2], betahat_glasso, group_idx, 'Group Lasso', num_coeffs, width)
    plt.suptitle('Comparisons of Coefficients on GNK Model (length {}, alphabet size {}, {} neighborhood with K = {}, training size = {})'.format(L, q, Vstr, K, train_size), fontsize=20)
    plt.tight_layout()
    plt.savefig('figures/gnk_beta_L{}_q{}_V{}_K{}_trainsize{}_numcoeffs{}.png'.format(L, q, Vstr, K, train_size, num_coeffs))