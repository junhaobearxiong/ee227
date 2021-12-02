import matplotlib.pyplot as plt
import pickle

fig, axs = plt.subplots(figsize=(12, 8), sharex=True, sharey=True)
models = ['lasso', 'ridge', 'ols']
r = 10
nmin = 50
nmax = 2000
step = 100
title_fontsize = 30
ticks_fontsize = 20

for i, m in enumerate(models):
    if m == 'lasso':
        cv = 1
        label = 'Lasso: alpha=1e-5'
    elif m == 'ridge':
        cv = 1
        label = 'Ridge: alpha=1e-4'
    elif m == 'ols':
        cv = 0
        label = 'OLS'
    filename = 'results/poelwijk_{}_r{}_n{}-{}_s{}_cv{}'.format(m, r, nmin, nmax, step, cv)
    filename += '.pkl'

    with open(filename, 'rb') as f:
        results_dict = pickle.load(f)
    ns = results_dict['num_samples']
    y = results_dict['y_pearson_r']**2
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    beta = results_dict['beta_pearson_r']**2
    beta_mean = beta.mean(axis=0)
    beta_std = beta.std(axis=0)

    axs.errorbar(ns, y_mean, yerr=y_std, marker='o', label=label)
    axs.set_xlabel('Number of Training Samples', fontsize=ticks_fontsize)
    axs.set_ylabel('Predicted Fitness R^2', fontsize=ticks_fontsize)
    # axs[1].errorbar(ns, beta_mean, yerr=beta_std, marker='o', label=label)
    # axs[1].set_ylabel('Predicted Coeffcients R^2')
axs.legend(fontsize=ticks_fontsize)
axs.grid()
# axs[1].grid()

plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.suptitle('Prediction R-Square on Poelwijk el al.', fontsize=title_fontsize)
plt.tight_layout()
plt.savefig('figures/poelwijk_r2.png')