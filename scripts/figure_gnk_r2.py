import numpy as np
import matplotlib.pyplot as plt
import pickle

title_fontsize = 30
ticks_fontsize = 20
def plot_comparison(L, q, K, r, nmin, nmax, step, cv):
    fig, axs = plt.subplots(figsize=(12, 8), sharex=True, sharey=True)

    models = ['lasso', 'group_lasso']
    # models = ['lasso']

    for i, m in enumerate(models):
        if m == 'group_lasso':
            cv = 0
            label = 'Group Lasso: l1_reg=1e-4, group_reg=1e-4'
        elif m == 'lasso':
            cv = 2
            label = 'Lasso: alpha=1e-6'
        filename = 'results/gnk_{}_L{}_q{}_Vrandom_K{}_r{}_n{}-{}s{}_cv{}'.format(m, L, q, K, r, nmin, nmax, step, cv)
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
    plt.suptitle('Prediction R^2 on GNK Model'.format(L, q, K), fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig('figures/gnk_rsquared_{}_L{}_q{}_Vrandom_K{}_r{}_n{}-{}s{}_cv{}.png'.format(m, L, q, K, r, nmin, nmax, step, cv))

plot_comparison(L=4, q=10, K=2, r=10, nmin=100, nmax=1000, step=100, cv=1)