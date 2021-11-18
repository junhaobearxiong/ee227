import matplotlib.pyplot as plt
import pickle

fontsize = 15
fig, ax = plt.subplots(figsize=(8, 6))
for m in ['lasso', 'ridge', 'ols']:
    if m == 'lasso':
        with open('results/poelwijk_lasso_sample_sizes_r50_n2000s100.pkl', 'rb') as f:
            results_dict = pickle.load(f)
    elif m == 'ridge':
        with open('results/poelwijk_ridge_sample_sizes_r50_n2000s100.pkl', 'rb') as f:
            results_dict = pickle.load(f)
    elif m == 'ols':
        with open('results/poelwijk_ols_sample_sizes_r50_n2000s100.pkl', 'rb') as f:
            results_dict = pickle.load(f)        
    
    ns = results_dict['num_samples']
    r2 = results_dict['y_r2']
    r2_mean = r2.mean(axis=0)
    r2_std = r2.std(axis=0)

    ax.errorbar(ns, r2_mean, yerr=r2_std, marker='o', label=m)
    ax.set_xlabel('Number of Training Samples', fontsize=fontsize)
    ax.set_ylabel('Prediction R-Square', fontsize=fontsize)

ax.legend()
ax.set_title('Prediction R-Square vs. Number of Training Samples on Poelwijk el al.', fontsize=fontsize)
plt.savefig('figures/poelwijk_r2.png')