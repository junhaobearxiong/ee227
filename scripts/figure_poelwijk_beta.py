from plot_utils import *
from experiment_utils import *

n = 500

X, y = poelwijk_construct_Xy()
bin_seqs = poelwijk_convert_binary_str_to_arr()
idx_epi_ord = get_idx_by_epistasis_order(bin_seqs)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000)
samples_idx = np.random.choice(np.arange(X_train.shape[0]), n, replace=False)
model = Lasso(alpha=1e-5).fit(X_train[samples_idx, :], y_train[samples_idx])

betahat = model.coef_
betahat[0] = model.intercept_
betahat = betahat[idx_epi_ord]
beta = X @ y
beta = beta[idx_epi_ord]
beta = beta[1:]
betahat = betahat[1:]

ax = plot_beta_comparison(beta, betahat, max_order=4, width=3)
plt.title('True vs. Predicted Fourier Coefficient on Poelwijk el al. ({} Training Samples)'.format(n), fontsize=40)
plt.tight_layout()
plt.savefig('figures/poelwijk_beta_n{}.png'.format(n))