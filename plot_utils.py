import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom


def plot_beta_comparison(beta, betahat, savefile, L=13, max_order=5):
    """
    Plot comparison between true (`beta`) and predicted (`betahat`) epistasis coeffcients
    Assumed beta and betahat are sorted by epistatic orders
    `max_order`: the maximum order to plot
    `L`: length of the sequences 
    """
    fig, ax = plt.subplots(figsize=(24, 10))

    # these parameters are set manually according to figure size
    use_order_labels = True
    order_label_fontsize=20
    order_lbl_offset=25, 
    order_lbl_height=20
    arrow1_xy=(7, 25)
    arrow1_text_xy=(52, 30)
    arrow2_xy=(35, 35)
    arrow2_text_xy=(83, 40) 
    yticks=(-40, -20, 0, 20, 40)
    yticklabels=('40', '20', '0', '20', '40')

    num_coeffs = int(np.sum([binom(L, i) for i in range(max_order+1)])) # up to 5th order interactions
    colors = sns.color_palette('Set1', n_colors=2)
    mv = np.max([np.max(beta), np.max(betahat)])

    ax.bar(range(num_coeffs), beta[:num_coeffs], width=3, color=colors[0], label='True')
    ax.bar(range(num_coeffs), -betahat[:num_coeffs], width=3, color=colors[1], label='Predicted')

    ax.plot((-10, num_coeffs),(0, 0), c='k')
    ticks = [np.sum([binom(L, j) for j in range(i)]) for i in range(L+1)]
    ticks = [t for t in ticks if t <= num_coeffs]
    ordlbls = ["1st", "2nd", "3rd"] + ["%ith" for i in range(3, L+1)]
    for i, tick in enumerate(ticks):
        ax.vlines(tick, ymin=-mv, ymax=mv, color='k', ls='--', lw=0.5, alpha=1)
        if i > 2 and i <= max_order:
            if use_order_labels:
                ax.text(tick+order_lbl_offset, order_lbl_height, "$r=%i$" %i, fontsize=order_label_fontsize)

    if use_order_labels:
        ax.annotate("",
                    xy=arrow1_xy, xycoords='data',
                    xytext=arrow1_text_xy, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>, head_width=0.15",facecolor='k',
                                    connectionstyle="arc3,rad=0.15"),
                    )

        ax.text(arrow1_text_xy[0], arrow1_text_xy[1], "$r=1$", fontsize=order_label_fontsize)

        ax.annotate("",
                    xy=arrow2_xy, xycoords='data',
                    xytext=arrow2_text_xy, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>, head_width=0.15",facecolor='k',
                                    connectionstyle="arc3,rad=0.15"),
                    )

        ax.text(arrow2_text_xy[0], arrow2_text_xy[1], "$r=2$", fontsize=order_label_fontsize)

    ax.tick_params(axis='y', which='major', direction='out')
    # ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim([-mv, mv])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=order_label_fontsize)
    ax.set_xlim([-10, num_coeffs+5])
    ax.set_ylabel("Magnitude of Fourier coefficient", labelpad=30, fontsize=order_label_fontsize)
    ax.legend(fontsize=order_label_fontsize)
    ax.grid(False)

    plt.savefig(savefile)