import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd
import math

import os


def plot_feature_histograms(X_val, X_sim, biased_unbiased, time_points=[80, 90, 99], bins=30):
    """
    Compare per-feature distributions using histograms.
    
    X_val, X_sim: shape (T, num_paths, d)
    time_points: list of time indices to plot
    """
    T, num_paths, d = X_val.shape
    colors = plt.cm.tab10(np.arange(d))

    n_rows = len(time_points)
    n_cols = d

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3*n_cols, 3*n_rows),
                             squeeze=False)
    #for t in time_points:
    #   fig, axes = plt.subplots(1, d, figsize=(4*d, 3))
    for i, t in enumerate(time_points):    
        for j in range(d):
            ax = axes[i, j]
            
            # flatten 1000 observations
            real_vals = X_val[t, :, j]
            sim_vals  = X_sim[t, :, j]
            
            ax.hist(real_vals, bins=bins, density=True,
                    alpha=0.5, color=colors[j], label="real")
            
            ax.hist(sim_vals, bins=bins, density=True,
                    histtype='step', linewidth=2,
                    color=colors[j], label="sim")
            
            if i == 0:
                ax.set_title(f"Feature {j+1}", fontsize=12)

            if j == 0:
                ax.set_ylabel(f"t = {t}", fontsize=12)

            ax.tick_params(labelsize=10)
            ax.legend(fontsize = 9)
            #ax.set_title(f"Feature {j+1}", fontsize=12)
            ax.tick_params(labelsize=10)
        
    #axes[0, 0].plot([], [], color='black', alpha=0.5, label='real')
    #axes[0, 0].plot([], [], color='black', linewidth=2, label='sim')
    #fig.legend(loc='upper right', fontsize=12)

    fig.suptitle(f"Feature distributions at selected time points ({biased_unbiased})", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        #fig.suptitle(f"Feature distributions at time t={t}", fontsize=16)
        #axes[0].legend(fontsize=10)
        #plt.tight_layout()
        #plt.show()
        
def plot_feature_histograms_3plots(
    X_val, X_sim_biased, X_sim_unbiased,
    time_points=[80, 90, 99], bins=30,
    save_dir="plots"
):
    """
    Compare per-feature empirical distributions using histograms.

    X_val, X_sim_biased, X_sim_unbiased: shape (T, num_paths, d)
    """
    os.makedirs(save_dir, exist_ok=True)

    T, num_paths, d = X_val.shape
    colors = plt.cm.tab10(np.arange(d))

    n_rows = len(time_points)
    n_cols = d

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False
    )

    for i, t in enumerate(time_points):
        for j in range(d):
            ax = axes[i, j]

            real_vals = X_val[t, :, j]
            biased_vals = X_sim_biased[t, :, j]
            unbiased_vals = X_sim_unbiased[t, :, j]

            ax.hist(real_vals, bins=bins, density=True,
                    alpha=0.45, color=colors[j], label="real")

            ax.hist(biased_vals, bins=bins, density=True,
                    histtype="step", linewidth=2,
                    color=colors[j], linestyle="-",
                    label="predicted biased")

            ax.hist(unbiased_vals, bins=bins, density=True,
                    histtype="step", linewidth=2,
                    color=colors[j], linestyle="--",
                    label="predicted unbiased")

            if i == 0:
                ax.set_title(f"Feature {j+1}", fontsize=12)

            if j == 0:
                ax.set_ylabel(f"t = {t}", fontsize=12)

            ax.tick_params(labelsize=10)
            ax.legend(fontsize=8)

    fig.suptitle("Feature distributions at selected time points", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_dir, "feature_histograms.png"), dpi=300)
    plt.show()



def plot_distribution_error(X_val, X_sim, biased_unbiased, steps_train, feature_names):
    def wasserstein_per_feature(X_val, X_sim):
        T, n, d = X_val.shape
        W = np.zeros((T, d))

        for t in range(T):
            for j in range(d):
                W[t, j] = wasserstein_distance(
                    X_val[t, :, j],
                    X_sim[t, :, j]
                )
        
        return W

    W = wasserstein_per_feature(X_val, X_sim)

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.arange(W.shape[1]))

    for j in range(W.shape[1]):
        plt.plot(W[:, j],
                 linewidth=3,
                 color=colors[j],
                 label=feature_names[j])

    plt.axvline(x=steps_train-1,
                linestyle=':',
                linewidth=2.5,
                color='black',
                label='Train/Test split')

    plt.title(f"Distribution error per feature ({biased_unbiased})", fontsize=18)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Distance", fontsize=16)

    plt.legend(fontsize=13, ncol=2)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
    
def plot_features_vs_time_groups(X, time_points, feature_cols, n_groups):
    n_times, n_obs, n_features = X.shape

    feature_indices = np.array_split(np.arange(n_features), n_groups)

    for g, idx in enumerate(feature_indices, start=1):
        n_in_group = len(idx)
        ncols = 2
        nrows = math.ceil(n_in_group / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        for k, j in enumerate(idx):
            ax = axes[k]

            for i in range(n_obs):
                y = X[:, i, j]
                mask = ~np.isnan(y)
                if mask.sum() > 0:
                    ax.plot(
                        np.array(time_points)[mask],
                        y[mask],
                        linewidth=2.0,      # thicker lines
                        alpha=0.8
                    )

            ax.set_title(feature_cols[j], fontsize=16)
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel("Feature value", fontsize=14)

            ax.tick_params(labelsize=12)
            ax.grid(alpha=0.3)

        for k in range(n_in_group, len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"Feature group {g}", fontsize=22)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        
##############33 evaluate bias plots: global max min, global range, smth else

