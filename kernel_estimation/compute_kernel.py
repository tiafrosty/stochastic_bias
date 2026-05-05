import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

import os

from geo_data_preprocess import *

def make_pairs(X_hist):
    state_dim = X_hist.shape[2]
    # X_hist has shape (time_steps, num_paths, state_dim)
    X_t  = X_hist[:-1].reshape(-1, state_dim)
    X_t1 = X_hist[1:].reshape(-1, state_dim)
    return X_t, X_t1


def estimate_kernel_biased(Xb_hist, Xb_t, Xb_t1, future_steps, seed = None):
    
    ########33 BIASED KERNEL  #####3
####3 learn the avergae change using linear regression
###############################3
    rng = np.random.default_rng(seed)
    X_design = np.hstack([np.ones((Xb_t.shape[0], 1)), Xb_t])   # shape (N, 1+d)

    # fit linear regression (least-squares) for all outputs
    B_hat, _, _, _ = np.linalg.lstsq(X_design, Xb_t1, rcond=None) # return x, res, rank, singular values
    # B_hat shape: (1 + state_dim, state_dim)

    intercept_b = B_hat[0]      # shape (state_dim,)
    A_b_hat     = B_hat[1:]     # shape (state_dim, state_dim)

    def m_hat(x):
        # x can be shape (n, d) or (d,)
        x = np.atleast_2d(x)
        # linear regression equation, substitute the learned parameters
        return x @ A_b_hat + intercept_b

    # Residuals and covariance
    residuals_b = Xb_t1 - m_hat(Xb_t)
    Sigma_b_hat = np.cov(residuals_b, rowvar=False)

    ############# generate future states

    T, num_paths, state_dim = Xb_hist.shape
    # Generate future trajectories starting from last observed state
    Xb_future = np.zeros((future_steps, num_paths, state_dim))
    Xb_current = Xb_hist[-1].copy()

    for t in range(future_steps):
        mean_next = m_hat(Xb_current)  # shape (num_paths, state_dim)
        noise_b = rng.multivariate_normal(
            mean=np.zeros(state_dim),
            cov=Sigma_b_hat,
            size=num_paths
        )
        Xb_current = mean_next + noise_b
        Xb_future[t] = np.clip(Xb_current, 0, 1)

    ############## ...DONE ######################

    # test and train size
    steps_train = int(0.8 * Xb_hist.shape[0]) # Xb_hist.shape[0]-future_steps
    #Xb_future = generate_future_states(Xu_hist, 0, Sigma_u_hat)

    #Xb_val = Xb_hist.copy() #[80:]
    #Xb_future_all = Xb_hist.copy()
    #Xb_future_all[:steps_train]+
    Xb_future_all = np.concatenate([Xb_hist[:steps_train], Xb_future])
    
    return Xb_future_all


def estimate_kernel_unbiased(Xu_hist, Xu_t, Xu_t1, future_steps, seed=None):
        # ============================================================
    # 1) UNBIASED KERNEL
    # Model:
    #   X_{t+1}^u ~ N(x, Sigma_u_hat)
    # so estimate Sigma from increments
    # ============================================================

    increments_u = Xu_t1 - Xu_t
    Sigma_u_hat = np.cov(increments_u, rowvar=False)

    print("Estimated covariance for unbiased kernel:")
    print(Sigma_u_hat)
    T, num_paths, state_dim = Xu_hist.shape

    # Generate future unbiased trajectories starting from last observed state
    def generate_future_states(Xu_hist, sigma_hat):
        rng = np.random.default_rng(seed)

        num_paths, state_dim = Xu_hist.shape[1], Xu_hist.shape[2]

        Xu_future = np.zeros((future_steps, num_paths, state_dim))
        Xu_current = Xu_hist[-1].copy()

        for h in range(future_steps):
            noise = rng.multivariate_normal(
                mean=np.zeros(state_dim),
                cov=sigma_hat,
                size=num_paths
            )

            Xu_current = Xu_current + noise
            Xu_future[h] = np.clip(Xu_current, 0, 1)

        return Xu_future
    
    # test and train size
    steps_train = int(0.8 * T)
    Xu_future = generate_future_states(Xu_hist, Sigma_u_hat)

    Xu_val = Xu_hist.copy() #[80:]
    Xu_future_all = Xu_hist.copy()
    Xu_future_all[steps_train:] = Xu_future

    return Xu_future_all


def evaluate_kernel(X_val, X_sim, plot=True, biased_unbiased = 'b', steps_train = 80):
    T, num_paths, d = X_val.shape
    
    mean_error = np.zeros(T)
    cov_error  = np.zeros(T)
    dist_error = np.zeros(T)
    
    for t in range(T):
        # ---- mean ----
        mu_val = X_val[t].mean(axis=0)
        mu_sim = X_sim[t].mean(axis=0)
        mean_error[t] = np.linalg.norm(mu_val - mu_sim)
        
        # ---- covariance ----
        cov_val = np.cov(X_val[t], rowvar=False)
        cov_sim = np.cov(X_sim[t], rowvar=False)
        cov_error[t] = np.linalg.norm(cov_val - cov_sim, ord='fro')
        
        # ---- distribution ----
        ## nah this is completely useless 
        #D = cdist(X_val[t], X_sim[t])
        #dist_error[t] = D.mean()
    
    if plot:
        t_axis = np.arange(T)

        # -------- error plot --------
        plt.figure(figsize=(10, 6))
        
        plt.plot(t_axis, mean_error, linewidth=3, label="Mean error")
        plt.plot(t_axis, cov_error, linewidth=3, label="Cov error")
        #plt.plot(t_axis, dist_error, linewidth=3, label="Dist error")
        
        # here we begin to predict
        plt.axvline(x=steps_train-1, linestyle=':', linewidth=2, color='black', label='Train/Test split')

        
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title(f"Errors over time ({biased_unbiased})", fontsize=18)
        
        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        # -------- per-feature means --------
        mean_val = X_val.mean(axis=1)  # shape (T, d)
        mean_sim = X_sim.mean(axis=1)

        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.arange(d))
        
        for j in range(d):
            plt.plot(mean_val[:, j],
             linestyle='-',
             linewidth=2.5,
             color=colors[j],
             label=f"real f{j+1}")
    
            plt.plot(mean_sim[:, j],
             linestyle='--',
             linewidth=2.5,
             color=colors[j],
             label=f"sim f{j+1}")
            
        # here we begin to predict
        plt.axvline(x=steps_train-1, linestyle=':', linewidth=2, color='black', label='Train/Test split')
        
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Mean", fontsize=16)
        plt.title(f"Per-feature mean trajectories ({biased_unbiased})", fontsize=18)
        
        plt.legend(fontsize=12, ncol=2)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # -------- Frobenius norm of (X_real - X_sim) for each t  --------
        frob_error = np.zeros(T)
        for t in range(T):
            frob_error[t] = np.linalg.norm(X_val[t] - X_sim[t], ord='fro')

        plt.figure(figsize=(10, 6))
        plt.plot(t_axis, frob_error, linewidth=3, label="Frobenius norm")
        plt.axvline(x=steps_train-1, linestyle=':', linewidth=2, color='black', label='Train/Test split')

        plt.xlabel("Time", fontsize=16)
        plt.ylabel("|X_val - X_sim|", fontsize=16)
        plt.title(f"Matrix difference over time ({biased_unbiased})", fontsize=18)

        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        # -------- individual trajectories: first 9 observations --------
        n_show = min(9, num_paths)

        fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True)
        axes = axes.flatten()

        colors = plt.cm.tab10(np.arange(d))

        for i in range(n_show):
            ax = axes[i]

            for j in range(d):
                ax.plot(t_axis,
                        X_val[:, i, j],
                        linestyle='--',
                        linewidth=1.8,
                        color=colors[j],
                        label=f"real f{j+1}" if i == 0 else None)

                ax.plot(t_axis,
                        X_sim[:, i, j],
                        linestyle='-',
                        linewidth=1.8,
                        color=colors[j],
                        label=f"sim f{j+1}" if i == 0 else None)

            ax.axvline(x=steps_train-1, linestyle=':', linewidth=2, color='black')
            ax.set_title(f"Observation {i+1}", fontsize=14)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=11)

        # hide unused axes if num_paths < 9
        for k in range(n_show, 9):
            axes[k].axis('off')

        fig.suptitle(f"First 9 observations: per-feature trajectories ({biased_unbiased})", fontsize=18)
        fig.supxlabel("Time", fontsize=16)
        fig.supylabel("Feature value", fontsize=16)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=11, bbox_to_anchor=(0.5, 0.97))

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
        
        return {
            "mean_error_avg": mean_error.mean(),
            "cov_error_avg": cov_error.mean(),
            "dist_error_avg": dist_error.mean()
        }
    

def evaluate_kernel_3plots(X_val, X_sim_biased, X_sim_unbiased, plot=True, steps_train=80, save_dir="plots"):
    T, num_paths, d = X_val.shape

    os.makedirs(save_dir, exist_ok=True)

    mean_error_biased = np.zeros(T)
    cov_error_biased  = np.zeros(T)
    frob_error_biased = np.zeros(T)

    mean_error_unbiased = np.zeros(T)
    cov_error_unbiased  = np.zeros(T)
    frob_error_unbiased = np.zeros(T)

    for t in range(T):
        mu_val = X_val[t].mean(axis=0)
        cov_val = np.cov(X_val[t], rowvar=False)

        mu_biased = X_sim_biased[t].mean(axis=0)
        cov_biased = np.cov(X_sim_biased[t], rowvar=False)

        mean_error_biased[t] = np.linalg.norm(mu_val - mu_biased)
        cov_error_biased[t] = np.linalg.norm(cov_val - cov_biased, ord="fro")
        frob_error_biased[t] = np.linalg.norm(X_val[t] - X_sim_biased[t], ord="fro")

        mu_unbiased = X_sim_unbiased[t].mean(axis=0)
        cov_unbiased = np.cov(X_sim_unbiased[t], rowvar=False)

        mean_error_unbiased[t] = np.linalg.norm(mu_val - mu_unbiased)
        cov_error_unbiased[t] = np.linalg.norm(cov_val - cov_unbiased, ord="fro")
        frob_error_unbiased[t] = np.linalg.norm(X_val[t] - X_sim_unbiased[t], ord="fro")

    if plot:
        t_axis = np.arange(T)
        colors = plt.cm.tab10(np.arange(d))

        # -------- error plot --------
        plt.figure(figsize=(10, 6))
        plt.plot(t_axis, mean_error_biased, linewidth=3, label="Mean error: predicted biased")
        plt.plot(t_axis, cov_error_biased, linewidth=3, label="Cov error: predicted biased")
        plt.plot(t_axis, mean_error_unbiased, linewidth=3, linestyle="--", label="Mean error: predicted unbiased")
        plt.plot(t_axis, cov_error_unbiased, linewidth=3, linestyle="--", label="Cov error: predicted unbiased")
        plt.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black", label="Train/Test split")
        plt.xlabel("Time"); plt.ylabel("Error"); plt.title("Errors over time")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "errors.png"), dpi=300)
        plt.show()

        # -------- per-feature means --------
        mean_val = X_val.mean(axis=1)
        mean_biased = X_sim_biased.mean(axis=1)
        mean_unbiased = X_sim_unbiased.mean(axis=1)

        plt.figure(figsize=(10, 6))
        for j in range(d):
            plt.plot(mean_val[:, j], "-", color=colors[j], label=f"real f{j+1}")
            plt.plot(mean_biased[:, j], "--", color=colors[j], label=f"predicted biased f{j+1}")
            plt.plot(mean_unbiased[:, j], ":", color=colors[j], label=f"predicted unbiased f{j+1}")
        plt.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black")
        plt.xlabel("Time"); plt.ylabel("Mean"); plt.title("Per-feature mean trajectories")
        plt.legend(ncol=2, fontsize=10); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "means.png"), dpi=300)
        plt.show()

        # -------- Frobenius norm --------
        plt.figure(figsize=(10, 6))
        plt.plot(t_axis, frob_error_biased, linewidth=3, label="Predicted biased")
        plt.plot(t_axis, frob_error_unbiased, linewidth=3, linestyle="--", label="Predicted unbiased")
        plt.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black")
        plt.xlabel("Time"); plt.ylabel(r"$\|X_{\mathrm{real}} - X_{\mathrm{pred}}\|_F$")
        plt.title("Matrix difference over time")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "frob.png"), dpi=300)
        plt.show()

        # -------- individual trajectories --------
        n_show = min(9, num_paths)
        fig, axes = plt.subplots(3, 3, figsize=(16, 14), sharex=True)
        axes = axes.flatten()

        for i in range(n_show):
            ax = axes[i]
            for j in range(d):
                ax.plot(t_axis, X_val[:, i, j], "-", color=colors[j])
                ax.plot(t_axis, X_sim_biased[:, i, j], "--", color=colors[j])
                ax.plot(t_axis, X_sim_unbiased[:, i, j], ":", color=colors[j])
            ax.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black")
            ax.set_title(f"Obs {i+1}")
            ax.grid(alpha=0.3)

        for k in range(n_show, 9):
            axes[k].axis("off")

        fig.suptitle("First 9 observations")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "trajectories.png"), dpi=300)
        plt.show()

    return {
        "mean_error_biased_avg": mean_error_biased.mean(),
        "cov_error_biased_avg": cov_error_biased.mean(),
        "frob_error_biased_avg": frob_error_biased.mean(),
        "mean_error_unbiased_avg": mean_error_unbiased.mean(),
        "cov_error_unbiased_avg": cov_error_unbiased.mean(),
        "frob_error_unbiased_avg": frob_error_unbiased.mean(),
    }

def state_increment_dependence_per_time(X):

    T, num_paths, d = X.shape
    corr_t = np.zeros(T-1)

    for t in range(T-1):
        X_t = X[t]          # (num_paths, d)
        dX  = X[t+1] - X[t]

        vals = []
        for j in range(d):
            c = np.corrcoef(X_t[:, j], dX[:, j])[0, 1]
            if not np.isnan(c):
                vals.append(c)

        corr_t[t] = np.mean(np.abs(vals)) if len(vals) > 0 else 0

    return corr_t


def plot_increment_dependence_per_time(X, X_sim_biased, X_sim_unbiased, plot=True, steps_train=80, save_dir="plots", full_process = False):
    T, num_paths, d = X.shape

    # plot
    corr_real = state_increment_dependence_per_time(X)
    corr_biased = state_increment_dependence_per_time(X_sim_biased)
    corr_unbiased = state_increment_dependence_per_time(X_sim_unbiased)


    t_axis = np.arange(len(corr_real))

    plt.figure(figsize=(10, 6))

    plt.plot(t_axis, corr_real, linewidth=3, label="real")
    plt.plot(t_axis, corr_biased, linewidth=3, label="predicted biased")
    plt.plot(t_axis, corr_unbiased, linewidth=3, linestyle="--", label="predicted unbiased")

    if full_process:
        plt.axvline(x=steps_train-1, linestyle=':', linewidth=2, color='black', label='Train/Test split')

    plt.xlabel("Time", fontsize=16)
    plt.ylabel(r"$\mathbb{E}[|\mathrm{corr}(X_t, X_{t+1}-X_t)|]$", fontsize=16)
    plt.title("State-dependence over time", fontsize=18)

    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_kernel_scaled(X_val, X_sim_biased, X_sim_unbiased, plot=True, steps_train=80, save_dir="plots", full_process = False):
    T, num_paths, d = X_val.shape
    os.makedirs(save_dir, exist_ok=True)


    corr_real = state_increment_dependence_per_time(X_val[steps_train-1:T])
    corr_biased = state_increment_dependence_per_time(X_sim_biased[steps_train-1:T])
    corr_unbiased = state_increment_dependence_per_time(X_sim_unbiased[steps_train-1:T])

    corr_error_b  = np.abs(corr_real - corr_biased)
    corr_error_u  = np.abs(corr_real - corr_unbiased)

    if not full_process:
        X_val = X_val[steps_train:T]
        X_sim_biased =  X_sim_biased[steps_train:T]
        X_sim_unbiased = X_sim_unbiased[steps_train:T]

        T, num_paths, d = X_val.shape

    mean_err_b = np.zeros(T)
    cov_err_b  = np.zeros(T)
    wd_b       = np.zeros(T)

    mean_err_u = np.zeros(T)
    cov_err_u  = np.zeros(T)
    wd_u       = np.zeros(T)


    for t in range(T):
        mu_val = X_val[t].mean(axis=0)
        cov_val = np.cov(X_val[t], rowvar=False)

        # biased
        mu_b = X_sim_biased[t].mean(axis=0)
        cov_b = np.cov(X_sim_biased[t], rowvar=False)

        mean_err_b[t] = np.linalg.norm(mu_val - mu_b) / (np.linalg.norm(mu_val) + 1e-8)
        cov_err_b[t]  = np.linalg.norm(cov_val - cov_b, 'fro') / (np.linalg.norm(cov_val, 'fro') + 1e-8)

        wd_b[t] = np.mean([
            wasserstein_distance(X_val[t,:,j], X_sim_biased[t,:,j]) /(np.linalg.norm(mu_val) + 1e-8)
            for j in range(d)
        ])

        # unbiased
        mu_u = X_sim_unbiased[t].mean(axis=0)
        cov_u = np.cov(X_sim_unbiased[t], rowvar=False)

        mean_err_u[t] = np.linalg.norm(mu_val - mu_u) / (np.linalg.norm(mu_val) + 1e-8)
        cov_err_u[t]  = np.linalg.norm(cov_val - cov_u, 'fro') / (np.linalg.norm(cov_val, 'fro') + 1e-8)

        wd_u[t] = np.mean([
            wasserstein_distance(X_val[t,:,j], X_sim_unbiased[t,:,j]) /(np.linalg.norm(mu_val) + 1e-8)
            for j in range(d)
        ])

    a = 0 
    b = np.max(np.concatenate([
    cov_err_b, cov_err_u,
    wd_u, wd_b,
    mu_u, mu_b,
    cov_u.flatten(), cov_b.flatten()
    ]))
    if plot:
        t_axis = np.arange(T)

        plt.figure(figsize=(10,6))

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        markers = ['o', 's', '^', 'D']

        # Mean
        plt.plot(t_axis, mean_err_b, color=colors[0], linewidth=3, linestyle='-',
                marker=markers[0], markersize=6, label="Mean (biased)")
        plt.plot(t_axis, mean_err_u, color=colors[0], linewidth=3, linestyle='--',
                marker=markers[0], markersize=6, label="Mean (unbiased)")

        # Cov
        plt.plot(t_axis, cov_err_b, color=colors[1], linewidth=3, linestyle='-',
                marker=markers[1], markersize=6, label="Cov (biased)")
        plt.plot(t_axis, cov_err_u, color=colors[1], linewidth=3, linestyle='--',
                marker=markers[1], markersize=6, label="Cov (unbiased)")

        # Wasserstein
        plt.plot(t_axis, wd_b, color=colors[2], linewidth=3, linestyle='-',
                marker=markers[2], markersize=6, label="WD (biased)")
        plt.plot(t_axis, wd_u, color=colors[2], linewidth=3, linestyle='--',
                marker=markers[2], markersize=6, label="WD (unbiased)")

        # Correlation
        plt.plot(t_axis, a + corr_error_b * (b - a), color=colors[3], linewidth=3, linestyle='-',
                marker=markers[3], markersize=6, label="Correlation (biased)")
        plt.plot(t_axis, a + corr_error_u * (b - a), color=colors[3], linewidth=3, linestyle='--',
                marker=markers[3], markersize=6, label="Correlation (unbiased)")

        if full_process:
            plt.axvline(x=steps_train-1, linestyle=':', color='black', linewidth=2)

        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title("Scaled errors", fontsize=18)

        plt.legend(fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "errors_scaled.png"), dpi=300)
        plt.show()




def evaluate_kernel_normalized(dataset_id, X_val, X_sim_biased, X_sim_unbiased, plot=True, steps_train=80, save_dir="plots", full_process = False):
    T, num_paths, d = X_val.shape
    os.makedirs(save_dir, exist_ok=True)


    corr_real = state_increment_dependence_per_time(X_val[steps_train-1:T])
    corr_biased = state_increment_dependence_per_time(X_sim_biased[steps_train-1:T])
    corr_unbiased = state_increment_dependence_per_time(X_sim_unbiased[steps_train-1:T])

    corr_error_b  = np.abs(corr_real - corr_biased)
    corr_error_u  = np.abs(corr_real - corr_unbiased)

    if not full_process:
        X_val = X_val[steps_train:T]
        X_sim_biased =  X_sim_biased[steps_train:T]
        X_sim_unbiased = X_sim_unbiased[steps_train:T]

        T, num_paths, d = X_val.shape

    mean_err_b = np.zeros(T)
    cov_err_b  = np.zeros(T)
    wd_b       = np.zeros(T)

    mean_err_u = np.zeros(T)
    cov_err_u  = np.zeros(T)
    wd_u       = np.zeros(T)


    for t in range(T):
        mu_val = X_val[t].mean(axis=0)
        cov_val = np.cov(X_val[t], rowvar=False)

        # biased
        mu_b = X_sim_biased[t].mean(axis=0)
        cov_b = np.cov(X_sim_biased[t], rowvar=False)

        mean_err_b[t] = np.linalg.norm(mu_val - mu_b) 
        cov_err_b[t]  = np.linalg.norm(cov_val - cov_b, 'fro')

        wd_b[t] = np.mean([
            wasserstein_distance(X_val[t,:,j], X_sim_biased[t,:,j]) 
            for j in range(d)
        ])

        # unbiased
        mu_u = X_sim_unbiased[t].mean(axis=0)
        cov_u = np.cov(X_sim_unbiased[t], rowvar=False)

        mean_err_u[t] = np.linalg.norm(mu_val - mu_u) 
        cov_err_u[t]  = np.linalg.norm(cov_val - cov_u, 'fro')

        wd_u[t] = np.mean([
            wasserstein_distance(X_val[t,:,j], X_sim_unbiased[t,:,j]) 
            for j in range(d)
        ])

    a = 0 
    b = np.max(np.concatenate([
    cov_err_b, cov_err_u,
    wd_u, wd_b,
    mu_u, mu_b,
    cov_u.flatten(), cov_b.flatten()
    ]))
    if plot:
        t_axis = np.arange(T)

        plt.figure(figsize=(10,6))

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        markers = ['o', 's', '^', 'D']

        # Mean
        plt.plot(t_axis, mean_err_b, color=colors[0], linewidth=3, linestyle='-',
                marker=markers[0], markersize=6, label="Mean (biased)")
        plt.plot(t_axis, mean_err_u, color=colors[0], linewidth=3, linestyle='--',
                markersize=6, label="Mean (unbiased)")

        # Cov
        plt.plot(t_axis, cov_err_b, color=colors[1], linewidth=3, linestyle='-',
                marker=markers[1], markersize=6, label="Cov (biased)")
        plt.plot(t_axis, cov_err_u, color=colors[1], linewidth=3, linestyle='--',
                 markersize=6, label="Cov (unbiased)")

        # Wasserstein
        plt.plot(t_axis, wd_b, color=colors[2], linewidth=3, linestyle='-',
                marker=markers[2], markersize=6, label="WD (biased)")
        plt.plot(t_axis, wd_u, color=colors[2], linewidth=3, linestyle='--',
                 markersize=6, label="WD (unbiased)")

        # Correlation
        plt.plot(t_axis, corr_error_b, color=colors[3], linewidth=3, linestyle='-',
                marker=markers[3], markersize=6, label="Correlation (biased)")
        plt.plot(t_axis, corr_error_u, color=colors[3], linewidth=3, linestyle='--',
                markersize=6, label="Correlation (unbiased)")

        if full_process:
            plt.axvline(x=steps_train-1, linestyle=':', color='black', linewidth=2)

        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title("Scaled errors", fontsize=18)

        plt.legend(
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0
        )
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"errors_scaled_{dataset_id}.png"), dpi=300)
        plt.show()