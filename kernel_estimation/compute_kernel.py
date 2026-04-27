import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

def make_pairs(X_hist):
    state_dim = X_hist.shape[2]
    # X_hist has shape (time_steps, num_paths, state_dim)
    X_t  = X_hist[:-1].reshape(-1, state_dim)
    X_t1 = X_hist[1:].reshape(-1, state_dim)
    return X_t, X_t1


def estimate_kernel_biased(Xb_hist, Xb_t, Xb_t1, future_steps):
    
    ########33 BIASED KERNEL  #####3
####3 learn the avergae change using linear regression
###############################3

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
        noise_b = np.random.multivariate_normal(
            mean=np.zeros(state_dim),
            cov=Sigma_b_hat,
            size=num_paths
        )
        Xb_current = mean_next + noise_b
        Xb_future[t] = Xb_current

    ############## ...DONE ######################

    # test and train size
    steps_train = int(0.8 * Xb_hist.shape[0]) # Xb_hist.shape[0]-future_steps
    #Xb_future = generate_future_states(Xu_hist, 0, Sigma_u_hat)

    #Xb_val = Xb_hist.copy() #[80:]
    #Xb_future_all = Xb_hist.copy()
    #Xb_future_all[:steps_train]+
    Xb_future_all = np.concatenate([Xb_hist[:steps_train], Xb_future])
    
    return Xb_future_all


def estimate_kernel_unbiased(Xu_hist, Xu_t, Xu_t1, future_steps):
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
        Xu_future = np.zeros((future_steps, num_paths, state_dim))
        Xu_current = Xu_hist[-1].copy()

        for h in range(future_steps):
            Xu_next = np.zeros_like(Xu_current)
            
            for i in range(num_paths):
                np.random.seed(i)
                Xu_next[i] = np.random.multivariate_normal(
                    mean=Xu_current[i],
                    cov=sigma_hat
                )
            
            Xu_current = Xu_next
            Xu_future[h] = Xu_current
            
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
    

def evaluate_kernel_3plots(X_val, X_sim_biased, X_sim_unbiased, plot=True, steps_train=80):
    T, num_paths, d = X_val.shape

    mean_error_biased = np.zeros(T)
    cov_error_biased  = np.zeros(T)
    frob_error_biased = np.zeros(T)

    mean_error_unbiased = np.zeros(T)
    cov_error_unbiased  = np.zeros(T)
    frob_error_unbiased = np.zeros(T)

    for t in range(T):
        # ---- real ----
        mu_val = X_val[t].mean(axis=0)
        cov_val = np.cov(X_val[t], rowvar=False)

        # ---- biased ----
        mu_biased = X_sim_biased[t].mean(axis=0)
        cov_biased = np.cov(X_sim_biased[t], rowvar=False)

        mean_error_biased[t] = np.linalg.norm(mu_val - mu_biased)
        cov_error_biased[t] = np.linalg.norm(cov_val - cov_biased, ord="fro")
        frob_error_biased[t] = np.linalg.norm(X_val[t] - X_sim_biased[t], ord="fro")

        # ---- unbiased ----
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

        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.title("Errors over time", fontsize=18)

        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # -------- per-feature means --------
        mean_val = X_val.mean(axis=1)
        mean_biased = X_sim_biased.mean(axis=1)
        mean_unbiased = X_sim_unbiased.mean(axis=1)

        plt.figure(figsize=(10, 6))

        for j in range(d):
            plt.plot(
                mean_val[:, j],
                linestyle="-",
                linewidth=2.5,
                color=colors[j],
                label=f"real f{j+1}"
            )

            plt.plot(
                mean_biased[:, j],
                linestyle="--",
                linewidth=2.5,
                color=colors[j],
                label=f"predicted biased f{j+1}"
            )

            plt.plot(
                mean_unbiased[:, j],
                linestyle=":",
                linewidth=2.5,
                color=colors[j],
                label=f"predicted unbiased f{j+1}"
            )

        plt.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black", label="Train/Test split")

        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Mean", fontsize=16)
        plt.title("Per-feature mean trajectories", fontsize=18)

        plt.legend(fontsize=10, ncol=2)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # -------- Frobenius norm --------
        plt.figure(figsize=(10, 6))

        plt.plot(t_axis, frob_error_biased, linewidth=3, label="Predicted biased")
        plt.plot(t_axis, frob_error_unbiased, linewidth=3, linestyle="--", label="Predicted unbiased")

        plt.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black", label="Train/Test split")

        plt.xlabel("Time", fontsize=16)
        plt.ylabel(r"$\|X_{\mathrm{real}} - X_{\mathrm{pred}}\|_F$", fontsize=16)
        plt.title("Matrix difference over time", fontsize=18)

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

        for i in range(n_show):
            ax = axes[i]

            for j in range(d):
                ax.plot(
                    t_axis,
                    X_val[:, i, j],
                    linestyle="-",
                    linewidth=1.8,
                    color=colors[j],
                    label=f"real f{j+1}" if i == 0 else None
                )

                ax.plot(
                    t_axis,
                    X_sim_biased[:, i, j],
                    linestyle="--",
                    linewidth=1.8,
                    color=colors[j],
                    label=f"predicted biased f{j+1}" if i == 0 else None
                )

                ax.plot(
                    t_axis,
                    X_sim_unbiased[:, i, j],
                    linestyle=":",
                    linewidth=1.8,
                    color=colors[j],
                    label=f"predicted unbiased f{j+1}" if i == 0 else None
                )

            ax.axvline(x=steps_train - 1, linestyle=":", linewidth=2, color="black")
            ax.set_title(f"Observation {i+1}", fontsize=14)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=11)

        for k in range(n_show, 9):
            axes[k].axis("off")

        fig.suptitle("First 9 observations: per-feature trajectories", fontsize=18)
        fig.supxlabel("Time", fontsize=16)
        fig.supylabel("Feature value", fontsize=16)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.97)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    return {
        "mean_error_biased_avg": mean_error_biased.mean(),
        "cov_error_biased_avg": cov_error_biased.mean(),
        "frob_error_biased_avg": frob_error_biased.mean(),
        "mean_error_unbiased_avg": mean_error_unbiased.mean(),
        "cov_error_unbiased_avg": cov_error_unbiased.mean(),
        "frob_error_unbiased_avg": frob_error_unbiased.mean(),
    }