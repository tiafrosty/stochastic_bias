import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

# connect my modules
from kernel_estimation import *

                        
# ---------------- parameters ----------------
state_dim   = 5
# age is modelled up to a 100 years
time_steps  = 16 # 100
num_paths   = 14

bias_strength = 40 / time_steps # 400
drift_strength = 1 #400/time_steps
noise_scale   = 1.0 / state_dim
# -------------------------------------------
#### damage
D_t = 0

# identity 2D matrix
Sigma = noise_scale * np.eye(state_dim)
X_0 = np.random.randn(num_paths, state_dim)

X_biased   = X_0.copy()
X_unbiased = X_0.copy()

def compute_A_from_X(X, bias_strength):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    z  = np.abs(mu) / (sd + 1e-12)
    A_t = bias_strength * np.diag(z)
    return A_t

# Discretization
def bin_states(X, bin_width):
    """
    Map continuous states X (n, d) to discrete bin indices (n, d) by rounding to grid.
    """
    return np.floor(X / bin_width).astype(np.int64)

def rows_to_tuples(B):
    """
    Convert (n, d) integer array to a set of tuples for hashing/unique counting.
    """
    return set(map(tuple, B))
# ------------------------------------------------------------

# ------------------ Idea 4: store full trajectories ------------------
Xb_hist = np.zeros((time_steps, num_paths, state_dim))
Xu_hist = np.zeros((time_steps, num_paths, state_dim))

for t in range(time_steps):
    noise = np.random.multivariate_normal(
        mean=np.zeros(state_dim),
        cov=Sigma,
        size=num_paths
    )

    A_t = compute_A_from_X(X_biased, bias_strength)

    # biased process
    mu = X_biased.mean(axis=0)              
    sd = X_biased.std(axis=0, ddof=1)
    b_t = drift_strength * (mu / (sd + 1e-12))  # drift direction
    # damage (is not needed probably)
    D_t += t

    X_biased = X_biased - (X_biased @ A_t.T) + b_t + noise
    ##X_biased = X_biased - (X_biased @ A_t.T) + noise
    
    ######################
    # unbiased reference
    X_unbiased = X_unbiased + noise

    # idea 4: store full trajectories
    Xb_hist[t] = X_biased
    Xu_hist[t] = X_unbiased
    

##############  EXPERIMENT 2 ########################33
############### TRANSITION PROBABILITY ESTIMATION #####
tr_step =  int(0.8 * time_steps)



future_steps = time_steps-tr_step

# build (X_t, X_{t+1}) pairs
def make_pairs(X_hist):
    # X_hist has shape (time_steps, num_paths, state_dim)
    X_t  = X_hist[:-1].reshape(-1, state_dim)
    X_t1 = X_hist[1:].reshape(-1, state_dim)
    return X_t, X_t1

# take 80% for training, leave 20% for validation
Xu_t, Xu_t1 = make_pairs(Xu_hist[:tr_step])
Xb_t, Xb_t1 = make_pairs(Xb_hist[:tr_step])

aa = 1

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

# Generate future unbiased trajectories starting from last observed state
def generate_future_states(Xu_hist, x_mean, sigma_hat):
    Xu_future = np.zeros((future_steps, num_paths, state_dim))
    Xu_current = Xu_hist[-1].copy()

    for h in range(future_steps):
        Xu_next = np.zeros_like(Xu_current)
        
        for i in range(num_paths):
            Xu_next[i] = np.random.multivariate_normal(
                mean=Xu_current[i],
                cov=sigma_hat
            )
        
        Xu_current = Xu_next
        Xu_future[h] = Xu_current
        
    return Xu_future
    
    
# test and train size
steps_train = tr_step
Xu_future = generate_future_states(Xu_hist, 0, Sigma_u_hat)

Xu_val = Xu_hist.copy() #[80:]
Xu_future_all = Xu_hist.copy()
Xu_future_all[steps_train:] = Xu_future

# save 
np.save("X_unbiased_val.npy", Xu_val)
np.save("X_unbiased_pred.npy", Xu_future_all)

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
steps_train = tr_step
#Xb_future = generate_future_states(Xu_hist, 0, Sigma_u_hat)

Xb_val = Xb_hist.copy() #[80:]
Xb_future_all = Xb_hist.copy()
Xb_future_all[steps_train:] = Xb_future

# save 
np.save("X_biased_val.npy", Xb_val)
np.save("X_biased_pred.npy", Xb_future_all)


plot_feature_histograms(Xu_val, Xu_future_all, 'UNBIASED', time_points=[13, 14, 15])
plot_feature_histograms(Xb_val, Xb_future_all, 'BIASED', time_points=[13, 14, 15])

# check accuracy    

plot_distribution_error(Xb_val, Xb_future_all, 'BIASED',  steps_train, ['f1', 'f2', 'f3', 'f4', 'f5'])
kernel_acc_b = evaluate_kernel(Xb_val, Xb_future_all, True, 'BIASED', steps_train)

aa = 1



 