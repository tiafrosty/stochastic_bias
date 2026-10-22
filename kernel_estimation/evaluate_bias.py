import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd
from scipy.stats import linregress

# try clustering as discretization

from sklearn.cluster import KMeans


from scipy.stats import ttest_rel

from common.common_modules import *
from sklearn.metrics import mean_squared_error


def curve_distance(model_curves, real_curves):
    model_curves = np.stack(model_curves)
    real_curves = np.stack(real_curves)

    #aa = mean_squared_error(model_curves, real_curves)

    return np.mean((model_curves - real_curves) ** 2, axis=1)


def compare_curve_distances(unbiased_curves, biased_curves_dict, real_curves):
    results = {}

    dist_unbiased = curve_distance(unbiased_curves, real_curves)

    for bias, biased_curves in biased_curves_dict.items():
        dist_biased = curve_distance(biased_curves, real_curves)

        t_stat, p_val = ttest_rel(
            dist_unbiased,
            dist_biased,
            alternative="greater"
        )

        results[bias] = {
            "mean_dist_unbiased": np.mean(dist_unbiased),
            "mean_dist_biased": np.mean(dist_biased),
            "t_stat": t_stat,
            "p_value": p_val
        }

    return results



def inaccessible_states_over_time(X, baseline_steps=2):
    T, n_obs, d = X.shape

    noise_scale   = 1.0 / d
    # -------------------------------------------

    # Choose bin width in a scale-aware way (based on noise per step)
    # Std per coordinate per step is sqrt(noise_scale). Use that as a natural resolution.
    bin_width = np.sqrt(noise_scale)/d
    B_all = []
    for t in range(T):
        B = np.floor(X[t] / bin_width).astype(int)
        B_all.append([tuple(row) for row in B])

    baseline_states = set()
    for t in range(baseline_steps):
        baseline_states.update(B_all[t])

    inaccessible_count = np.zeros(T)
    inaccessible_fraction = np.zeros(T)
    inaccessible_states = []

    for t in range(T):
        current_states = set(B_all[t])

        lost = baseline_states - current_states

        inaccessible_states.append(lost)
        inaccessible_count[t] = len(lost)
        inaccessible_fraction[t] = len(lost) / len(baseline_states)

    return inaccessible_count, inaccessible_fraction, inaccessible_states



n_grid = 50
# Discretization idea 2
def discretize(X):
    return np.round(X * (n_grid - 1)).astype(int)

def rows_to_tuples(B):
    """
    Convert (n, d) integer array to a set of tuples for hashing/unique counting.
    """
    return set(map(tuple, B))
# ------------------------------------------------------------


def cumulative_feature_range(X):
    """
    X shape: time x observations x features

    Returns:
    - range_curve: average cumulative explored range across features
    - feature_ranges_time: per-feature cumulative ranges over time
    """

    T, n_obs, d = X.shape

    running_min = np.full(d, np.inf)
    running_max = np.full(d, -np.inf)

    range_curve = np.zeros(T)
    feature_ranges_time = np.zeros((T, d))

    for t in range(T):

        running_min = np.minimum(running_min, X[t].min(axis=0))
        running_max = np.maximum(running_max, X[t].max(axis=0))

        feature_ranges = running_max - running_min

        feature_ranges_time[t] = feature_ranges
        range_curve[t] = feature_ranges.mean()

    return range_curve, feature_ranges_time


def evaluate_exploration_full(X_biased):

    time_steps, n_obs, state_dim = X_biased.shape

    noise_scale = 1.0 / state_dim
    bin_width = np.sqrt(noise_scale) / state_dim

    # Each individual has its own set of visited profiles
    visited = [set() for _ in range(n_obs)]

    visited_size = np.zeros(time_steps, dtype=int)
    new_bins = np.zeros(time_steps, dtype=int)
    repeat_rate = np.zeros(time_steps)

    for t in range(time_steps):

        # Discretize the full methylation matrix
        B = bin_states(X_biased[t], bin_width)

        total_new = 0

        for i in range(n_obs):

            # Entire CpG profile is one state
            current_state = tuple(B[i, :])

            if current_state not in visited[i]:
                total_new += 1

            visited[i].add(current_state)

        if t > 0:
            visited_size[t] = sum(len(v) for v in visited)
            new_bins[t] = total_new

            # Total number of profile observations up to time t
            total_visits = (t + 1) * n_obs

            repeat_rate[t] = (
                1.0 - visited_size[t] / total_visits
            )

    return visited_size, new_bins, repeat_rate

# here we loop over a time series
def evaluate_exploration(X_biased):
        
    # time series we work with         

    # ---------------- parameters ----------------
    time_steps, n_obs, state_dim   = X_biased.shape
    # age is modelled up to a 100 years
    noise_scale   = 1.0 / state_dim
    # -------------------------------------------

    # Choose bin width in a scale-aware way (based on noise per step)
    # Std per coordinate per step is sqrt(noise_scale). Use that as a natural resolution.
    bin_width = np.sqrt(noise_scale)/state_dim # added /10

    visited_biased =  [set() for _ in range(n_obs)]

    visited_size_biased = np.zeros(time_steps, dtype=int)

    # optional: store coverage increment per step (new bins discovered at each step)
    new_bins_biased = np.zeros(time_steps, dtype=int)

    # ------------------ NEW IDEA: global min/max 
    ##### not global tho but avereged over observations 
    global_min_biased = np.zeros(time_steps)
    global_max_biased = np.zeros(time_steps)

    global_range_biased = np.zeros(time_steps)

    entropy_time_steps = np.zeros(time_steps)


    mu_ref = X_biased[0].mean(axis=0)  # fixed reference at t=0

    # population mean vector across all time points and observations
    mean_dist_to_mu = np.zeros(time_steps)

    # repeat rate
    total_visits_biased = np.zeros(time_steps, dtype=int)
    repeat_rate_biased = np.zeros(time_steps)

    # average mean'
    global_mean_biased = np.zeros(time_steps)

    occupied_fraction_biased = np.zeros(time_steps)

    reachable_log_volume = np.zeros(time_steps)

    running_min = np.full(state_dim, np.inf)
    running_max = np.full(state_dim, -np.inf)
    eps = 1e-12

    mean_residence_time = np.zeros(time_steps)
    prev_B = None
    run_lengths = np.ones((n_obs, state_dim))

    # temporal corrleation
    temporal_corr_biased = np.zeros(time_steps)
    temporal_corr_biased[0] = np.nan

    for t in range(time_steps):

        if t > 0:
            x_prev = X_biased[t - 1].reshape(-1)
            x_cur = X_biased[t].reshape(-1)

            temporal_corr_biased[t] = np.corrcoef(x_prev, x_cur)[0, 1]
        
        B = bin_states(X_biased[t], bin_width)  # shape: (n_obs, n_features) # was like thaaaat
        # try my new discretization
        #B = discretize(X_biased[t])

        total_new = 0
        
        for i in range(n_obs):
        # feature-wise states for observation i
            current_states_i = set(B[i, :])

            prev_i = len(visited_biased[i])

            visited_biased[i] |= current_states_i

            states_i = len(visited_biased[i]) - prev_i

            total_new += states_i
        
        #new_bins_biased[t] = total_new # was like this
        if t > 0: # maybe comment this out, so the visited states at time 0 is not 0 but how many we visited in the first slot
            # no should be 0
            
            # state growth
            visited_size_biased[t] = sum(len(v) for v in visited_biased)
            # exploration rate
            new_bins_biased[t] = total_new
        
        # ---- Idea 1 & 2: visited-set size + EXPLORATION curves ---
        # # Obs! I chaneghd discreteization method
        #bins_b = rows_to_tuples(bin_states(X_biased[t], bin_width)) # W was like this
        # new one
        
        # ------------------ NEW IDEA 3: track global min/max each step
        ### change and take average over observations then min and max
        global_min_biased[t] = np.min(np.mean(X_biased[t], axis = 0))
        global_max_biased[t] = np.max(np.mean(X_biased[t], axis = 0))

        global_range_biased[t] = global_max_biased[t] - global_min_biased[t]
        # ------------------------------------------------------------------


        ##### mean 
        mean_dist_to_mu[t] = np.mean(
        np.linalg.norm(X_biased[t] - mu_ref, axis=1)
        )

        total_visits = (t + 1) * n_obs * state_dim

        # repeat visits rate
        total_visits_biased[t] = total_visits

        repeat_rate_biased[t] = (
            1.0 - visited_size_biased[t] / total_visits_biased[t]
        )

        # average mean
        global_mean_biased[t] = np.mean(X_biased[t])



        
    mean_biased   = X_biased.mean(axis=1)
    var_biased  = X_biased.var(axis=1, ddof=1)

    variation_t = np.mean((X_biased[1:] - X_biased[:-1])**2, axis=(1,2))  
    # fix the first element in exploration
    #visited_size_biased[0] = visited_size_biased[1]
        
    return visited_size_biased, new_bins_biased, repeat_rate_biased
    

def evaluate_exploration_clustered(X, kmeans):

    time_steps, n_obs, state_dim = X.shape

    visited = [set() for _ in range(n_obs)]

    visited_size = np.zeros(time_steps, dtype=int)
    new_states = np.zeros(time_steps, dtype=int)
    repeat_rate = np.zeros(time_steps)

    for t in range(time_steps):

        labels = kmeans.predict(X[t])  # shape: n_obs

        total_new = 0

        for i in range(n_obs):

            prev_i = len(visited[i])

            visited[i].add(labels[i])

            total_new += len(visited[i]) - prev_i

        new_states[t] = total_new
        visited_size[t] = sum(len(v) for v in visited)

        total_visits = (t + 1) * n_obs

        repeat_rate[t] = 1.0 - visited_size[t] / total_visits

    return visited_size, new_states, repeat_rate