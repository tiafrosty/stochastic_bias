import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

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

# here we loop over a time series
def evaluate_exploration(X_biased):
        
    # time series we work with         

    # ---------------- parameters ----------------
    time_steps, _, state_dim   = X_biased.shape
    # age is modelled up to a 100 years
    noise_scale   = 1.0 / state_dim
    # -------------------------------------------

    # Choose bin width in a scale-aware way (based on noise per step)
    # Std per coordinate per step is sqrt(noise_scale). Use that as a natural resolution.
    bin_width = np.sqrt(noise_scale)

    visited_biased = set()

    visited_size_biased = np.zeros(time_steps, dtype=int)

    # optional: store coverage increment per step (new bins discovered at each step)
    new_bins_biased = np.zeros(time_steps, dtype=int)

    # ------------------ NEW IDEA: global min/max 
    global_min_biased = np.zeros(time_steps)
    global_max_biased = np.zeros(time_steps)

    global_range_biased = np.zeros(time_steps)

    for t in range(time_steps):
        
        # ---- Idea 1 & 2: visited-set size + EXPLORATION curves ----
        bins_b = rows_to_tuples(bin_states(X_biased[t], bin_width))

        prev_b = len(visited_biased)

        visited_biased |= bins_b

        visited_size_biased[t] = len(visited_biased)

        new_bins_biased[t] = visited_size_biased[t] - prev_b
        
        # ------------------ NEW IDEA 3: track global min/max each step
        global_min_biased[t] = np.min(X_biased[t])
        global_max_biased[t] = np.max(X_biased[t])

        global_range_biased[t] = global_max_biased[t] - global_min_biased[t]
        # -----------------------------------------------------------------------------
        
    mean_biased   = X_biased.mean(axis=1)
    var_biased  = X_biased.var(axis=1, ddof=1)
        
    return visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased
    