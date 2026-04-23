import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd
import math


import numpy as np
import matplotlib.pyplot as plt
import math

# connect my modules
from kernel_estimation import *

def plot_features_vs_time_groups(X, time_points, feature_cols, n_groups=9):
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


omics_data = pd.read_csv('Lifespan_Study_selected_data.csv')[:496]#.sort_values(by="Days_grown_Udays", ascending=True)

###################################################333#
#   DATA PRE-PROCESS BEGINS #######################  
# sort by time, remove all the columns containing time
time_col = "Days_grown_Udays"

# drop all other time-related columns
cols_to_drop = [col for col in omics_data.columns
                if ("day" in col.lower() or "time" in col.lower() or "passage" in col.lower())
                and col != time_col]
# also drop non-numeric columns
omics_data_clean = omics_data.drop(columns=cols_to_drop).select_dtypes(include=[np.number]).dropna(subset=[time_col])

# RESHAPE INTO AN ARRAY TIME X OBSERVATIONS X FEATURES
# WE COUNT BY FULL DAYS, TO TIME IS ROUNDED TILL THE FULL NUMBER 
def reshape_longitudinal_by_replicate(
    df,
    time_col="Days_grown_Udays",
    cell_col="Cell_line",
    rep_col="Replicate_line",
    feature_cols=None
):
    """
    Reshape to X with shape:
        (time_points, observations, features)

    where observation = one (cell line, replicate) trajectory.

    Rules:
    - time point = floor(Days_grown_Udays)
    - rows are indexed by (cell line, replicate)
    - same row across time = same trajectory
    - if multiple rows exist for the same (cell line, replicate, day), keep the first
    - missing combinations are filled with NaN
    """
    df = df.copy()

    df = df.dropna(subset=[time_col, cell_col, rep_col])
    df["time_point"] = np.floor(df[time_col]).astype(int)

    if feature_cols is None:
        exclude = {time_col, "time_point", cell_col, rep_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    keep_cols = [cell_col, "time_point", rep_col] + feature_cols
    df = df[keep_cols].copy()

    # define trajectory ID = (cell line, replicate)
    df["obs_id"] = df[cell_col].astype(str) + "_" + df[rep_col].astype(str)
    
    df = df[~df.isna().any(axis=1)]

    # if duplicates exist within same trajectory and day, keep first
    df = (
        df.sort_values(["obs_id", "time_point"])
          .drop_duplicates(subset=["obs_id", "time_point"], keep="first")
    )

    obs_ids = sorted(df["obs_id"].unique())
    time_points = sorted(df["time_point"].unique())

    n_times = len(time_points)
    n_obs = len(obs_ids)
    n_features = len(feature_cols)

    time_to_idx = {t: i for i, t in enumerate(time_points)}
    obs_to_idx = {obs: i for i, obs in enumerate(obs_ids)}

    X = np.full((n_times, n_obs, n_features), np.nan)

    for _, row in df.iterrows():
        t_idx = time_to_idx[row["time_point"]]
        o_idx = obs_to_idx[row["obs_id"]]
        X[t_idx, o_idx, :] = row[feature_cols].to_numpy(dtype=float)

    return X, time_points, obs_ids, feature_cols, df

feature_cols =  [
    "Horvath1_PanTissue_clock_Uyears",
    "PhenoAge_clock_Uyears",
    "GrimAge_clock_Uyears",
    "DunedinPACE_rate_of_aging_Uau",
    "Cell_diameter_Umicrometer",
    "Cell_volume_Ucubed_micrometers",
    "Percent_dead_cells",
    "Population_doublings_Utotal_divisions",
    "Mitotic_age_Udoublings",
    "DNAmTL_Uau",
]

# select like 10 main features 
#new_clean_data = 

#remove nan rows
#new_clean_data = omics_data[~omics_data.isna().any(axis=1)]

X, time_points, obs_ids, feature_cols, df_long = reshape_longitudinal_by_replicate(
    omics_data,
    time_col="Days_grown_Udays",
    cell_col="Cell_line",
    rep_col="Replicate_line",
    feature_cols= feature_cols
)

##########################################
####  DATA PRE-PRCOESS DONE #############


plot_features_vs_time_groups(X, time_points, feature_cols, n_groups=9)

train_steps = 80
#######3 estimate probabilities
X_real_t, X_real_t1 = make_pairs(X[:train_steps])

X_real = X.copy()
X_future = estimate_kernel(X_real, X_real_t, X_real_t1, train_steps)

aa = 1

