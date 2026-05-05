import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

# connect my modules
from kernel_estimation import *

##
# worms
dataset_id = 'E-GEOD-12290' 

# mice, RNAseq data from blood and muscle , 6-30 weeks,  ~15-20 observationsper time  point
#dataset_id = 'GSE132741' 

# rats methylation, 1.5-35 months,  ~5 observationsper time  point
#dataset_id = 'GSE161141' 

#  load real data, shape [time x Patients x Genes]
X_real_val = np.load(f"X_real_val_30_{dataset_id}.npy")
#n_features = 30
#X_real_val= X_real_val[:, :, np.arange(n_features)]

# test and train size GLOBALLY FOR ALL DATA
T, n_obs, n_features = X_real_val.shape  # number of timepoints
train_steps = int(0.8 * T)


#######3 estimate probabilities
#X_real_t, X_real_t1 = make_pairs(X_real_val[:train_steps])
#X_real_pred = estimate_kernel(X_real_val, X_real_t, X_real_t1, T-train_steps)

#aa = 1
#X_real_pred = np.load("X_real_pred.npy")

# just a fix, remove later
#X_real_val = np.load(f"X_real_pred_large_{dataset_id}.npy")
#T, n_obs, n_features = X_real_val.shape  # number of timepoints
#train_steps = int(0.8 * T)


# for the simulated data, we match the data shape exactly with the real data.
# load simulated data BIASED

X_biased_val = np.load("X_biased_val.npy")[0:T][:, np.arange(n_obs)[:, None], np.arange(n_features)]

#X_biased_t, X_biased_t1 = make_pairs(X_biased_val[:train_steps])
#X_biased_pred = estimate_kernel_biased(X_biased_val, X_biased_t, X_biased_t1, T-train_steps)

#X_biased_pred = np.load("X_biased_pred.npy")

# load simulated data UNBIASED
X_unbiased_val = np.load("X_unbiased_val.npy")[0:T][:, np.arange(n_obs)[:, None], np.arange(n_features)]

#X_unbiased_pred = np.load("X_unbiased_pred.npy")
#X_unbiased_t, X_unbiased_t1 = make_pairs(X_unbiased_val[:train_steps])
#X_unbiased_pred = estimate_kernel_unbiased(X_unbiased_val, X_unbiased_t, X_unbiased_t1, T-train_steps)


## set up the shape of the states
#time_steps, n_paths, n_dim = X_biased_val.shape #time_steps, n_paths, n_dim = X_real_val.shape
#train_steps = int(0.8 * time_steps)


# real 
#plot_distribution_error(X_real_val, X_real_pred, 'GSE167665 gene expr',  train_steps, ['f1', 'f2', 'f3', 'f4', 'f5'])
#plot_feature_histograms(X_real_val, X_real_pred, 'GSE167665 gene expr', time_points=np.arange(train_steps, train_steps+1))

#evaluate_kernel(X_real_val, X_real_pred, plot=True, biased_unbiased='GSE167665 gene expr', steps_train=train_steps)

aa = 1
########3 1st evaluate accuracy of the predicted states
# biased
#plot_distribution_error(X_biased_val, X_biased_pred, 'BIASED',  train_steps,  ['f1', 'f2', 'f3', 'f4', 'f5'])
#plot_feature_histograms(X_biased_val, X_biased_pred, 'BIASED', time_points=np.arange(train_steps, train_steps+3))

#evaluate_kernel(X_biased_val, X_biased_pred, plot=True, biased_unbiased='BIASED', steps_train=train_steps)



#33333333333333333333 evalute the effect of bias and check global sprad 

visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased, entropy_biased, var_t_biased = evaluate_exploration(X_biased_val)

visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased,  entropy_unbiased, var_t_unbiased = evaluate_exploration(X_unbiased_val)

visited_size_real, mean_real, var_real, global_min_real, global_max_real, global_range_real, new_bins_real,  entropy_real, var_t_real = evaluate_exploration(X_real_val)

#bias_plots_without_real_data(
#    X_biased_val,
#    visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased, entropy_biased,
#   visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased, entropy_unbiased
#)

########3 PLOT ###############

all_bias_plots(dataset_id,
    X_biased_val,
    visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased, entropy_biased, var_t_biased,
    visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased,entropy_unbiased, var_t_unbiased,
   visited_size_real, mean_real, var_real, global_min_real, global_max_real, global_range_real, new_bins_real, entropy_real, var_t_real
)


####### Part 2: evaluate kenrel accuracy
#### we estimate kernel using 2 approaches: biased and unbiased.
### then from the last observed states in the train set, we generate trajectories using both apporahces.
## then validate using real data and see which one captures the dynamics better

seed = 150
future_steps = T - train_steps
X_real_t, X_real_t1 = make_pairs(X_real_val[:train_steps])
#X_real_pred = estimate_kernel(X_real_val, X_real_t, X_real_t1, T-train_steps)
X_pred_biased = estimate_kernel_biased(X_real_val, X_real_t, X_real_t1, future_steps, seed = seed)

X_pred_unbiased = estimate_kernel_unbiased(X_real_val, X_real_t, X_real_t1, future_steps, seed = seed)

#plot_increment_dependence_per_time(X_real_val[train_steps:T], X_pred_biased[train_steps:T], X_pred_unbiased[train_steps:T], plot=True, steps_train=train_steps)

#evaluate_kernel_3plots(X_real_val, X_pred_biased, X_pred_unbiased, plot=True, steps_train=train_steps)
evaluate_kernel_normalized(dataset_id, X_real_val, X_pred_biased, X_pred_unbiased, plot=True, steps_train=train_steps)
#evaluate_kernel_scaled(X_real_val, X_pred_biased, X_pred_unbiased, plot=True, steps_train=train_steps, full_process=True)

plot_feature_histograms_3plots(X_real_val, X_pred_biased, X_pred_unbiased, time_points=np.arange(train_steps+1,train_steps+4), bins=30)
aa = 1 