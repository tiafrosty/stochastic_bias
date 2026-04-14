import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

# connect my modules
from kernel_estimation import *

#  load real data, shape [time x Patients x Genes]
X_real_val = np.load("X_real_val.npy")
X_real_pred = np.load("X_real_pred.npy")

# load simulated data BIASED
X_biased_val = np.load("X_biased_val.npy")
X_biased_pred = np.load("X_biased_pred.npy")

# load simulated data UNBIASED
X_unbiased_val = np.load("X_unbiased_val.npy")
X_unbiased_pred = np.load("X_unbiased_pred.npy")

## set up the shape of the states
time_steps, n_paths, n_dim = X_real_val.shape
train_steps = int(0.8 * time_steps)


########3 1st evaluate accuracy of the predicted states
# biased
plot_distribution_error(X_biased_val, X_biased_pred, 'BIASED',  train_steps,  ['f1', 'f2', 'f3', 'f4', 'f5'])
plot_feature_histograms(X_biased_val, X_biased_pred, 'BIASED', time_points=[13, 14, 15])

evaluate_kernel(X_biased_val, X_biased_pred, plot=True, biased_unbiased='BIASED', steps_train=train_steps)

# real 
plot_distribution_error(X_real_val, X_real_pred, 'GSE30550 gene expr',  train_steps, ['f1', 'f2', 'f3', 'f4', 'f5'])
plot_feature_histograms(X_real_val, X_real_pred, 'GSE30550 gene expr', time_points=[13, 14, 15])

evaluate_kernel(X_real_val, X_real_pred, plot=True, biased_unbiased='GSE30550 gene expr', steps_train=train_steps)


#33333333333333333333 evalute the effect of bias and check global sprad 

visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased = evaluate_exploration(X_biased_val)

visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased = evaluate_exploration(X_unbiased_val)

visited_size_real, mean_real, var_real, global_min_real, global_max_real, global_range_real, new_bins_real = evaluate_exploration(X_real_val)

########3 PLOT ###############

all_bias_plots(
    X_biased_val,
    visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased,
    visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased,
    visited_size_real, mean_real, var_real, global_min_real, global_max_real, global_range_real, new_bins_real
)

aa = 1