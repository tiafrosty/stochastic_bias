from pathlib import Path
import sys
from scipy.stats import ttest_rel
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

from tqdm import tqdm

# connect my modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_estimation import *
import stoch_sim_CHECK




models_all = ['elastic_net', 'ridge', 'ridge', 'elastic_net']

bias_terms = ['FSMR', 'LMR', 'NMR', 'FS']



def run_comp_study_bias(
    dataset_id,
    n_features,
    evaluate_bias=True,
    plot_mean_trajectories=False, 
    do_t_test = False
):

    save_dir = Path("all_data_files")

    n_bootstrap = 100

    all_visited_biased = {bias: [] for bias in bias_terms}
    all_new_biased = {bias: [] for bias in bias_terms}
    all_repeat_biased = {bias: [] for bias in bias_terms}

    all_visited_unbiased = []
    all_visited_real = []

    all_new_unbiased = []
    all_new_real = []

    all_repeat_unbiased = []
    all_repeat_real = []

    # Store mean trajectories per bootstrap run
    all_mean_traj_biased = {bias: [] for bias in bias_terms}
    all_mean_traj_unbiased = []
    all_mean_traj_real = []

    for i in tqdm(range(n_bootstrap)):

        n_sel_features = n_features
        X_real_val = np.load(
            save_dir / f"X_real_val_{n_sel_features}_{dataset_id}_boot{i}.npy"
        )

        T, n_obs, n_features_cur = X_real_val.shape

        # Average over observations and features: shape [T]
        if plot_mean_trajectories:
            mean_traj_real = np.mean(X_real_val, axis=(1, 2))
            all_mean_traj_real.append(mean_traj_real)

        if evaluate_bias:
            visited_size_real, new_bins_real, repeat_rate_real = evaluate_exploration(
                X_real_val
            )

            all_visited_real.append(visited_size_real)
            all_new_real.append(new_bins_real)
            all_repeat_real.append(repeat_rate_real)

        for cur_bias in bias_terms:

            stoch_sim_CHECK.run_entire_simulation(
                dataset_id,
                n_sel_features,
                which_bias=cur_bias
            )

            X_biased_val = np.load(save_dir / f"X_biased_val_{cur_bias}.npy")[0:T][
                :, np.arange(n_obs)[:, None], np.arange(n_features_cur)
            ]

            X_unbiased_val = np.load(save_dir / "X_unbiased_val.npy")[0:T][
                :, np.arange(n_obs)[:, None], np.arange(n_features_cur)
            ]

            if plot_mean_trajectories:
                mean_traj_biased = np.mean(X_biased_val, axis=(1, 2))
                all_mean_traj_biased[cur_bias].append(mean_traj_biased)

                # Store unbiased only once per bootstrap
                if cur_bias == bias_terms[0]:
                    mean_traj_unbiased = np.mean(X_unbiased_val, axis=(1, 2))
                    all_mean_traj_unbiased.append(mean_traj_unbiased)

            if evaluate_bias:
                visited_size_biased, new_bins_biased, repeat_rate_biased = evaluate_exploration(
                    X_biased_val
                )

                all_visited_biased[cur_bias].append(visited_size_biased)
                all_new_biased[cur_bias].append(new_bins_biased)
                all_repeat_biased[cur_bias].append(repeat_rate_biased)

                if cur_bias == bias_terms[0]:
                    visited_size_unbiased, new_bins_unbiased, repeat_rate_unbiased = evaluate_exploration(
                        X_unbiased_val
                    )

                    all_visited_unbiased.append(visited_size_unbiased)
                    all_new_unbiased.append(new_bins_unbiased)
                    all_repeat_unbiased.append(repeat_rate_unbiased)

    
    if do_t_test:
        significance_results = {}

        significance_results["visited_state_growth"] = compare_curve_distances(
            all_visited_unbiased,
            all_visited_biased,
            all_visited_real
        )

        significance_results["exploration_rate"] = compare_curve_distances(
            all_new_unbiased,
            all_new_biased,
            all_new_real
        )

        significance_results["repeat_rate"] = compare_curve_distances(
            all_repeat_unbiased,
            all_repeat_biased,
            all_repeat_real
        )
        results_file = save_dir / f"significance_results.txt"


            
        with open(results_file, "a") as f:

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {dataset_id}, {n_sel_features} genes \n")
            f.write("=" * 80 + "\n\n")

            for metric, res in significance_results.items():

                f.write(f"{metric}\n")
                f.write("-" * 40 + "\n")

                for bias, vals in res.items():

                    f.write(
                        f"{bias:20s} "
                        f"unbiased_dist={vals['mean_dist_unbiased']:.6f} "
                        f"biased_dist={vals['mean_dist_biased']:.6f} "
                        f"t={vals['t_stat']:.4f} "
                        f"p={vals['p_value']:.6g}\n"
                    )

                f.write("\n")
            
    if evaluate_bias:

        visited_size_biased_mean = {}
        new_bins_biased_mean = {}
        repeat_rate_biased_mean = {}

        for cur_bias in bias_terms:
            visited_size_biased_mean[cur_bias] = np.mean(
                np.stack(all_visited_biased[cur_bias]), axis=0
            )
            new_bins_biased_mean[cur_bias] = np.mean(
                np.stack(all_new_biased[cur_bias]), axis=0
            )
            repeat_rate_biased_mean[cur_bias] = np.mean(
                np.stack(all_repeat_biased[cur_bias]), axis=0
            )

        visited_size_unbiased = np.mean(np.stack(all_visited_unbiased), axis=0)
        visited_size_real = np.mean(np.stack(all_visited_real), axis=0)

        new_bins_unbiased = np.mean(np.stack(all_new_unbiased), axis=0)
        new_bins_real = np.mean(np.stack(all_new_real), axis=0)

        repeat_rate_unbiased = np.mean(np.stack(all_repeat_unbiased), axis=0)
        repeat_rate_real = np.mean(np.stack(all_repeat_real), axis=0)

        all_bias_plots_multi(
            X_real_val,
            dataset_id,
            visited_size_biased_mean,
            new_bins_biased_mean,
            repeat_rate_biased_mean,
            visited_size_unbiased,
            new_bins_unbiased,
            repeat_rate_unbiased,
            visited_size_real,
            new_bins_real,
            repeat_rate_real
        )

    if plot_mean_trajectories:


        # average over bootstrap runs
        mean_traj_biased_mean = {}

        for cur_bias in bias_terms:
            mean_traj_biased_mean[cur_bias] = np.mean(
                np.stack(all_mean_traj_biased[cur_bias]), axis=0
            )

        mean_traj_unbiased_mean = np.mean(
            np.stack(all_mean_traj_unbiased), axis=0
        )

        mean_traj_real_mean = np.mean(
            np.stack(all_mean_traj_real), axis=0
        )

        plot_mean_traj(X_real_val,
            dataset_id,
            mean_traj_biased_mean,
            mean_traj_unbiased_mean,
            mean_traj_real_mean
        )



def run_all_datasets(dataset_id, n_features, evaluate_bias = False, evaluate_kern = False, evaluate_model = False, which_bias = bias_terms[0]):

    save_dir = Path(__file__).resolve().parents[1] / "all_data_files"
    #project_dir = Path(__file__).resolve().parents[1]
    data_dir = Path(__file__).resolve().parents[1] / "stochastic_bias/all_data_files"

    #methylation bats, 3 observations per time point
    #dataset_id = "GSE164127" # works

    # methylation cross species (only taken mouse Cerebellum), 0.5 - 27 months, 
    # dont use this for now, doesnt work 
    #dataset_id = "GSE223748" # such a good and complete data BUT behaves as unbiased??


    # mice, methylation from kindey, 2-20 months, 15 observations per time point
    #dataset_id = "GSE120132" # works

    ##
    # rats methylation, 1.5-35 months,  ~5 observationsper time  point
    #dataset_id = 'GSE161141' # works!

    # mice, methylation 0.67-35 months, 6 observations per time point
    #dataset_id =  'GSE80672' # doesnt work 

    #X_real_val = np.load(f"X_real_val_{dataset_id}.npy")

    all_visited_biased = []
    all_visited_unbiased = []
    all_visited_real = []

    all_new_biased = []
    all_new_unbiased = []
    all_new_real = []

    all_repeat_biased = []
    all_repeat_unbiased = []
    all_repeat_real = []

    ####### kernel evaluation
    # all erros, to be later averaged 
    all_errors = []

    # to compare regression model
    all_errors_model = []

    n_bootstrap = 100
    n_kernel_runs = 1
    for i in tqdm(range(n_bootstrap)):
        n_sel_features = n_features
        X_real_val = np.load(f"all_data_files/X_real_val_{n_sel_features}_{dataset_id}_boot{i}.npy")

        if i == 79:
            aa = 1

        # run simulation and locally save the simulated prcoesses
        
        stoch_sim_CHECK.run_entire_simulation(dataset_id, n_sel_features)

        #  load real data, shape [time x Patients x Genes]

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

        X_biased_val = np.load(f"all_data_files/X_biased_val_{which_bias}.npy")[0:T][:, np.arange(n_obs)[:, None], np.arange(n_features)]

        #X_biased_t, X_biased_t1 = make_pairs(X_biased_val[:train_steps])
        #X_biased_pred = estimate_kernel_biased(X_biased_val, X_biased_t, X_biased_t1, T-train_steps)

        #X_biased_pred = np.load("X_biased_pred.npy")

        # load simulated data UNBIASED
        X_unbiased_val = np.load("all_data_files/X_unbiased_val.npy")[0:T][:, np.arange(n_obs)[:, None], np.arange(n_features)]



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


        if evaluate_bias:
        #33333333333333333333 evalute the effect of bias and check global sprad 


            visited_size_biased, new_bins_biased, repeat_rate_biased = evaluate_exploration(X_biased_val)

            visited_size_unbiased, new_bins_unbiased,  repeat_rate_unbiased = evaluate_exploration(X_unbiased_val)

            visited_size_real, new_bins_real,  repeat_rate_real  = evaluate_exploration(X_real_val)


            ########
            all_visited_biased.append(visited_size_biased)
            all_visited_unbiased.append(visited_size_unbiased)
            all_visited_real.append(visited_size_real)

            all_new_biased.append(new_bins_biased)
            all_new_unbiased.append(new_bins_unbiased)
            all_new_real.append(new_bins_real)

            all_repeat_biased.append(repeat_rate_biased)
            all_repeat_unbiased.append(repeat_rate_unbiased)
            all_repeat_real.append(repeat_rate_real)


        
    # GENERATE NEW STATES, compute error, repeat n times, take average

        if evaluate_model:

            future_steps = T - train_steps
            X_real_t, X_real_t1 = make_pairs(X_real_val[:train_steps])

            all_errors_this_model = []

            for cur_model in models:

                X_pred_biased = estimate_kernel_biased_drift(
                    X_real_val,
                    X_real_t,
                    X_real_t1,
                    future_steps,
                    my_model= models[cur_model],
                    seed=None
                )

                X_pred_unbiased = estimate_kernel_unbiased(
                    X_real_val,
                    X_real_t,
                    X_real_t1,
                    future_steps,
                    seed=None
                )

                errors_r = compute_kernel_scaled_errors(
                    X_real_val,
                    X_pred_biased,
                    X_pred_unbiased,
                    steps_train=train_steps,
                    full_process=False,
                )

                all_errors_this_model.append(errors_r)

            all_errors_model.append(all_errors_this_model)




        if evaluate_kern:

            future_steps = T - train_steps
            X_real_t, X_real_t1 = make_pairs(X_real_val[:train_steps])

            all_errors_this_bootstrap = []

            for r in range(n_kernel_runs):

                X_pred_biased = estimate_kernel_biased_drift(
                    X_real_val,
                    X_real_t,
                    X_real_t1,
                    future_steps,
                    my_model= my_model, 
                    seed=r
                )

                X_pred_unbiased = estimate_kernel_unbiased(
                    X_real_val,
                    X_real_t,
                    X_real_t1,
                    future_steps,
                    seed=r
                )

                errors_r = compute_kernel_scaled_errors(
                    X_real_val,
                    X_pred_biased,
                    X_pred_unbiased,
                    steps_train=train_steps,
                    full_process=False,
                )

                all_errors_this_bootstrap.append(errors_r)

            avg_errors_this_bootstrap = {}

            keys_to_average = [
                "mean_err_b",
                "mean_err_u",
                "cov_err_b",
                "cov_err_u",
                "wd_b",
                "wd_u",
                "corr_error_b",
                "corr_error_u",
            ]

            for key in keys_to_average:
                avg_errors_this_bootstrap[key] = np.mean(
                    np.stack([e[key] for e in all_errors_this_bootstrap], axis=0),
                    axis=0
                )

            avg_errors_this_bootstrap["T"] = all_errors_this_bootstrap[0]["T"]
            avg_errors_this_bootstrap["d"] = all_errors_this_bootstrap[0]["d"]

            all_errors.append(avg_errors_this_bootstrap)

    if evaluate_model:
        models_names = ["ridge", "lasso", "elastic_net"]

        keys_to_average = [
            "mean_err_b",
            "mean_err_u",
            "cov_err_b",
            "cov_err_u",
            "wd_b",
            "wd_u",
            "corr_error_b",
            "corr_error_u",
        ]

        avg_errors_models = {}

        for model_idx, model_name in enumerate(models_names):

            avg_errors_models[model_name] = {}

            for key in keys_to_average:

                avg_errors_models[model_name][key] = np.mean(
                    np.stack(
                        [
                            run_errors[model_idx][key]
                            for run_errors in all_errors_model
                        ],
                        axis=0,
                    ),
                    axis=0,
                )

            avg_errors_models[model_name]["T"] = all_errors_model[0][model_idx]["T"]
            avg_errors_models[model_name]["d"] = all_errors_model[0][model_idx]["d"]

        aa = 1
        plot_all_model_errors(avg_errors_models, dataset_id)





    if evaluate_bias:
        visited_size_biased = np.mean(np.stack(all_visited_biased), axis=0)
        visited_size_unbiased = np.mean(np.stack(all_visited_unbiased), axis=0)
        visited_size_real = np.mean(np.stack(all_visited_real), axis=0)

        new_bins_biased = np.mean(np.stack(all_new_biased), axis=0)
        new_bins_unbiased = np.mean(np.stack(all_new_unbiased), axis=0)
        new_bins_real = np.mean(np.stack(all_new_real), axis=0)

        repeat_rate_biased = np.mean(np.stack(all_repeat_biased), axis=0)
        repeat_rate_unbiased = np.mean(np.stack(all_repeat_unbiased), axis=0)
        repeat_rate_real = np.mean(np.stack(all_repeat_real), axis=0)

        ########3 PLOT ###############
        all_bias_plots(dataset_id,
                    X_biased_val,
                    visited_size_biased, new_bins_biased,repeat_rate_biased,
                    visited_size_unbiased, new_bins_unbiased, repeat_rate_unbiased, 
                visited_size_real, new_bins_real, repeat_rate_real
                ) 
        


    ####### Part 2: evaluate kenrel accuracy
    #### we estimate kernel using 2 approaches: biased and unbiased.
    ### then from the last observed states in the train set, we generate trajectories using both apporahces.
    ## then validate using real data and see which one captures the dynamics better

    #seed = None
   

    #X_biased_pred = estimate_kernel_biased_drift(X_real_val, X_real_t, X_real_t1, future_steps)[train_steps:]
    #X_unbiased_pred = estimate_kernel_unbiased(X_real_val, X_real_t, X_real_t1, future_steps)[train_steps:]
    #X_val = X_real_val[train_steps:]


    if evaluate_kern:


    ### for t-test:
        error_pairs = {
            "mean_err": ("mean_err_b", "mean_err_u"),
            "cov_err": ("cov_err_b", "cov_err_u"),
            "corr_err": ("corr_error_b", "corr_error_u"),
        }

        #### plot
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        for ax, (name, (key_b, key_u)) in zip(axes, error_pairs.items()):

            err_b = np.array([
                np.mean(err[key_b])
                for err in all_errors
            ])

            err_u = np.array([
                np.mean(err[key_u])
                for err in all_errors
            ])

            ax.plot(err_b, label="Biased")
            ax.plot(err_u, label="Unbiased")

            ax.set_title(f'{name}, dataset {dataset_id}, {n_features} genes')
            ax.set_ylabel("Average error")
            ax.legend()

        axes[-1].set_xlabel("Bootstrap replicate")

        plt.tight_layout()
        plt.show()


        #results_file = save_dir / f"significance_results.txt"
        results_file =  data_dir / "prediction_error_tests.txt"
        #results_file = save_dir / f"prediction_error_tests.txt"

        with open(results_file, "a") as f:

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: {dataset_id}, {n_features} genes\n")
            f.write("=" * 80 + "\n\n")

            for name, (key_b, key_u) in error_pairs.items():

                err_b = np.array([
                    np.mean(err[key_b])
                    for err in all_errors
                ])

                err_u = np.array([
                    np.mean(err[key_u])
                    for err in all_errors
                ])

                t_stat, p_val = ttest_rel(
                    err_u,
                    err_b,
                    alternative="greater"
                )

                f.write(
                    f"{name:12s} "
                    f"biased={err_b.mean():.6f} "
                    f"unbiased={err_u.mean():.6f} "
                    f"t={t_stat:.4f} "
                    f"p={p_val:.6g}\n"
                )

            f.write("\n")

        avg_errors = {}

        for key in keys_to_average:
            avg_errors[key] = np.mean(
                np.stack([errors[key] for errors in all_errors], axis=0),
                axis=0
            )

        avg_errors["T"] = all_errors[0]["T"]
        avg_errors["d"] = all_errors[0]["d"]

        avg_errors["scale_min"] = 0
        avg_errors["scale_max"] = np.max(np.concatenate([
            avg_errors["mean_err_b"],
            avg_errors["mean_err_u"],
            avg_errors["cov_err_b"],
            avg_errors["cov_err_u"],
            avg_errors["wd_b"],
            avg_errors["wd_u"],
        ]))

        plot_kernel_scaled_errors(
        dataset_id,
        avg_errors,
        steps_train=train_steps,
        save_dir="plots",
        full_process=False
    )

    #plot_increment_dependence_per_time(X_real_val[train_steps:T], X_pred_biased[train_steps:T], X_pred_unbiased[train_steps:T], plot=True, steps_train=train_steps)

    #evaluate_kernel_3plots(X_real_val, X_pred_biased, X_pred_unbiased, plot=True, steps_train=train_steps)
   
   # im using this
    #evaluate_kernel_normalized(dataset_id, X_real_val, X_pred_biased, X_pred_unbiased, plot=True, steps_train=train_steps)


    #evaluate_kernel_scaled(dataset_id, X_real_val, X_pred_biased, X_pred_unbiased, plot=True, steps_train=train_steps, full_process=False)

dataset_ids = [
   #"GSE164127",  # bats methylation
   # "GSE120132",  # mouse kidney methylation
    "GSE161141"  # rat methylation
    #"GSE80672"    # mouse methylation
]

aa = 1
# after I change anything in gene selecting method, i need to run all the datasets files and do preprocessing again 
# to make sure i select the same subset if genes
#subprocess.run([sys.executable, "run_all_data_files.py"])


for i in range(len(dataset_ids)):
    cur_data_set = dataset_ids[i] #in dataset_ids:
    #run_all_datasets(cur_data_set, 100, evaluate_bias=1, evaluate_kern=1)
    my_model =  models[models_all[i]]
    run_all_datasets(cur_data_set, 10, evaluate_bias=0, evaluate_kern=1, evaluate_model = 0)
    run_all_datasets(cur_data_set, 20, evaluate_bias=0, evaluate_kern=1, evaluate_model = 0)
    run_all_datasets(cur_data_set, 30, evaluate_bias=0, evaluate_kern=1, evaluate_model = 0)
    run_all_datasets(cur_data_set, 40, evaluate_bias=0, evaluate_kern=1, evaluate_model = 0)

    #run_all_datasets(cur_data_set, 100, evaluate_bias=True)

aa = 1


# do the comparative study of various bias terms first
for i in range(len(dataset_ids)):
    cur_data_set = dataset_ids[i] #in dataset_ids:
    #run_all_datasets(cur_data_set, 100, evaluate_bias=1, evaluate_kern=1)
    my_model =  models[models_all[i]]
    run_comp_study_bias(cur_data_set, 10, evaluate_bias=1, plot_mean_trajectories=0, do_t_test=1)
    run_comp_study_bias(cur_data_set, 20, evaluate_bias=1 , plot_mean_trajectories=0, do_t_test=1)
    run_comp_study_bias(cur_data_set, 30, evaluate_bias=1, plot_mean_trajectories=0, do_t_test=1)
    run_comp_study_bias(cur_data_set, 40, evaluate_bias=1, plot_mean_trajectories=0, do_t_test=1)

    #run_all_datasets(cur_data_set, 100, evaluate_bias=True)


#cur_data_set = dataset_ids[0]



