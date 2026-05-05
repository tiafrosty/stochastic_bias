import numpy as np
import matplotlib.pyplot as plt


def all_bias_plots(dataset_id,
    X_b,
    visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased, entropy_biased, var_t_biased,
    visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased, entropy_unbiased, var_t_unbiased,
    visited_size_real, mean_real, var_real, global_min_real, global_max_real, global_range_real, new_bins_real, entropy_real, var_t_real
):
    import numpy as np
    import matplotlib.pyplot as plt

    # common font size for legend in all the plots
    legend_font = 16
    label_xy_size = 16
    title_size = 25
    xy_tick_size = 14

    time_steps, _, state_dim = X_b.shape

    # ---------------- Visited-set growth ----------------
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(np.arange(1, time_steps + 1), visited_size_biased, "-", lw=2, label="Biased (unique bins)")
    plt.plot(np.arange(1, time_steps + 1), visited_size_unbiased, ":", lw=3, label="Unbiased (unique bins)")
    plt.plot(np.arange(1, time_steps + 1), visited_size_real, "--", lw=2, label="Real (unique bins)")
    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Unique states visited", fontsize=label_xy_size)
    plt.title("Visited-set growth", fontsize=title_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('visited-states-growth.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Mean ----------------
    tgrid = np.arange(time_steps)

    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 2.2 * state_dim), sharex=True)
    if state_dim == 1:
        axes = [axes]

    for j in range(state_dim):
        axes[j].plot(tgrid, mean_biased[:, j], "-",  label="biased mean", lw=2)
        axes[j].plot(tgrid, mean_unbiased[:, j], ":", label="unbiased mean", lw=3)
        axes[j].plot(tgrid, mean_real[:, j], "--", label="real mean", lw=2)

        axes[j].set_title(f"Average X[{j}]", fontsize=16)
        axes[j].grid(alpha=0.3)
        axes[j].legend(loc='best', fontsize=legend_font)

    axes[-1].set_xlabel("time step", fontsize=label_xy_size)
    axes[-1].tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('averaged_features_all.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Variance ----------------
    var_biased_t = var_biased
    var_unbiased_t = var_unbiased
    var_real_t = var_real

    tgrid = np.arange(1, time_steps + 1)

    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 2.2 * state_dim), sharex=True)
    if state_dim == 1:
        axes = [axes]

    for j in range(state_dim):
        axes[j].plot(tgrid, var_biased_t[:, j], "-",  lw=2, label="Biased")
        axes[j].plot(tgrid, var_unbiased_t[:, j], ":", lw=3, label="Unbiased")
        axes[j].plot(tgrid, var_real_t[:, j], "--", lw=2, label="Real")

        axes[j].grid(True, alpha=0.4)
        axes[j].tick_params(axis="both", labelsize=12)
        axes[j].set_title(f"Variance X[{j}]", fontsize=16)
        axes[j].legend(loc="best", fontsize=legend_font)

    axes[-1].set_xlabel("Time step", fontsize=label_xy_size)
    plt.tight_layout()
    plt.savefig("variance-vs-time-all-features.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # ---------------- Exploration rate ----------------
    tgrid = np.arange(1, time_steps + 1)

    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(tgrid, new_bins_biased, "-",  label="Biased (new bins per step)", lw=2)
    plt.plot(tgrid, new_bins_unbiased, ":", label="Unbiased (new bins per step)", lw=3)
    plt.plot(tgrid, new_bins_real, "--", label="Real (new bins per step)", lw=2)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("# new states", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Exploration rate", fontsize=title_size)
    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('exploration-rate.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Global min/max envelope ----------------
    tgrid = np.arange(1, time_steps + 1)

    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(tgrid, global_min_biased,   "-",  label="Biased global min", lw=2)
    plt.plot(tgrid, global_max_biased,   "-",  label="Biased global max", lw=2)
    plt.plot(tgrid, global_min_unbiased, ":",  label="Unbiased global min", lw=3)
    plt.plot(tgrid, global_max_unbiased, ":",  label="Unbiased global max", lw=3)
    plt.plot(tgrid, global_min_real,     "--", label="Real global min", lw=2)
    plt.plot(tgrid, global_max_real,     "--", label="Real global max", lw=2)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Value", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Global min/max envelope over time", fontsize=title_size)
    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('min-max.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Global spread ----------------
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(tgrid, global_range_biased, "-",  label="Biased global range (max–min)", lw=2)
    plt.plot(tgrid, global_range_unbiased, ":", label="Unbiased global range (max–min)", lw=3)
    plt.plot(tgrid, global_range_real, "--", label="Real global range (max–min)", lw=2)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Range", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Global spread over time", fontsize=title_size)
    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('spread.png', dpi=300, bbox_inches="tight")
    plt.show()


    # ---------------- Entropy ----------------
    plt.figure(figsize=(9, 5), dpi=120)

    plt.plot(tgrid, entropy_biased, "-",  label="Biased entropy", lw=2)
    plt.plot(tgrid, entropy_unbiased, ":", label="Unbiased entropy", lw=3)
    plt.plot(tgrid, entropy_real, "--", label="Real entropy", lw=2)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Entropy", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Entropy over time", fontsize=title_size)

    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)

    plt.tight_layout()
    plt.savefig('entropy.png', dpi=300, bbox_inches="tight")
    plt.show()


    # ---------------- Mean squared temporal change ----------------
    plt.figure(figsize=(9, 5), dpi=120)

    tgrid_var = np.arange(1, len(var_t_biased) + 1)

    plt.plot(tgrid_var, var_t_biased, "-",  label="Biased", lw=2)
    plt.plot(tgrid_var, var_t_unbiased, ":", label="Unbiased", lw=3)
    plt.plot(tgrid_var, var_t_real, "--", label="Real", lw=2)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Mean squared change", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Temporal variation between states", fontsize=title_size)

    plt.grid(True, alpha=0.4)
    plt.legend(loc="best", fontsize=legend_font)

    plt.tight_layout()
    plt.savefig("temporal-variation.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------- Combined 2x2 panel ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    lw = 3.5
    label_size = 16
    title_size = 20
    tick_size = 14

    # ---- 1. Visited-set growth ----
    ax = axes[0, 0]
    tgrid = np.arange(1, time_steps + 1)

    ax.plot(tgrid, visited_size_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, visited_size_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, visited_size_real, "--", lw=lw, label="Real")

    ax.set_title("Visited-state growth", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("Unique states", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)


    # ---- 2. Exploration rate ----
    ax = axes[0, 1]

    ax.plot(tgrid, new_bins_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, new_bins_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, new_bins_real, "--", lw=lw, label="Real")

    ax.set_title("Exploration rate", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("# new states", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)


    # ---- 3. Entropy ----
    ax = axes[1, 0]

    ax.plot(tgrid, entropy_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, entropy_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, entropy_real, "--", lw=lw, label="Real")

    ax.set_title("Entropy", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    #ax.set_ylabel("Entropy", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)


    # ---- 4. Temporal variation ----
    ax = axes[1, 1]
    tgrid_var = np.arange(1, len(var_t_biased) + 1)

    ax.plot(tgrid_var, var_t_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid_var, var_t_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid_var, var_t_real, "--", lw=lw, label="Real")

    ax.set_title("Temporal variation", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    #ax.set_ylabel("Mean squared change", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)


    # ---- Final layout ----
    plt.tight_layout()
    plt.savefig(f"combined_dynamics_panel_{dataset_id}.png", dpi=300, bbox_inches="tight")
    plt.show()
        
    

def bias_plots_without_real_data(
    X_b,
    visited_size_biased, mean_biased, var_biased, global_min_biased, global_max_biased, global_range_biased, new_bins_biased, entropy_biased,
    visited_size_unbiased, mean_unbiased, var_unbiased, global_min_unbiased, global_max_unbiased, global_range_unbiased, new_bins_unbiased, entropy_unbiased
):
    import numpy as np
    import matplotlib.pyplot as plt

    # common font size for legend in all the plots
    legend_font = 16
    label_xy_size = 16
    title_size = 25
    xy_tick_size = 14

    time_steps, _, state_dim = X_b.shape

    # ---------------- Visited-set growth ----------------
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(np.arange(1, time_steps + 1), visited_size_biased, "-", lw=2, label="Biased (unique bins)")
    plt.plot(np.arange(1, time_steps + 1), visited_size_unbiased, ":", lw=3, label="Unbiased (unique bins)")
    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Unique states visited", fontsize=label_xy_size)
    plt.title("Visited-set growth", fontsize=title_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('visited-states-growth.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Mean ----------------
    tgrid = np.arange(time_steps)

    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 2.2 * state_dim), sharex=True)
    if state_dim == 1:
        axes = [axes]

    for j in range(state_dim):
        axes[j].plot(tgrid, mean_biased[:, j], "-", label="Biased mean", lw=2)
        axes[j].plot(tgrid, mean_unbiased[:, j], ":", label="Unbiased mean", lw=3)

        axes[j].set_title(f"Average X[{j}]", fontsize=16)
        axes[j].grid(alpha=0.3)
        axes[j].legend(loc='best', fontsize=legend_font)
        axes[j].tick_params(axis='both', labelsize=12)

    axes[-1].set_xlabel("Time step", fontsize=label_xy_size)
    plt.tight_layout()
    plt.savefig('averaged_features_all.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Variance ----------------
    tgrid = np.arange(1, time_steps + 1)

    fig, axes = plt.subplots(state_dim, 1, figsize=(10, 2.2 * state_dim), sharex=True)
    if state_dim == 1:
        axes = [axes]

    for j in range(state_dim):
        axes[j].plot(tgrid, var_biased[:, j], "-", lw=2, label="Biased")
        axes[j].plot(tgrid, var_unbiased[:, j], ":", lw=3, label="Unbiased")

        axes[j].grid(True, alpha=0.4)
        axes[j].tick_params(axis="both", labelsize=12)
        axes[j].set_title(f"Variance X[{j}]", fontsize=16)
        axes[j].legend(loc="best", fontsize=legend_font)

    axes[-1].set_xlabel("Time step", fontsize=label_xy_size)
    plt.tight_layout()
    plt.savefig("variance-vs-time-all-features.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # ---------------- Exploration rate ----------------
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(tgrid, new_bins_biased, "-", label="Biased (new bins per step)", lw=2)
    plt.plot(tgrid, new_bins_unbiased, ":", label="Unbiased (new bins per step)", lw=3)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("# new states", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Exploration rate", fontsize=title_size)
    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('exploration-rate.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Global min/max envelope ----------------
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(tgrid, global_min_biased, "-", label="Biased global min", lw=2)
    plt.plot(tgrid, global_max_biased, "-", label="Biased global max", lw=2)
    plt.plot(tgrid, global_min_unbiased, ":", label="Unbiased global min", lw=3)
    plt.plot(tgrid, global_max_unbiased, ":", label="Unbiased global max", lw=3)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Value", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Global min/max envelope over time", fontsize=title_size)
    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('min-max.png', dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------------- Global spread ----------------
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(tgrid, global_range_biased, "-", label="Biased global range (max–min)", lw=2)
    plt.plot(tgrid, global_range_unbiased, ":", label="Unbiased global range (max–min)", lw=3)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Range", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Global spread over time", fontsize=title_size)
    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)
    plt.tight_layout()
    plt.savefig('spread.png', dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------- Entropy ----------------
    plt.figure(figsize=(9, 5), dpi=120)

    plt.plot(tgrid, entropy_biased, "-", label="Biased entropy", lw=2)
    plt.plot(tgrid, entropy_unbiased, ":", label="Unbiased entropy", lw=3)

    plt.xlabel("Time step", fontsize=label_xy_size)
    plt.ylabel("Entropy", fontsize=label_xy_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.title("Entropy over time", fontsize=title_size)

    plt.grid(True, alpha=0.4)
    plt.legend(loc='best', fontsize=legend_font)

    plt.tight_layout()
    plt.savefig('entropy.png', dpi=300, bbox_inches="tight")
    plt.show()


    
