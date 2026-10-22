import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def bias_3_plots(dataset_id,
    X_b,
    visited_size_biased, new_bins_biased, repeat_rate_biased,
    visited_size_unbiased, new_bins_unbiased, repeat_rate_unbiased,
    visited_size_real,  new_bins_real, repeat_rate_real
):
    import numpy as np
    import matplotlib.pyplot as plt

    # common font size for legend in all the plots
    legend_font = 16
    label_xy_size = 16
    title_size = 25
    xy_tick_size = 14

    time_steps, _, state_dim = X_b.shape

    # ---------------- Combined 2x2 panel ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    lw = 3.5
    label_size = 16
    title_size = 20
    tick_size = 14

    # ---- 1. Visited-set growth ----
    ax = axes[0]
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
    ax = axes[1]

    ax.plot(tgrid, new_bins_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, new_bins_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, new_bins_real, "--", lw=lw, label="Real")

    ax.set_title("Exploration rate", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("# new states", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)


    # ---- 3. Repeat rate ----
    ax = axes[2]

    ax.plot(tgrid, repeat_rate_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, repeat_rate_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, repeat_rate_real, "--", lw=lw, label="Real")

    ax.set_title("Repeat rate", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    #ax.set_ylabel("Repeat rate", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)

    # ---- Final layout ----
    plt.tight_layout()
    plt.savefig(
        f"plots/combined_dynamics_panel_{state_dim}_{dataset_id}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


def plot_mean_traj(X,
    dataset_id,
    mean_traj_biased_dict,
    mean_traj_unbiased,
    mean_traj_real
):

    time_steps, _, state_dim = X.shape
    Path("plots").mkdir(parents=True, exist_ok=True)

    T = len(mean_traj_real)
    tgrid = np.arange(1, T + 1)

    plt.figure(figsize=(8, 5), dpi=120)

    lw = 3

    cmap = plt.get_cmap("tab10")

    for i, (bias_name, traj) in enumerate(mean_traj_biased_dict.items()):
        plt.plot(
            tgrid,
            traj,
            lw=lw,
            color=cmap(i),
            label=bias_name
        )

    plt.plot(
        tgrid,
        mean_traj_unbiased,
        ":",
        color="black",
        lw=lw,
        label="Unbiased"
    )

    plt.plot(
        tgrid,
        mean_traj_real,
        "--",
        color="black",
        lw=lw,
        label="Real"
    )

    plt.xlabel("Time step", fontsize=16)
    plt.ylabel("Mean value", fontsize=16)
    plt.title("Mean trajectory over time", fontsize=20)
    plt.tick_params(labelsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()

    plt.savefig(
        f"plots/mean_trajectory_bias_comparison_{state_dim}_{dataset_id}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



def all_bias_plots_multi(X_b,
    dataset_id,
    visited_size_biased_dict,
    new_bins_biased_dict,
    repeat_rate_biased_dict,
    visited_size_unbiased,
    new_bins_unbiased,
    repeat_rate_unbiased,
    visited_size_real,
    new_bins_real,
    repeat_rate_real
):


    time_steps, _, state_dim = X_b.shape

    lw = 3
    label_size = 16
    title_size = 20
    tick_size = 14
    legend_font = 15

    tgrid = np.arange(1, len(visited_size_real) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    cmap = plt.get_cmap("tab10")
    bias_names = list(visited_size_biased_dict.keys())

    # ---------- PANEL 1 ----------
    ax = axes[0]

    for i, bias in enumerate(bias_names):
        ax.plot(
            tgrid,
            visited_size_biased_dict[bias],
            lw=lw,
            color=cmap(i),
            label=bias
        )

    ax.plot(
        tgrid,
        visited_size_unbiased,
        ":",
        color="black",
        lw=lw,
        label="Unbiased"
    )

    ax.plot(
        tgrid,
        visited_size_real,
        "--",
        color="black",
        lw=lw,
        label="Real"
    )

    ax.set_title("Visited-state growth G(t)", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("Unique states", fontsize=label_size)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=tick_size)

    # ---------- PANEL 2 ----------
    ax = axes[1]

    for i, bias in enumerate(bias_names):
        ax.plot(
            tgrid,
            new_bins_biased_dict[bias],
            lw=lw,
            color=cmap(i),
            label=bias
        )

    ax.plot(
        tgrid,
        new_bins_unbiased,
        ":",
        color="black",
        lw=lw,
        label="Unbiased"
    )

    ax.plot(
        tgrid,
        new_bins_real,
        "--",
        color="black",
        lw=lw,
        label="Real"
    )

    ax.set_title("Exploration rate E(t)", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("# new states", fontsize=label_size)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=tick_size)

    # ---------- PANEL 3 ----------
    ax = axes[2]

    for i, bias in enumerate(bias_names):
        ax.plot(
            tgrid,
            repeat_rate_biased_dict[bias],
            lw=lw,
            color=cmap(i),
            label=bias
        )

    ax.plot(
        tgrid,
        repeat_rate_unbiased,
        ":",
        color="black",
        lw=lw,
        label="Unbiased"
    )

    ax.plot(
        tgrid,
        repeat_rate_real,
        "--",
        color="black",
        lw=lw,
        label="Real"
    )

    ax.set_title("Repeat rate R(t)", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("Repeat rate", fontsize=label_size)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=tick_size)

    # Single legend for entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=legend_font,
        bbox_to_anchor=(0.5, 0.995)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    #plt.tight_layout()

    Path("plots").mkdir(exist_ok=True)

    plt.savefig(
        f"plots/panel_bias_comparison_{state_dim}_{dataset_id}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()



def all_bias_plots(dataset_id,
    X_b,
    visited_size_biased, new_bins_biased, repeat_rate_biased, 
    visited_size_unbiased,new_bins_unbiased, repeat_rate_unbiased, 
    visited_size_real, new_bins_real, repeat_rate_real
):
    import numpy as np
    import matplotlib.pyplot as plt

    # common font size for legend in all the plots
    legend_font = 16
    label_xy_size = 16
    title_size = 25
    xy_tick_size = 14

    time_steps, _, state_dim = X_b.shape

    

    # ---------------- Combined 2x2 panel ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    lw = 3.5
    label_size = 16
    title_size = 20
    tick_size = 14

    # ---- 1. Visited-set growth ----
    ax = axes[0]
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
    ax = axes[1]

    ax.plot(tgrid, new_bins_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, new_bins_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, new_bins_real, "--", lw=lw, label="Real")

    ax.set_title("Exploration rate", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    ax.set_ylabel("# new states", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)


    # ---- 3. Repeat rate ----
    ax = axes[2]

    ax.plot(tgrid, repeat_rate_biased, "-", lw=lw, label="Biased")
    ax.plot(tgrid, repeat_rate_unbiased, ":", lw=lw, label="Unbiased")
    ax.plot(tgrid, repeat_rate_real, "--", lw=lw, label="Real")

    ax.set_title("Repeat rate", fontsize=title_size)
    ax.set_xlabel("Time step", fontsize=label_size)
    #ax.set_ylabel("Repeat rate", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)

    # ---- Final layout ----
    plt.tight_layout()
    plt.savefig(
        f"plots/combined_dynamics_panel_{state_dim}_{dataset_id}.png",
        dpi=300,
        bbox_inches="tight"
    )
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


    
