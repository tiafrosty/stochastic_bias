import pandas as pd
from tqdm import tqdm
import numpy as np

# connect my modules
from kernel_estimation import *

df_gene_expr = pd.read_csv(
    "GSE30550_series_matrix.txt",
    sep="\t",
    comment="!",   # removes GEO metadata lines
    index_col=0    # genes as index
)

aa = 1

samples = []
current = None

with open("GSE30550_family.soft", encoding="utf-8", errors="ignore") as f:
    total_lines = sum(1 for _ in f)

with open("GSE30550_family.soft", encoding="utf-8", errors="ignore") as f:
    for raw_line in tqdm(f, total=total_lines):
        line = raw_line.strip()

        if line.startswith("^SAMPLE"):
            if current is not None:
                samples.append(current)
            current = {"GSM": line.split(" = ", 1)[1]}

        elif current is not None and line.startswith("!Sample_title"):
            current["title"] = line.split(" = ", 1)[1]

        elif current is not None and line.startswith("!Sample_characteristics_ch1"):
            val = line.split(" = ", 1)[1]
            if ":" in val:
                k, v = val.split(":", 1)
                k = k.strip()
                v = v.strip()

                if k in current:
                    if isinstance(current[k], list):
                        current[k].append(v)
                    else:
                        current[k] = [current[k], v]
                else:
                    current[k] = v

if current is not None:
    samples.append(current)

meta_d = pd.DataFrame(samples)

####################################################
####### STep 2: PRE-PROCESS DATA SO ITS IN THE SHAPE OF 
###########  TIME X SUBJECT X FEATURES (GENE EXPRESSION VALUES)
#############################################################


#df_gene_expr.columns = meta_d.iloc[:, 1]

####################################################

# Extract subject and time from metadata 
meta = meta_d.copy()

# Expected: meta["GSM"], meta["title"]
meta["subject"] = meta["title"].str.extract(r"Subject\s+(\d+)").astype(int)


time_raw = meta["time_raw"] = meta["title"].str.extract(r"(Baseline|Hour\s*\d+)")[0]

meta["time"] = (
    time_raw
    .str.replace("Hour", "", regex=False)
    .str.strip()
    .replace("Baseline", "0")
    .astype(int)
)

# get ordered unique timepoints
time_levels = sorted(meta["time_raw"].unique())

# map to indices: 0,1,2,...
time_to_idx = {t: i for i, t in enumerate(time_levels)}

# assign index
meta["time"] = meta["time_raw"].map(time_to_idx)

# Keep only what is needed
meta = meta[["GSM", "subject", "time"]].copy()
# keep time as just indices, dont care about what hour is that
#meta['time'] =  np.arange(len(meta['time']))


# Keep only samples present in both
common_samples = [s for s in meta["GSM"] if s in df_gene_expr.columns]

meta = meta[meta["GSM"].isin(common_samples)].copy()
expr = df_gene_expr[common_samples].copy()

# 3. Sort metadata so subject order is fixed and time order is fixed
meta = meta.sort_values(["time", "subject"]).reset_index(drop=True)

# 
# Each subject should appear once at each time
counts = meta.groupby(["time", "subject"]).size().unstack(fill_value=0)

# OBS! Some patients contain missing data. Check first which ones are complete 
complete_subjects = counts.columns[(counts == 1).all(axis=0)]
# then drop the rest
meta_clean = meta[meta["subject"].isin(complete_subjects)].copy()

# should be fine now but lets still check 
counts_clean = meta_clean.groupby(["time", "subject"]).size().unstack(fill_value=0)
if not ((counts_clean == 1).all().all()):
    raise ValueError(
        "Panel is not complete: some subject-time combinations are missing "
        "or duplicated. Check meta before reshaping."
    )

# ordered lists
subjects = sorted(meta_clean["subject"].unique())
times = sorted(meta_clean["time"].unique())

n_subjects = len(subjects)
n_times = len(times)
n_genes = expr.shape[0]

# Maps from labels to array indices
subject_to_idx = {s: i for i, s in enumerate(subjects)}
time_to_idx = {t: i for i, t in enumerate(times)}
# X with shape (Time, subject, Features) ---
X = np.empty((n_times, n_subjects, n_genes), dtype=float)

for _, row in meta_clean.iterrows():
    gsm = row["GSM"]
    t_idx = time_to_idx[row["time"]]
    s_idx = subject_to_idx[row["subject"]]
    
    # expr[gsm] is a vector of length n_genes
    X[t_idx, s_idx, :] = expr[gsm].to_numpy()

#
subject_order = subjects
time_order = times
gene_order = expr.index.tolist()

# take like 5-10 genes (out of like 11k) and look
n_features = 5
X_sub = X[:, :, np.arange(n_features)]
feature_cols = np.array(df_gene_expr.index[0:n_features])
#plot_features_vs_time_groups(X_sub, np.arange(X.shape[0]), feature_cols, n_groups=3)
#plot_distribution_error(Xb_val, Xb_future_all, 'BIASED',  steps_train)

# All the pre-processing is done now. APply my method of probability kernel estimation

# test and train size
T = X_sub.shape[0]  # number of timepoints
train_steps = int(0.8 * T)

#######3 estimate probabilities
X_real_t, X_real_t1 = make_pairs(X_sub[:train_steps])

X_real = X_sub.copy()
X_future = estimate_kernel(X_real, X_real_t, X_real_t1, T-train_steps)

np.save("X_real_val.npy", X_real)
np.save("X_real_pred.npy", X_future)


#plot_distribution_error(X_real, X_future, 'GSE30550 gene expr',  train_steps,  feature_cols)
#plot_feature_histograms(X_real, X_future, 'GSE30550 gene expr', time_points=[13, 14, 15])

evaluate_kernel(X_real, X_future, plot=True, biased_unbiased='GSE30550 gene expr', steps_train=train_steps)

aa = 1