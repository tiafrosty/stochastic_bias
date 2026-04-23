import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *





dataset_id = "GSE167665"

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"

df_gene_expr = pd.read_csv(series_matrix_name,
    sep="\t",
    comment="!",   # removes GEO metadata lines
   index_col=0    # genes as index
)

samples = []
current = None

with open(family_soft_name, encoding="utf-8", errors="ignore") as f:
    total_lines = sum(1 for _ in f)

with open(family_soft_name, encoding="utf-8", errors="ignore") as f:
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

##########33
####################################################

# Extract subject and time from metadata 
meta = meta_d.copy()

files = glob.glob("extracted/*.csv")

# sort them, rn its a mess
files = sorted(files, key=lambda x: int(x.split("GSM")[1].split("_")[0]))


rows = []
sample_names = None

for f in tqdm(files, desc="Processing"):
    df = pd.read_csv(f)
    
    # first column = gene IDs, second = expression
    if sample_names is None:
        sample_names = df.iloc[:, 0]  # gene IDs
    
    expr = df.iloc[:, 1].values  # expression column
    
    rows.append(expr)

# build dataframe
df_all = pd.DataFrame(rows, columns=sample_names)

# set index as sample names
df_all.index = [f.split("/")[-1].replace(".csv", "") for f in files]

# we have 63 samples, grouped by tissue: brain, lung, liver, bone. Lets take brain, its the first 14 samples.
df_brain = df_all.iloc[:15]

# reshape it in the form of time x features

# 1. get GSM ids (strip suffix)
gsm = df_brain.index.str.replace(r"_s\d+$", "", regex=True)

# 2. get ages aligned with rows
ages = meta_d.set_index("GSM").loc[gsm, "age"].astype(int).values

# 3. get data matrix
X_raw = df_brain.values   # shape: samples × features

# 4. sort by time
idx = np.argsort(ages)

ages_sorted = ages[idx]
X_sorted = X_raw[idx]

# 5. aggregate duplicates (same time → average)
unique_times = np.unique(ages_sorted)

X = np.vstack([
    X_sorted[ages_sorted == t].mean(axis=0)
    for t in unique_times
])


### fix it later:
# i add an extra observation, identical, jsut to keep the shape of time x observations x features 
n_obs = 2
X_fixed = np.repeat(X[:, None, :], n_obs, axis=1)


####################################################
##############33 preprocess is done ################
####################################################

# again  take 5 genes
n_features = 5
X_sub = X_fixed[:, :, np.arange(n_features)]
feature_cols = np.array(df_gene_expr.index[0:n_features])


# test and train size
#T = X_fixed.shape[0]  # number of timepoints
#train_steps = int(0.8 * T)

#######3 estimate probabilities
#X_real_t, X_real_t1 = make_pairs(X_sub[:train_steps])

#X_real = X_sub.copy()
#X_future = estimate_kernel(X_real, X_real_t, X_real_t1, T-train_steps)

np.save(f"X_real_val_{dataset_id}.npy" , X_sub)

#np.save("X_real_pred.npy", X_future)

aa = 1




