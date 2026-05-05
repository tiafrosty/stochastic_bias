import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm
import re

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *



# mice, RNAseq data from blood and muscle , 6-30 weeks,  ~15-20 observationsper time  point
dataset_id = 'E-GEOD-12290' 
#dataset_id = 'GSE167665'

# A rat epigenetic clock with methylation, ages 1-27 months 
#dataset_id = 'GSE161141' 

family_soft_name =  f"clinical-data(doita)/{dataset_id}.sdrf.txt"
#series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"


meta_d = pd.read_csv(family_soft_name, sep = '\s+')

aa = 1
# get meta data and gene expression (gene expression is usually empty though, but keep it just in case)
#meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)

# now, read the extracted file with gene data (should only be used after running an .sh script for extraction) and put it together in a table
genes_table = get_extracted_table(f'{dataset_id}', 'txt') 


#####33 prepocess: (ofc i have to do it separately AGAIN for each dataset oh well)
actual_data = genes_table 
#######3 normalize to get relative percentage 
###########3 was sigmoid but its all large and positive so minmax is better
actual_data = minmax(actual_data)
#actual_data =  sigmoid(actual_data) #(actual_data - actual_data.min()) / (actual_data.max() - actual_data.min())
#actual_data = actual_data.div(actual_data.sum(axis=1), axis=0) * 100


idx_df = meta_d.index.to_frame(index=False)

# first index level is GSM ID
idx_df["gsm_id"] = idx_df.iloc[:, 0].astype(str)

# second index level is row type: 1 or 2
idx_df["row_type"] = idx_df.iloc[:, 1].astype(str)

# full index text
idx_df["index_text"] = idx_df.astype(str).agg(" ".join, axis=1)

meta_tmp = meta_d.reset_index(drop=True).copy()
meta_tmp["gsm_id"] = idx_df["gsm_id"].values
meta_tmp["row_type"] = idx_df["row_type"].values
meta_tmp["index_text"] = idx_df["index_text"].values

# --------------------------------------------------
# 2. Extract true age from row_type == 1
# --------------------------------------------------

def extract_age_days(s):
    s = str(s).lower()
    m = re.search(r"\b(\d+)\s+day\s+old\b", s)
    if not m:
        raise ValueError(f"Could not extract age from: {s}")
    return int(m.group(1))

age_map = (
    meta_tmp.loc[meta_tmp["row_type"] == "1", ["gsm_id", "index_text"]]
    .drop_duplicates("gsm_id")
    .copy()
)

age_map["age_days"] = age_map["index_text"].apply(extract_age_days)

age_map = age_map.set_index("gsm_id")["age_days"]

# --------------------------------------------------
# 3. Get sample filenames from sample_table rows
# --------------------------------------------------

sample_rows = meta_tmp[
    meta_tmp["ArrayExpress"].astype(str).str.contains("_sample_table.txt", na=False)
].copy()

sample_rows["sample_file"] = sample_rows["ArrayExpress"].astype(str)
sample_rows["age_days"] = sample_rows["gsm_id"].map(age_map)

meta_aligned = sample_rows.set_index("sample_file").loc[actual_data.index]

print(meta_aligned["age_days"].value_counts().sort_index())

# --------------------------------------------------
# 4. Group gene data by unique ages
# --------------------------------------------------

time_points = np.sort(meta_aligned["age_days"].unique())

X_list = []
sample_ids_by_time = {}

for age in time_points:
    sample_ids = meta_aligned.index[meta_aligned["age_days"] == age]
    X_age = actual_data.loc[sample_ids].to_numpy()

    X_list.append(X_age)
    sample_ids_by_time[age] = list(sample_ids)

    print(f"age {age}: {X_age.shape}")

# --------------------------------------------------
# 5. Make X: time x observations x features
# --------------------------------------------------

obs_counts = [x.shape[0] for x in X_list]

if len(set(obs_counts)) == 1:
    X = np.stack(X_list, axis=0)
else:
    max_obs = max(obs_counts)
    n_features = actual_data.shape[1]

    X = np.full((len(time_points), max_obs, n_features), np.nan)

    for t, X_age in enumerate(X_list):
        X[t, :X_age.shape[0], :] = X_age

aa = 1


n_keep = 9

X_selected = []
selected_sample_ids_by_time = {}

for t, age in enumerate(time_points):
    sample_ids = np.array(sample_ids_by_time[age])
    X_age = actual_data.loc[sample_ids].to_numpy()

    # count missing genes per observation
    missing_per_obs = np.isnan(X_age).sum(axis=1)

    # choose 9 observations with fewest missing values
    keep_idx = np.argsort(missing_per_obs)[:n_keep]

    X_selected.append(X_age[keep_idx])
    selected_sample_ids_by_time[age] = sample_ids[keep_idx].tolist()

    print(
        f"age {age}: total={X_age.shape[0]}, "
        f"kept={n_keep}, "
        f"missing in kept rows={missing_per_obs[keep_idx]}"
    )

X = np.stack(X_selected, axis=0)

print("Before gene filtering:", X.shape)
print("Any NaNs before gene filtering?", np.isnan(X).any())

# remove genes/features that have NaN anywhere in selected data
good_genes = ~np.isnan(X).any(axis=(0, 1))

X = X[:, :, good_genes]
#gene_names_clean = gene_names[good_genes]

# take a few features as befoer
n_features = 5
X_sub = X[:, :, np.arange(n_features)]


# try this: predict like 50 states, then we have more time points
# look at the long term run
#######3 estimate probabilities
train_steps = int(0.8 * X_sub.shape[0])


#X_real_t, X_real_t1 = make_pairs(X_sub[:train_steps])
#X_real_pred = estimate_kernel(X_sub, X_real_t, X_real_t1, 60)


np.save(f"X_real_val_5_{dataset_id}.npy" , X_sub)

n_features = 30
X_sub = X[:, :, np.arange(n_features)]


# try this: predict like 50 states, then we have more time points
# look at the long term run
#######3 estimate probabilities
train_steps = int(0.8 * X_sub.shape[0])
np.save(f"X_real_val_30_{dataset_id}.npy" , X_sub)

#np.save(f"X_real_pred_{dataset_id}.npy" , minmax(X_real_pred))


####### try takking more features
#X_sub = X[:, :, np.arange(70)]
#train_steps = int(0.8 * X_sub.shape[0])
#X_real_t, X_real_t1 = make_pairs(X_sub[:train_steps])
#X_real_pred_large = estimate_kernel(X_sub, X_real_t, X_real_t1, 60)

#np.save(f"X_real_pred_large_{dataset_id}.npy" , minmax(X_real_pred_large))

aa = 1