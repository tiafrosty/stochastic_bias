import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *

# methylation mice
dataset_id = "GSE120132"

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"

# get meta data and gene expression (gene expression is usually empty though, but keep it just in case)
meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)

# now, read the extracted file with gene data (should only be used after running an .sh script for extraction) and put it together in a table
genes_table = get_extracted_table(dataset_id, 'txt', True, 10000) 


aa = 1
##############

####################################################
####### STep 2: PRE-PROCESS DATA SO ITS IN THE SHAPE OF 
###########  TIME X observationss X FEATURES (GENE EXPRESSION VALUES)
#############################################################
# ---------- keep only one tissue ----------
tissue = "Kidney"

mask = meta_d["tissue"].str.contains(tissue, case=False, na=False)

meta_sub = meta_d.loc[mask].copy()

genes_sub = genes_table.iloc[mask.values].copy()

# ---------- extract numeric age ----------
meta_sub["age_num"] = (
    meta_sub["age"]
    .str.extract(r"(\d+\.?\d*)")
    .astype(float)
)

# ---------- sort by age ----------
order = np.argsort(meta_sub["age_num"].values)

meta_sub = meta_sub.iloc[order].reset_index(drop=True)
genes_sub = genes_sub.iloc[order].reset_index(drop=True)


# ---------- build 3D tensor ----------
ages_unique = np.sort(meta_sub["age_num"].unique())

X_list = []

for age in ages_unique:

    idx = meta_sub["age_num"] == age

    # observations x features
    X_age = genes_sub.loc[idx].values.astype(float)

    X_list.append(X_age)


min_obs = min(mat.shape[0] for mat in X_list)
# keep only groups with at least 6 observations
filtered_matrices = [
    mat for mat in X_list
    if mat.shape[0] >= min_obs
]

filtered_ages = [
    age for age, mat in zip(ages_unique, X_list)
    if mat.shape[0] >= min_obs
]

# trim all to exactly 6 observations
# time x observations x features
X = np.stack(
    [mat[:min_obs, :] for mat in filtered_matrices],
    axis=0
)


#df_gene_expr.columns = meta_d.iloc[:, 1]

import config

for cur_subset_genes in config.n_sel_features:

    select_age_associated_features(X, dataset_id, filtered_ages, cur_subset_genes)
#select_age_associated_features(X, dataset_id, filtered_ages, 10)
#select_age_associated_features(X, dataset_id, filtered_ages, 20)
#select_age_associated_features(X, dataset_id, filtered_ages, 100)


aa = 1

# try this: predict like 50 states, then we have more time points
# look at the long term run
#######3 estimate probabilities
#train_steps = int(0.8 * X_sub.shape[0])


#X_real_t, X_real_t1 = make_pairs(X_sub[:train_steps])
#X_real_pred = estimate_kernel(X_sub, X_real_t, X_real_t1, 60)


