import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *


dataset_id = "GSE80672"

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"

# get meta data and gene expression (gene expression is usually empty though, but keep it just in case)
meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)

# now, read the extracted file with gene data (should only be used after running an .sh script for extraction) and put it together in a table
genes_table = get_extracted_table(dataset_id, 'txt') 


aa = 1
##############

####################################################
####### STep 2: PRE-PROCESS DATA SO ITS IN THE SHAPE OF 
###########  TIME X SUBJECT X FEATURES (GENE EXPRESSION VALUES)
#############################################################


#df_gene_expr.columns = meta_d.iloc[:, 1]

####################################################

# Extract subject and time from metadata 
# 1. methylation % -> 0–1
genes_01 = genes_table.astype(float) / 100.0

# 2. clean ages: remove ages with text, keep numeric only
ages = pd.to_numeric(meta_d["age (years)"], errors="coerce")

valid = ages.notna()
ages = ages[valid]
genes_01 = genes_01.loc[valid.values]

# 3. sort by age
order = np.argsort(ages.values)
ages = ages.iloc[order]
genes_01 = genes_01.iloc[order]

# 4. group samples by age
age_groups = sorted(ages.unique())

grouped_matrices = []
for age in age_groups:
    idx = np.where(ages.values == age)[0]
    mat = genes_01.iloc[idx].to_numpy()
    grouped_matrices.append(mat)


# 5. lowest number of observations across ages
min_obs = 6 #min(mat.shape[0] for mat in grouped_matrices)
# keep only groups with at least 6 observations
filtered_matrices = [
    mat for mat in grouped_matrices
    if mat.shape[0] >= min_obs
]

filtered_ages = [
    age for age, mat in zip(age_groups, grouped_matrices)
    if mat.shape[0] >= min_obs
]

# trim all to exactly 6 observations
X = np.stack(
    [mat[:min_obs, :] for mat in filtered_matrices],
    axis=0
)

import config

for cur_subset_genes in config.n_sel_features:

    select_age_associated_features(X, dataset_id, filtered_ages, cur_subset_genes)
#select_age_associated_features(X, dataset_id, filtered_ages, 10)
#select_age_associated_features(X, dataset_id, filtered_ages, 20)
#select_age_associated_features(X, dataset_id, filtered_ages, 100)


aa = 1

