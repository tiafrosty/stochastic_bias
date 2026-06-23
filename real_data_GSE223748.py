import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *

# methylation mice
dataset_id = "GSE223748"

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"


f = 'clinical-data(doita)/GSE223748_datBetaNormalized.csv'

df = pd.read_csv(f)
# get meta data and gene expression (gene expression is usually empty though, but keep it just in case)
meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)


species = 'Mouse Striatum' #'mouse Cerebellum' #

mask = meta_d["title"].str.contains(species, case=False, na=False)

meta_mice = meta_d.loc[mask].copy()

meth = df.set_index(df.columns[0]).T

meta_mice["array_id"] = meta_mice["title"].str.extract(
    r"(20\d+_R\d+C\d+)"
)

sample_ids = meta_mice["array_id"].values

genes_sub = meth.loc[sample_ids].copy()

# now, read the extracted file with gene data (should ONLY be used after running an .sh script for extraction) and put it together in a table
#genes_table = get_extracted_table(dataset_id, 'txt', True, 10000) 


aa = 1
##############

####################################################
####### STep 2: PRE-PROCESS DATA SO ITS IN THE SHAPE OF 
###########  TIME X observationss X FEATURES (GENE EXPRESSION VALUES)
#############################################################
# ---------- keep only one tissue ----------

# ---------- extract numeric age ----------

# if age column has nan, remove them
meta_mice = meta_mice[meta_mice["age"] != "NA"]

meta_mice["age_num"] = (
    meta_mice["age"].astype(float)
)

# ---------- sort by age ----------
order = np.argsort(meta_mice["age_num"].values)

meta_sub = meta_mice.iloc[order].reset_index(drop=True)
genes_sub = genes_sub.iloc[order].reset_index(drop=True)


# ---------- build 3D tensor ----------
ages_unique = np.sort(meta_sub["age_num"].unique())

X_list = []

for age in ages_unique:

    idx = meta_sub["age_num"] == age

    # observations x features
    X_age = genes_sub.loc[idx].values.astype(float)

    X_list.append(X_age)


min_obs = 6 #min(mat.shape[0] for mat in X_list)
# keep only groups with at least min_obs observations
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


# new idea: instead of randomly choosing genes, i choose those with lower variance
select_age_associated_features(X, dataset_id, filtered_ages, 5)
select_age_associated_features(X, dataset_id, filtered_ages, 30)
select_age_associated_features(X, dataset_id, filtered_ages, 100)

aa = 1

