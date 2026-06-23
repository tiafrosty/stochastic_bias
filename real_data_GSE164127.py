import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *

import config

# how to select genes
    #return X[:, :, idx]

# bats
dataset_id = "GSE164127"

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"

# get meta data and gene expression (gene expression is usually empty though, but sometimes its included)
meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)

# now, read the extracted file with gene data (should only be used after running an .sh script for extraction) and put it together in a table
genes_table = gene_d #get_extracted_table(dataset_id, 'idat') 

# ---------- choose species with most samples ----------
meta_d = meta_d.copy()

meta_d["species"] = meta_d["title"].str.split(".").str[1]

species_counts = meta_d["species"].value_counts()
species = species_counts.index[1] # take 2nd max maybe?

print("Selected species:", species)
print("Number of samples:", species_counts.iloc[0])


# ---------- subset metadata ----------
mask = meta_d["species"] == species
meta_sub = meta_d.loc[mask].copy()


# ---------- get age and remove NaNs ----------
meta_sub["age_num"] = pd.to_numeric(meta_sub["age"], errors="coerce")

# if age is stored in "age (years)" for some samples
if "age (years)" in meta_sub.columns:
    age_years = pd.to_numeric(meta_sub["age (years)"], errors="coerce")
    meta_sub["age_num"] = meta_sub["age_num"].fillna(age_years)

meta_sub = meta_sub.dropna(subset=["age_num"])


# ---------- align methylation data ----------
sample_ids = meta_sub["GSM"].values

genes_sub = gene_d[sample_ids].T.copy()   # rows = samples, cols = CpGs


# ---------- sort by age ----------
order = np.argsort(meta_sub["age_num"].values)

meta_sub = meta_sub.iloc[order].reset_index(drop=True)
genes_sub = genes_sub.iloc[order].reset_index(drop=True)


# ---------- build X: time x observations x features ----------
ages_unique = np.sort(meta_sub["age_num"].unique())

X_list = []

for age in ages_unique:
    idx = meta_sub["age_num"] == age
    X_age = genes_sub.loc[idx].values.astype(float)
    X_list.append(X_age)

min_obs = 3#min(mat.shape[0] for mat in X_list)

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


#sel_genes = [10, 20]
# keep same number of observations per time point
for cur_subset_genes in config.n_sel_features:

    select_age_associated_features(X, dataset_id, filtered_ages, cur_subset_genes)
#select_age_associated_features(X, dataset_id, filtered_ages, 20)
#select_age_associated_features(X, dataset_id, filtered_ages, 100)

aa = 1
##############

####################################################
####### STep 2: PRE-PROCESS DATA SO ITS IN THE SHAPE OF 
###########  TIME X SUBJECT X FEATURES (GENE EXPRESSION VALUES)
#############################################################
