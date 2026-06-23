import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *


# A rat epigenetic clock with methylation, ages 1-27 months 
dataset_id = 'GSE161141' 

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"


# get meta data and gene expression (gene expression is usually empty though, but keep it just in case)
meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)


#aa = get_extracted_table_cgmap(dataset_id, 'CGmap', 10000) 
# now, read the extracted file with gene data (should only be used after running an .sh script for extraction) and put it together in a table
genes_table = pd.read_csv("df_rats.csv", index_col=0) #get_extracted_table_cgmap(dataset_id, 'CGmap', 10000) 


#####33 prepocess: (ofc i have to do it separately AGAIN for each dataset oh well)
actual_data = genes_table 
#######3 normalize to get relative percentage 
###########3 was sigmoid but its all large and positive so minmax is better
actual_data = actual_data if (actual_data.all().min(skipna=True) >= 0 and actual_data.all().max(skipna=True) <= 1) else minmax(actual_data)
#actual_data =  sigmoid(actual_data) #(actual_data - actual_data.min()) / (actual_data.max() - actual_data.min())
#actual_data = actual_data.div(actual_data.sum(axis=1), axis=0) * 100

actual_data.index = actual_data.index.str.split("_").str[0]
meta_data = meta_d

meta_data = meta_data.set_index("GSM")
common_ids = actual_data.index.intersection(meta_data.index)
actual_data = actual_data.loc[common_ids]
meta_data = meta_data.loc[common_ids]

# make integer week groups: 6.4 -> 6, 6.1 -> 6 ---
meta_data = meta_data.copy()
meta_data["month_group"] = np.floor(meta_data["age (months)"].astype(float))

n_obs = 5
# keep only weeks with at least 4 observations ---
valid_weeks = meta_data["month_group"].value_counts()
valid_weeks = valid_weeks[valid_weeks >= n_obs].index.sort_values()

meta_filt = meta_data[meta_data["month_group"].isin(valid_weeks)].copy()
data_filt = actual_data.loc[meta_filt.index]

# take exactly 4 observations per week
meta_4 = (
    meta_filt
    .sort_index()
    .groupby("month_group", group_keys=False)
    .head(n_obs)
)

data_4 = data_filt.loc[meta_4.index]

# time series X: time_points x observations x features ---
weeks = sorted(meta_4["month_group"].unique())
n_time = len(weeks)
n_obs = n_obs
n_feat = data_4.shape[1]

X = np.empty((n_time, n_obs, n_feat), dtype=data_4.to_numpy().dtype)

for i, w in enumerate(weeks):
    ids = meta_4.index[meta_4["month_group"] == w]
    X[i] = data_4.loc[ids].to_numpy()
    
filtered_ages = weeks

import config

for cur_subset_genes in config.n_sel_features:

    select_age_associated_features(X, dataset_id, filtered_ages, cur_subset_genes)
#select_age_associated_features(X, dataset_id, filtered_ages, 10)
#select_age_associated_features(X, dataset_id, filtered_ages, 20)
#select_age_associated_features(X, dataset_id, filtered_ages, 100)

# take a few features as befoer

aa = 1