import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *



# mice, RNAseq data from blood and muscle , 6-30 weeks,  ~15-20 observationsper time  point
dataset_id = 'GSE132741' 

# A rat epigenetic clock with methylation, ages 1-27 months 
#dataset_id = 'GSE161141' 

family_soft_name =  f"clinical-data(doita)/{dataset_id}_family.soft"
series_matrix_name =  f"clinical-data(doita)/{dataset_id}_series_matrix.txt"


# get meta data and gene expression (gene expression is usually empty though, but keep it just in case)
meta_d, gene_d = get_metadata(family_soft_name,  series_matrix_name)

# now, read the extracted file with gene data (should only be used after running an .sh script for extraction) and put it together in a table
genes_table = get_extracted_table(dataset_id, 'txt') 


#####33 prepocess: (ofc i have to do it separately AGAIN for each dataset oh well)
actual_data = genes_table 
#######3 normalize to get relative percentage 
actual_data =  sigmoid(actual_data) #(actual_data - actual_data.min()) / (actual_data.max() - actual_data.min())
#actual_data = actual_data.div(actual_data.sum(axis=1), axis=0) * 100

actual_data.index = actual_data.index.str.split("_").str[0]
meta_data = meta_d

meta_data = meta_data.set_index("GSM")
common_ids = actual_data.index.intersection(meta_data.index)
actual_data = actual_data.loc[common_ids]
meta_data = meta_data.loc[common_ids]

# make integer week groups: 6.4 -> 6, 6.1 -> 6 ---
meta_data = meta_data.copy()
meta_data["week_group"] = np.floor((meta_data["week"]).astype(int))

# keep only weeks with at least 4 observations ---
valid_weeks = meta_data["week_group"].value_counts()
valid_weeks = valid_weeks[valid_weeks >= 4].index.sort_values()

meta_filt = meta_data[meta_data["week_group"].isin(valid_weeks)].copy()
data_filt = actual_data.loc[meta_filt.index]

# take exactly 4 observations per week
meta_4 = (
    meta_filt
    .sort_index()
    .groupby("week_group", group_keys=False)
    .head(4)
)

data_4 = data_filt.loc[meta_4.index]

# time series X: time_points x observations x features ---
weeks = sorted(meta_4["week_group"].unique())
n_time = len(weeks)
n_obs = 4
n_feat = data_4.shape[1]

X = np.empty((n_time, n_obs, n_feat), dtype=data_4.to_numpy().dtype)

for i, w in enumerate(weeks):
    ids = meta_4.index[meta_4["week_group"] == w]
    X[i] = data_4.loc[ids].to_numpy()
    

# take a few features as befoer
n_features = 5
X_sub = X[:, :, np.arange(n_features)]


np.save(f"X_real_val_{dataset_id}.npy" , X_sub)

aa = 1