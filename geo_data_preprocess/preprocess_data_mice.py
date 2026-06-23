import pandas as pd
from tqdm import tqdm
import numpy as np

from methylprep import run_pipeline

import glob
from tqdm import tqdm

from pathlib import Path

# connect my modules
from kernel_estimation import *


def temporal_feature_correlations(X):
    """
    X shape: time x observations x features
    """

    T, n_obs, d = X.shape

    corrs = np.zeros((T - 1, d))

    for t in range(T - 1):

        for j in range(d):

            x1 = X[t, :, j]
            x2 = X[t + 1, :, j]

            corrs[t, j] = np.corrcoef(x1, x2)[0, 1]

    corrs = np.nan_to_num(corrs, nan=0.0)

    # average over time
    mean_corrs = corrs.mean(axis=0)

    return mean_corrs


def select_age_associated_features(
    X,
    dataset_id,
    ages,
    n_features=100,
    seed=None
): 
    n_bootstrap = 100
    save_dir = Path(__file__).resolve().parents[1] / "all_data_files"
    save_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)

    for b in range(n_bootstrap):

        idx = rng.choice(
            X.shape[2],
            size=n_features,
            replace=False
        )

        print(idx)

        X_sub = X[:, :, idx]

        np.save(
            save_dir / f"X_real_val_{n_features}_{dataset_id}_boot{b}.npy",
            X_sub
        )


# def select_age_associated_features(
#     X,
#     dataset_id,
#     ages,
#     n_features=100
# ):
#     """
#     Select CpGs with strongest correlation to age.

#     Parameters
#     ----------
#     X : ndarray
#         Shape (time, observations, features)

#     ages : ndarray
#         Shape (time,)

#     dataset_id : str
#         GEO dataset identifier

#     n_features : int
#         Number of CpGs to select
#     """

#     save_dir = (
#         Path(__file__).resolve().parents[1]
#         / "all_data_files"
#     )

#     save_dir.mkdir(exist_ok=True)

#     # mean trajectory per feature
#     M = X.mean(axis=1)   # shape: time x features

#     corrs = np.zeros(X.shape[2])

#     # correlation of each CpG trajectory with age
#     for j in range(X.shape[2]):

#         corrs[j] = np.corrcoef(
#             M[:, j],
#             ages
#         )[0, 1]

#     corrs = np.nan_to_num(corrs, nan=0.0)

#     # strongest age-associated CpGs
#     idx = np.argsort(np.abs(corrs))[-n_features:]

#     X_sub = X[:, :, idx]

#     np.save(
#         save_dir / f"X_real_val_{n_features}_{dataset_id}.npy",
#         X_sub
#     )

#     return X_sub


# normalize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def get_metadata(family_soft_name,  series_matrix_name):

    # some data stored in such a shitty way that i cant even read the rows, just return an empty df i dont need it in any case
    try:
        df_gene_expr = pd.read_csv(
            series_matrix_name,
            sep="\t",
            comment="!",   # removes GEO metadata lines
            index_col=0    # genes as index
        )
    except Exception:
        df_gene_expr = pd.DataFrame()
    

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
    
    return meta_d, df_gene_expr




def get_extracted_table(data_id, data_extension, sel_features = False, n_sel_features = 0):
    '''
    takes 2 files, data_extension is like txt or csv
    '''

    files = glob.glob(f"extracted_{data_id}/*.{data_extension}")
    #files = glob.glob("extracted/*.csv")

    # sort them, rn its a mess
    files = sorted(files, key=lambda x: int(x.split("GSM")[1].split("_")[0]))

    rows = []
    sample_names = None

    for f in tqdm(files, desc="Processing"):
 
        if data_extension == 'txt':
            df = pd.read_csv(f, sep = '\s+')
        else:
            df = pd.read_csv(f)
        
        # if the number of genes is too big, just cut it to some subset
        if sel_features == False:
            n_sel_features = df.shape[0]
        # first column = gene IDs, second = expression
        if sample_names is None:
            sample_names = df.iloc[:, 0][:n_sel_features]  # gene IDs
        
        # if its a csv, then most likely its 2 columns
        data_col_ind = len(df.columns) - 1
        if data_id == 'GSE120132':
            expr = df.iloc[:, 5][:n_sel_features].values


        else:
            expr = df.iloc[:, 1][:n_sel_features].values  # expression column

        rows.append(expr)

    # build dataframe
    df_all = pd.DataFrame(rows, columns=sample_names)

    # set index as sample names
    df_all.index = [f.split("/")[-1].replace(".csv", "") for f in files]
    
    return df_all


def get_extracted_table_cgmap(data_id, data_extension, n_selected_genes):
    '''
    extracts CGmap data for selected genes
    '''

    files = glob.glob(f"extracted_{data_id}/*.{data_extension}")
    #files = glob.glob("extracted/*.csv")

    # sort them, rn its a mess
    files = sorted(files, key=lambda x: int(x.split("GSM")[1].split("_")[0]))

    rows = []
    sample_names = None

    cols = [
    "chr", "base", "pos", "context", "dinuc",
    "meth_ratio", "meth_count", "coverage"
    ]

    for f in tqdm(files, desc="Processing"):
 
        df = pd.read_csv(
        f,
        sep="\t",
        header=None,
        names=cols,
    )
    
        # first column = gene IDs, second = expression
        if sample_names is None:
            sample_names = df.iloc[:, 0][:n_selected_genes]  # gene IDs
        
        # if its a csv, then most likely its 2 columns
        data_col_ind = len(df.columns) - 1
        expr = np.array(df['meth_ratio'][:n_selected_genes]) #df.iloc[:, 1].values  # expression column
        
        rows.append(expr)

    # build dataframe
    df_all = pd.DataFrame(rows, columns=sample_names)

    # set index as sample names
    df_all.index = [f.split("/")[-1].replace(".csv", "") for f in files]

    df_all.to_csv("df_rats.csv")
    
    return df_all