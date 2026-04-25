import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
from tqdm import tqdm

# connect my modules
from kernel_estimation import *

# normalize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def get_metadata(family_soft_name,  series_matrix_name):

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
    
    return meta_d, df_gene_expr




def get_extracted_table(data_id, data_extension):
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
        
        # first column = gene IDs, second = expression
        if sample_names is None:
            sample_names = df.iloc[:, 0]  # gene IDs
        
        # if its a csv, then most likely its 2 columns
        data_col_ind = len(df.columns) - 1
        expr = df.iloc[:, 1].values  # expression column
        
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