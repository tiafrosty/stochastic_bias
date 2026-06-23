import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

# connect my modules
from kernel_estimation import *

from pathlib import Path


import subprocess
import sys

dataset_ids = [
    #"GSE164127",  # bats methylation
    #"GSE223748",  # cross-species mammalian methylation
    #"GSE120132",  # mouse kidney methylation
    "GSE161141"  # rat methylation
    #"GSE80672"    # mouse methylation
]

for cur_dataset in dataset_ids:
    print(f'dataset {cur_dataset} is running...')
    subprocess.run([sys.executable, f"real_data_{cur_dataset}.py"])
    print('...Done')
#subprocess.run([sys.executable, "file2.py"])
#subprocess.run([sys.executable, "file3.py"])



