import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import pandas as pd

# connect my modules
from kernel_estimation import *
from geo_data_preprocess import *
from common.common_modules import compute_A_from_X


from scipy.stats import truncnorm




#SEED = 42

SEED = 100

from pathlib import Path



def sample_truncated_mvnormal(mean_next, Sigma, rng=np.random):
    """
    Sample from N(mean_next, Sigma) 

    mean_next : (num_paths, state_dim)
    Sigma     : (state_dim, state_dim)
    """

    num_paths, state_dim = mean_next.shape

    accepted = np.zeros_like(mean_next)
    done = np.zeros(num_paths, dtype=bool)

    while not np.all(done):

        idx = np.where(~done)[0]
        n_left = len(idx)

        proposal = rng.multivariate_normal(
            mean=np.zeros(state_dim),
            cov=Sigma,
            size=n_left
        ) + mean_next[idx]

        valid = np.all((proposal >= 0) & (proposal <= 1), axis=1)

        accepted[idx[valid]] = proposal[valid]
        done[idx[valid]] = True

    return accepted


def run_entire_simulation(dataset_id, n_sel_features, which_bias = 'FSMR'):

    save_dir = Path(__file__).resolve().parents[1] / "all_data_files"

    ###333 do like this: use real data as initial state and match dimensions
    # mice, RNAseq data from blood and muscle , 6-30 weeks,  ~15-20 observationsper time  point
    #dataset_id = 'GSE132741' 

    # rats, methylaiton 
    #dataset_id = 'GSE161141' 

    # mice, methylation 0.67-35 months, 6 observations per time point
    #dataset_id =  'GSE80672'


    #  load real data, shape [time x Patients x Genes]
    #X_real_val = np.load(f"X_real_val_30_{dataset_id}.npy")

    X_real_val = np.load(f"all_data_files/X_real_val_{n_sel_features}_{dataset_id}.npy")
    
    time_steps, num_paths, state_dim = X_real_val.shape
    # ---------------- parameters ----------------
    #state_dim   = 5
    ## age is modelled up to a 100 years
    #time_steps  = 100 # 100
    #num_paths   = 1000

    scale =  (100 / n_sel_features)**0.5  #1 + 0.35 * np.exp(-(n_sel_features - 5) / 20)

    bias_strength = 1 * scale
    drift_strength = 1.0 * scale

    """ bias_strength = 0.8 * (100 / n_sel_features)**0.25 #10 #/ time_steps # 400
    #drift_strength = 0.6 #400/time_steps
    
    drift_strength =  1 * (100 / n_sel_features)**0.25 #0.6 * (100 / n_sel_features)**0.25 """
    
    noise_scale = 1.0 / state_dim

    power = 0.5
    # -------------------------------------------
    #### damage
    D_t = 0

    # identity 2D matrix
    Sigma = noise_scale * np.eye(state_dim)
    #np.random.seed(SEED)
    X_0 = np.random.randn(num_paths, state_dim)

    X_biased   = X_0.copy()
    X_unbiased = X_0.copy()


    # Discretization
    def bin_states(X, bin_width):
        """
        Map continuous states X (n, d) to discrete bin indices (n, d) by rounding to grid.
        """
        return np.floor(X / bin_width).astype(np.int64)

    def rows_to_tuples(B):
        """
        Convert (n, d) integer array to a set of tuples for hashing/unique counting.
        """
        return set(map(tuple, B))
    # ------------------------------------------------------------

    # ------------------ Idea 4: store full trajectories ------------------

    Xb_hist = np.zeros((time_steps, num_paths, state_dim))
    Xu_hist = np.zeros((time_steps, num_paths, state_dim))

    # initialize both processes identicall
    Xb_hist[0] = X_real_val[0]
    Xu_hist[0] = X_real_val[0]

    # compute population mean globally
    # earliest age or oldest age?
    # try oldest
    mu_global = X_real_val[0].mean(axis=0)

    for t in range(1, time_steps):
        #np.random.seed(t*3)

        # transofrm to 0-1
        #X_biased = logit(X_biased)
        #X_unbiased = logit(X_biased)

        noise = np.random.multivariate_normal(
            mean=np.zeros(state_dim),
            cov=Sigma,
            size=num_paths
        )

        A_t = compute_A_from_X(X_0, bias_strength) # was X_biased, buttt # need X_0, not X_t
        #A_t = compute_A_from_X(X_biased, bias_strength) 

        # biased process
        mu = mu_global #X_biased.mean(axis=0)              
        sd = X_biased.std(axis=0, ddof=1)
        
        # try different drift approach
        # mean-reverting: this is weaker
        # b_t = drift_strength * (mu - X_biased)

        # mean-repelling 
        #b_t = drift_strength * (X_biased - mu)

       
        
        # using this        
        b_t = drift_strength * (X_biased - mu) #was like thaaaaaaat
       
       
        #b_t = drift_strength * (mu / (sd + 1e-12)) # was like this  # drift direction
        # damage (is not needed probably)
        #D_t += t

        #X_biased = X_biased + b_t + noise
        ##X_biased = X_biased - (X_biased @ A_t.T) + noise

        ##### new idea
        #diff = mu - X_biased

        # nonlinear drift: large deviations are corrected more strongly
        #b_t = drift_strength * np.sign(diff) * np.abs(diff)**power
        #X_biased = X_biased + b_t + noise
        

        # try this: instead of clipping, sample from truncated normal, then its within [0, 1]
        if which_bias == 'FSMR':
            mean_next_b = X_biased - (X_biased @ A_t.T) + b_t
        
        elif which_bias == 'LMR':
            mean_next_b = X_biased + b_t

        elif which_bias == 'NMR':
            mean_next_b = X_biased - (X_biased**2) + b_t

        elif which_bias == 'FS':
            mean_next_b = X_biased - (X_biased @ A_t.T)   

        

        ##### to tets a few options 
        #1. mean_next_b_1 = X_biased - (X_biased @ A_t.T) + b_t
        #2. mean_next_b_2 = X_biased + b_t 
        #3. mean_next_b_3 = X_biased - (X_biased**2) + b_t
        #4. mean_next_b_3 = X_biased - (np.sqrt(X_biased)) + b_t
    


        X_biased = sample_truncated_normal(
        mean_next_b,
        Sigma
      )


        #X_biased = X_biased - (X_biased @ A_t.T) + b_t + noise # was like thaaaaat
        #X_biased = np.clip(X_biased, 0, 1)
        
        #X_biased = np.minimum(np.abs(X_biased), 1)

        #X_biased = X_biased - (X_biased**2) + b_t + noise 
        #X_biased = X_biased - np.sqrt(X_biased) + b_t + noise
        
        
        ######################
        # unbiased reference
        mean_next_u = X_unbiased

        X_unbiased = sample_truncated_normal(
            mean_next_u,
            Sigma
        )
        #X_unbiased = X_unbiased + noise
        #X_unbiased = np.clip(X_unbiased, 0, 1)
        
        
        #X_unbiased = np.minimum(np.abs(X_unbiased), 1)

        # idea 4: store full trajectories
        Xb_hist[t] = X_biased
        Xu_hist[t] = X_unbiased


    aa = 1
    # normalize 0-1
    #Xb_hist[1:] = minmax(Xb_hist[1:]) # change sigmoid for minmax
    #Xu_hist[1:] = minmax(Xu_hist[1:])

    ##############  EXPERIMENT 2 ########################33
    ############### TRANSITION PROBABILITY ESTIMATION #####

    # ============================================================
    # 1) UNBIASED KERNEL
    # Model:
    #   X_{t+1}^u ~ N(x, Sigma_u_hat)
    # so estimate Sigma from increments
    # ============================================================

    Xu_val = Xu_hist.copy() #[80:]
    # save 
    np.save("all_data_files/X_unbiased_val.npy", Xu_val)

    ########33 BIASED KERNEL  #####3

    Xb_val = Xb_hist.copy() #[80:]

    # save 
    np.save(f"all_data_files/X_biased_val_{which_bias}.npy", Xb_val)
    #np.save("all_data_files/X_biased_pred.npy", Xb_future_all)
