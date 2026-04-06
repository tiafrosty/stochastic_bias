import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from tqdm import tqdm

# -----------------------------
# Parameters
# -----------------------------
n_features = 2000         # number of features (like CpGs)
max_age = 100            # simulated ages: 1..max_age
samples_per_age = 5      # how many samples per age for training
test_samples_per_age = 5 # how many samples per age for testing
sigma = 0.05             # std dev of noise N(0, sigma^2)
random_seed = 42

# test different sigmas
all_sigmas = [0.025, 0.05, 0.1, 0.2]

rng = np.random.default_rng(random_seed)

# -----------------------------
# Helper functions
# -----------------------------
def clamp01(x):
    """Clamp array values to [0, 1]."""
    return np.clip(x, 0.0, 1.0)

def generate_sample_at_age(ground_state, n_features, age, sigma, rng):
    """
    Generate one sample at a given 'simulated age':
    X_age = ground_state + sum_{k=1}^age epsilon_k,
    where epsilon_k ~ N(0, sigma^2), independently.
    Values are clamped to [0, 1].
    """
    x = ground_state
    for _ in range(age):
        eps = rng.normal(0.0, sigma, size=n_features)
        x += eps
        x = clamp01(x)
    #noise = rng.normal(loc=0.0, scale=sigma, size=(age, n_features))
    #x = ground_state + noise.sum(axis=0)
    #x = clamp01(x)
    return x

# -----------------------------
# 1) Create ground state
# -----------------------------
# initial state for each featuer: random values in [0, 1]
ground_state = rng.random(n_features)

#generate_sample_at_age(ground_state, 60, 0.05)
model = ElasticNetCV(
        l1_ratio = np.arange(0.1, 1, 0.1), #[0.1, 0.3, 0.5, 0.7, 0.9],
        cv=5,
        n_alphas=50,
        random_state=random_seed
    )


def gen_and_plot_BM(n_features):
    n = 1000
    all_x = []
    ground_state = rng.random(n_features)
    for age in tqdm(range(1, max_age + 1)):
        all_x_cur_n = []
        for _ in range(n):
                # train always on 0.2
            x = generate_sample_at_age(ground_state, n_features, age, 0.2, rng)
            all_x_cur_n.append(float(x))
        all_x.append(np.mean(all_x_cur_n))
        
    return all_x


all_x = gen_and_plot_BM(1)

aa = 1
# Generate training data only once!
X_train = []
y_train = []

for age in range(1, max_age + 1):
    for _ in range(samples_per_age):
        # train always on 0.2
        x = generate_sample_at_age(ground_state, age, 0.2, rng)
        X_train.append(x)
        y_train.append(age)

X_train = np.vstack(X_train)
y_train = np.array(y_train, dtype=float)

####3 only test
def generate_data(sigma):

    # -----------------------------
    # 3) Generate test data
    # -----------------------------
    # test for various sigma
    X_test = []
    y_test = []

    for age in  tqdm(range(1, max_age + 1), desc="Generating test samples"):
        for _ in range(test_samples_per_age):
            x = generate_sample_at_age(ground_state, age, sigma, rng)
            X_test.append(x)
            y_test.append(age)

    X_test = np.vstack(X_test)
    y_test = np.array(y_test, dtype=float)

    print(f"Training samples: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Test samples:     {X_test.shape[0]}, features: {X_test.shape[1]}")

    # -----------------------------
    # 4) Train ElasticNet clock
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # 5) Evaluate on test data
    # -----------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_r, pearson_p = pearsonr(y_test, y_pred)

    print("\n=== Test set performance ===")
    print(f"MAE (|predicted - true age|): {mae:.3f}")
    print(f"R^2:                          {r2:.3f}")
    print(f"Pearson r:                    {pearson_r:.3f} (p = {pearson_p:.2e})")
    
    return y_test, y_pred


plt.figure(figsize=(7, 6))

for cur_sigma in all_sigmas:
    y_test, y_pred = generate_data(cur_sigma)

    # Scatter for this sigma
    plt.scatter(y_test, y_pred, alpha=0.5, label=f"sigma={cur_sigma}")

# Identity line (perfect prediction)
#min_age = min(y_test.min() for cur_sigma in all_sigmas for y_test, _ in [generate_data(cur_sigma)])
#max_age = max(y_test.max() for cur_sigma in all_sigmas for y_test, _ in [generate_data(cur_sigma)])

#plt.plot([min_age, max_age],
#         [min_age, max_age],
#         color='black', linestyle='--', linewidth=2,
#         label='Identity line')

plt.xlabel("Simulated age (true)")
plt.ylabel("Predicted age")
plt.title("Toy stochastic aging clock\n(predicted vs true age)")
plt.legend()
plt.tight_layout()
plt.show()


#generate_data(all_sigmas[1])

aa = 1