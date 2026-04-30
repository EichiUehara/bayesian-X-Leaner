import numpy as np
import urllib.request
import os

def simulate_level1_algebraic_sanity(N=1000):
    P = 5
    X = np.random.normal(0, 1, size=(N, P))
    confounding = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    pi_true = 1 / (1 + np.exp(-confounding))
    W = np.random.binomial(1, pi_true)
    tau = 2.0
    Y0 = 1.0 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * X[:, 2] + np.random.normal(0, 0.1, size=N)
    Y1 = Y0 + tau
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, np.full(N, tau)

def simulate_level2_sparsity_stress(N=500, P=2000):
    X = np.random.normal(0, 1, size=(N, P))
    confounding_W = 2.0 * X[:, 0] + 2.0 * X[:, 1] + 2.0 * X[:, 2]
    pi_true = 1 / (1 + np.exp(-confounding_W))
    pi_true = np.clip(pi_true, 0.05, 0.95)
    W = np.random.binomial(1, pi_true)
    Y0 = 3.0 * np.sin(X[:, 3]) + 2.0 * np.exp(-np.abs(X[:, 4])) + 1.5 * X[:, 5] + np.random.normal(0, 0.5, size=N)
    tau = 2.0 + 1.5 * X[:, 3] - X[:, 4]**2
    Y1 = Y0 + tau
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, tau

def get_level3_ihdp():
    url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    filepath = "ihdp_npci_1.csv"
    if not os.path.exists(filepath):
        print(f"Downloading IHDP dataset to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    data = np.loadtxt(filepath, delimiter=',')
    W = data[:, 0].astype(int)
    Y = data[:, 1]
    mu_0 = data[:, 3]
    mu_1 = data[:, 4]
    X = data[:, 5:]
    tau = mu_1 - mu_0
    return X, Y, W, tau

def get_level4_acic_hostile():
    N = 1000
    P = 20
    X = np.random.normal(0, 1, size=(N, P))
    confounding = 10.0 * X[:, 0] + 5.0 * X[:, 1]
    pi_true = 1 / (1 + np.exp(-confounding))
    W = np.random.binomial(1, pi_true)
    Y0 = 5.0 * np.log(np.abs(X[:, 2]) + 1) * np.sign(X[:, 2]) + np.random.normal(0, 1, size=N)
    tau = 10.0 * np.sin(X[:, 0] * X[:, 1]) + 5.0 * np.where(X[:, 3] > 0, 1, 0)
    Y1 = Y0 + tau
    Y = np.where(W == 1, Y1, Y0)
    return X, Y, W, tau

def simulate_level5_imbalance(N=1000):
    P = 10
    X = np.random.normal(0, 1, size=(N, P))
    logits = 3.0 + 0.5 * X[:, 0] - 0.2 * X[:, 1]
    pi = 1 / (1 + np.exp(-logits))
    W = np.random.binomial(1, pi)
    true_cate = 1.0 + X[:, 2]
    mu_0 = X[:, 0]**2 + X[:, 1]
    Y = mu_0 + W * true_cate + np.random.normal(0, 0.5, size=N)
    return X, Y, W, true_cate

def simulate_level6_unobserved_confounding(N=1000, confounding_strength=2.0):
    P = 5
    X = np.random.normal(0, 1, size=(N, P))
    U = np.random.normal(0, 1, size=N)
    logits = 0.5 * X[:, 0] + confounding_strength * U
    pi = 1 / (1 + np.exp(-logits))
    pi = np.clip(pi, 0.05, 0.95)
    W = np.random.binomial(1, pi)
    true_cate = 2.0
    mu_0 = X[:, 1] + confounding_strength * U
    Y = mu_0 + W * true_cate + np.random.normal(0, 0.5, size=N)
    return X, Y, W, np.full(N, true_cate)

def simulate_level7_heteroskedasticity(N=1000):
    P = 5
    X = np.random.normal(0, 1, size=(N, P))
    logits = 0.5 * X[:, 0]
    pi = 1 / (1 + np.exp(-logits))
    W = np.random.binomial(1, pi)
    true_cate = 2.0
    mu_0 = X[:, 1]
    noise_std = np.exp(0.5 * X[:, 0]) 
    Y = mu_0 + W * true_cate + np.random.normal(0, noise_std, size=N)
    return X, Y, W, np.full(N, true_cate)

def simulate_level8_weak_signal(N=200):
    P = 5
    X = np.random.normal(0, 1, size=(N, P))
    pi = np.full(N, 0.5)
    W = np.random.binomial(1, pi)
    true_cate = 0.01
    mu_0 = 0.0
    Y = mu_0 + W * true_cate + np.random.normal(0, 5.0, size=N)
    return X, Y, W, np.full(N, true_cate)

def simulate_level10_null_effect(N=1000):
    P = 10
    X = np.random.normal(0, 1, size=(N, P))
    logits = 0.5 * X[:, 0] - 0.5 * X[:, 1]
    pi = 1 / (1 + np.exp(-logits))
    W = np.random.binomial(1, pi)
    true_cate = np.zeros(N)
    mu_0 = X[:, 0]**2 + 2.0 * np.sin(X[:, 2])
    Y = mu_0 + W * true_cate + np.random.normal(0, 2.0, size=N)
    return X, Y, W, true_cate

def simulate_level11_discontinuity(N=1000):
    P = 5
    X = np.random.normal(0, 1, size=(N, P))
    W = np.random.binomial(1, 0.5, size=N)
    true_cate = np.where(X[:, 0] > 0, 5.0, 0.0)
    mu_0 = X[:, 1]
    Y = mu_0 + W * true_cate + np.random.normal(0, 0.5, size=N)
    return X, Y, W, true_cate
