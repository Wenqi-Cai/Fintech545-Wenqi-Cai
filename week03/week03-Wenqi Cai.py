import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#problem 1
data = pd.read_csv('DailyReturn.csv')

lambdas = np.linspace(0.1, 0.99, 10) 

def exp_weighted_covariance_matrix(data, lamb):
    returns = data.values
    n, m = returns.shape 
    weighted_cov = np.zeros((m, m))  

    mean_returns = np.mean(returns, axis=0)

    weights = np.array([lamb**(n-i) for i in range(1, n+1)])
    weights /= weights.sum()  

    for i in range(n):
        diff = returns[i] - mean_returns
        weighted_cov += weights[i] * np.outer(diff, diff)
    
    return weighted_cov

# PCA
def plot_pca_variance(cov_matrix, lamb):
    pca = PCA()
    pca.fit(cov_matrix)
    
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.plot(explained_variance, label=f'lamb={lamb}')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained by Each Eigenvalue')
    plt.legend()

#
plt.figure(figsize=(10, 6))
for lamb in lambdas:
    cov_matrix = exp_weighted_covariance_matrix(data, lamb)
    plot_pca_variance(cov_matrix, lamb)

plt.show()

#problem 2
import numpy as np
import time
from scipy.linalg import norm

#
def chol_psd(a):
    n = a.shape[0]
    root = np.zeros_like(a)

    for j in range(n):
        s = 0.0
     
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        temp = a[j, j] - s
        
        if 0 >= temp >= -1e-8:
            temp = 0.0
        
        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir

    return root

#
def generate_non_psd_matrix(n):
    np.random.seed(42)  
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  
    np.fill_diagonal(A, 1)  
    return A

# Higham 
def higham_psd(a, max_iter=100):
    n = a.shape[0]
    W = np.identity(n)
    delta_S = np.zeros_like(a)
    Y = np.copy(a)
    for _ in range(max_iter):
        R = Y - delta_S
        X = np.copy(R)
        vals, vecs = np.linalg.eigh(X)
        vals[vals < 0] = 0
        X = vecs @ np.diag(vals) @ vecs.T
        delta_S = X - R
        Y = W @ X @ W
    return Y

#near_psd
def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    out = np.copy(a)
    invSD = None

    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)

    T = 1.0 / np.sqrt(vecs @ np.diag(vals) @ vecs.T)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

#test and compare
def test_and_compare(n):
    
    A = generate_non_psd_matrix(n)
    
    #near_psd
    start_time = time.time()
    near_psd_result = near_psd(A)
    near_psd_time = time.time() - start_time
    
    #Higham
    start_time = time.time()
    higham_psd_result = higham_psd(A)
    higham_time = time.time() - start_time
    
    #Frobenius
    frobenius_near_psd = norm(A - near_psd_result, 'fro')
    frobenius_higham_psd = norm(A - higham_psd_result, 'fro')

    #outcome
    print(f"Matrix size: {n}x{n}")
    print(f"near_psd runtime: {near_psd_time:.5f} seconds, Frobenius norm: {frobenius_near_psd:.5f}")
    print(f"Higham runtime: {higham_time:.5f} seconds, Frobenius norm: {frobenius_higham_psd:.5f}")
    
    return near_psd_time, higham_time, frobenius_near_psd, frobenius_higham_psd

#
matrix_sizes = [100, 200, 500]
for size in matrix_sizes:
    test_and_compare(size)


#problem 3
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA
import time

# 
data = pd.read_csv('DailyReturn.csv')
returns = data.values

#
pearson_corr = np.corrcoef(returns, rowvar=False)
pearson_var = np.var(returns, axis=0)

#
def exponentially_weighted_covariance(data, lamb=0.97):
    weights = np.array([lamb**i for i in range(len(data))])
    weights = weights[::-1] / weights.sum()
    weighted_returns = data * weights[:, np.newaxis]
    return np.cov(weighted_returns, rowvar=False)

ew_cov = exponentially_weighted_covariance(returns)

# 
# cov1: Pearson correlation + Pearson variance
cov1 = pearson_corr * np.outer(np.sqrt(pearson_var), np.sqrt(pearson_var))

# cov2: Pearson correlation + Exponentially weighted variance
ew_var = np.diag(ew_cov)
cov2 = pearson_corr * np.outer(np.sqrt(ew_var), np.sqrt(ew_var))

# cov3: Exponentially weighted covariance matrix
cov3 = ew_cov

# cov4: Exponentially weighted covariance with Pearson variance
cov4 = ew_cov * np.outer(np.sqrt(pearson_var), np.sqrt(pearson_var))

cov_matrices = [cov1, cov2, cov3, cov4]

# 
def simulate_data(cov_matrix, n_draws=25000, method='direct', explained_variance=1.0):
    if method == 'direct':
        mean = np.zeros(cov_matrix.shape[0]) 
        return np.random.multivariate_normal(mean, cov_matrix, size=n_draws)
    elif method == 'pca':
        pca = PCA()
        pca.fit(np.linalg.cholesky(cov_matrix))
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(explained_var, explained_variance) + 1 
        components = pca.components_[:n_components]
        # 
        projected_data = np.random.randn(n_draws, n_components) @ components
        return projected_data

# 
def compare_frobenius_norms(original_cov, simulated_data):
    simulated_cov = np.cov(simulated_data, rowvar=False)
    return norm(original_cov - simulated_cov, 'fro')

# 
n_draws = 25000
methods = ['direct', 'pca_100', 'pca_75', 'pca_50']
explained_variances = [1.0, 0.75, 0.50, 0.50]  # 

for i, cov_matrix in enumerate(cov_matrices):
    print(f"Covariance Matrix {i+1}")
    for method, explained_var in zip(methods, explained_variances):
        start_time = time.time()
        if method == 'direct':
            simulated_data = simulate_data(cov_matrix, n_draws, method='direct')
        else:
            simulated_data = simulate_data(cov_matrix, n_draws, method='pca', explained_variance=explained_var)
        
        frobenius_norm = compare_frobenius_norms(cov_matrix, simulated_data)
        runtime = time.time() - start_time
        print(f"Method: {method}, Frobenius Norm: {frobenius_norm:.5f}, Runtime: {runtime:.5f} seconds")
    print("\n")
