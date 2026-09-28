import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
np.random.seed(42)

from typing import Literal
from beartype import beartype
from scipy.optimize import minimize

@beartype
def make_positive_semifinite_matrix(cov_matrix: np.ndarray, calculation_type: Literal["fast", "long"] = "fast") -> np.ndarray:
    @beartype
    def B(theta: np.ndarray):
        cosine_B = np.cos(theta)
        cosine_B_shifted = np.c_[cosine_B, np.ones(cosine_B.shape[0])]
        sin_B = np.sin(theta)
        sin_B_shifted = np.c_[np.ones(sin_B.shape[0]), sin_B]
        sin_B_shifted_cumprod = np.cumprod(sin_B_shifted, axis=-1)
        return cosine_B_shifted * sin_B_shifted_cumprod
    
    @beartype
    def reshape_theta_to_appropriate_size(theta: np.ndarray, n: int):
        theta_size = theta.shape[0]
        rows_dim = int(n)
        cols_dim = int(theta_size / rows_dim)
        return theta.reshape(rows_dim, cols_dim)
    
    @beartype
    def l2_norm(theta: np.ndarray, C: np.ndarray):
        theta = reshape_theta_to_appropriate_size(theta=theta, n=C.shape[0])
        B_ = B(theta)
        C_approx = np.dot(B_, B_.T)
        return np.linalg.norm(C_approx - C, ord = 'fro')
    
    @beartype
    def eigenvalue_clipping(matrix: np.ndarray, delta: float = 1e-8, jitter: float = 1e-6) -> np.ndarray:
        # eigenvalues, eigenvectors = np.linalg.eig(matrix)
        # positive_eigenvalues = np.where(eigenvalues < 0, delta, eigenvalues)
        # Q, Λ = eigenvectors, np.diag(positive_eigenvalues)
        # return np.dot(np.dot(Q, Λ), Q.T)
    
        corr_matrix = (matrix + matrix.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.clip(eigenvalues, delta, None)
        # eigenvalues = np.where(eigenvalues < delta, delta, eigenvalues)
        corr_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)
        corr_matrix += jitter * np.eye(corr_matrix.shape[0])
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        return corr_matrix

    
    @beartype
    def check_matrix_positive_semifinite(matrix: np.ndarray) -> np.bool_:
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues >= 0)
    
    if check_matrix_positive_semifinite(cov_matrix):
        return cov_matrix
    
    else:
        if calculation_type == "fast":
            new_matrix = eigenvalue_clipping(cov_matrix)
        
        elif calculation_type == "long":
            n = cov_matrix.shape[0]
            theta = np.random.rand(n * (n - 1))
            res = minimize(fun=l2_norm, x0 =theta, args=(cov_matrix,), method="BFGS", options={'maxiter': 100})
            theta_opt = res.x
            theta_opt_reshaped = reshape_theta_to_appropriate_size(theta=theta_opt, n=n)
            new_matrix = np.dot(B(theta_opt_reshaped), B(theta_opt_reshaped).T)
        
        else:
            raise ValueError(f"Unsupported type: {calculation_type}")
        
        # print(check_matrix_positive_semifinite(new_matrix))
        return new_matrix


@beartype
def cholesky_decomposition(cov_matrix: np.ndarray) -> np.ndarray:
    def safe_cholesky(matrix, max_tries=5):
        jitter = 1e-8
        for _ in range(max_tries):
            try:
                return np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                matrix = matrix + jitter * np.eye(matrix.shape[0])
                jitter *= 10
        raise np.linalg.LinAlgError("Matrix not PD even after jitter.")

    cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
    return safe_cholesky(cov_matrix)


@beartype
def calculate_multivariate_shock_matrix(cov_matrix: np.ndarray, n_samples: int) -> np.ndarray:
    L = cholesky_decomposition(cov_matrix)
    n_assets = L.shape[0]
    Z = np.random.normal(loc=0, scale=1, size=(n_samples, n_assets))
    epsilon = np.dot(Z, L.T)
    return epsilon
