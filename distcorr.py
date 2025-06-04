import numpy as np

def distance_covariance(X, Y):
    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError("Samples must have the same length")
    
    a = np.abs(X[:, None] - X[None, :])
    b = np.abs(Y[:, None] - Y[None, :])
    
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    
    dcov = np.sqrt(np.sum(A * B) / (n * n))
    return dcov

def distance_correlation(X, Y):
    dcov_xy = distance_covariance(X, Y)
    dcov_xx = distance_covariance(X, X)
    dcov_yy = distance_covariance(Y, Y)
    
    if dcov_xx * dcov_yy == 0:
        return 0.0
    else:
        return dcov_xy / np.sqrt(dcov_xx * dcov_yy)

def distance_correlation_matrix(returns):
    """
    Compute distance correlation matrix between assets (rows).
    
    Parameters:
    returns: np.ndarray of shape (N, T) (assets x time)
    
    Returns:
    dist_corr_mat: np.ndarray of shape (N, N)
    """
    n_assets = returns.shape[0]
    dist_corr_mat = np.eye(n_assets)
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            dcorr = distance_correlation(returns[i, :], returns[j, :])
            dist_corr_mat[i, j] = dcorr
            dist_corr_mat[j, i] = dcorr
            
    return dist_corr_mat
