import numpy as np
import numpy.linalg as LA

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    cov = np.cov(standardized_data, rowvar=False)
    _, eigenvectors = LA.eigh(cov)
    ret = eigenvectors[:, ::-1][:, :k]
    for i in range(ret.shape[1]):
        for j in range(ret.shape[0]):
            if np.abs(ret[j, i]) > 1e-10:
                if ret[j, i] < 0:
                    ret[:, i] *= -1
                break
    return np.round(ret, decimals=4)