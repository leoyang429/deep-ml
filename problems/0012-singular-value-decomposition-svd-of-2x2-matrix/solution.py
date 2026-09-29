import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    B = A.T @ A
    theta = np.pi / 4 if B[0][0] == B[1][1] else 0.5 * np.arctan(2 * B[0][1] / (B[0][0] - B[1][1]))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    D = R.T @ B @ R
    l1, l2 = D[0][0], D[1][1]
    s1, s2 = l1 ** 0.5, l2 ** 0.5
    U = A @ R @ np.array([[1/s1, 0], [0, 1/s2]])
    if s1 < s2:
        s1, s2 = s2, s1
    return (U, np.array([s1, s2]), R.T)
