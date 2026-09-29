import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    A = np.array(matrix)
    symmetric = np.sum(A.T - A) == 0
    eigenvalues = np.sort(np.linalg.eig(A)[0].real.round(4))
    return {
        'is_positive_definite': symmetric and eigenvalues[0] > 1e-10,
        'eigenvalues': eigenvalues.tolist()
    }
