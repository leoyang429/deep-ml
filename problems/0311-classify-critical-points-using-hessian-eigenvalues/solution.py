import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    eigvals = np.sort(np.linalg.eig(hessian)[0].real)
    if np.any(np.abs(eigvals) < tol):
        return None
    if eigvals[0] > tol:
        return -1
    if eigvals[-1] < -tol:
        return 1
    return 0