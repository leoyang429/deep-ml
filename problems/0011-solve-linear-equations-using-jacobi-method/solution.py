import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	x = np.zeros_like(b)
	d = np.diag(A)
	R = A - np.diagflat(d)
	for _ in range(n):
		x = np.round((b - R @ x) / d, decimals=4)
	return x