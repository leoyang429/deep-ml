import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	# Your code here
	A = np.array(A).astype(float)
	m, n = A.shape
	Q = np.zeros_like(A).astype(float)
	R = np.zeros((n, n)).astype(float)
	for j in range(n):
		v = A[:, j:j+1]
		for i in range(j):
			R[i][j] = np.dot(Q[:, i], A[:, j])
			v -= R[i][j] * Q[:, i:i+1]
		R[j][j] = (np.sum(v ** 2) ** 0.5)
		Q[:, j:j+1] = v / R[j][j]
	return (Q, R)
