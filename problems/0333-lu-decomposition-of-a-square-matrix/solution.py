import numpy as np

def lu_decomposition(A: list) -> tuple:
	"""
	Perform LU decomposition on a square matrix using Doolittle's method.
	
	Args:
		A: Square matrix as a list of lists
	
	Returns:
		tuple: (L, U) where L is lower triangular with 1s on diagonal,
		       U is upper triangular, and A = L @ U
	"""
	# Your code here
	A = np.array(A)
	m, n = A.shape
	L = np.eye(m)
	U = np.zeros((m, m))
	for j in range(m):
		for i in range(j+1):
			u, l = A[i][j], A[j][i]
			for k in range(i):
				u -= L[i][k] * U[k][j]
				l -= L[j][k] * U[k][i]
			U[i][j] = u
			if i != j:
				L[j][i] = l / U[i][i]
	return (L, U)
