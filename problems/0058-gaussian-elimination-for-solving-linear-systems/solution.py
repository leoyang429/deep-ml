import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	aug = np.hstack([A, b.reshape(-1, 1)])
	m, n = aug.shape
	p = 0
	tol = 1e-10
	for i in range(m):
		if p > n - 2:
			break
		v = 0
		r = i
		for j in range(i, m):
			if abs(aug[j, p]) > v:
				v = abs(aug[j, p])
				r = j
		if v < tol:
			p += 1
			continue
		aug[[i, r]] = aug[[r, i]]
		for j in range(i+1, m):
			aug[j] -= (aug[j, p] / aug[i, p]) * aug[i]
		p += 1
	sol = np.zeros(A.shape[1])
	for i in range(m-1, -1, -1):
		b = aug[i, -1]
		for j in range(m-1, i, -1):
			b -= sol[j] * aug[i, j]
		sol[i] = b / aug[i, i]
	return sol
		

