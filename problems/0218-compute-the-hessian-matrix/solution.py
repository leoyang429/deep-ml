from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	n = len(point)
	x = point
	H = [[0] * n for _ in range(n)]
	for i in range(n):
		for j in range(n):
			if i == j:
				f0 = f(x)
				x[i] += h
				f1 = f(x)
				x[i] -= 2 * h
				f2 = f(x)
				x[i] += h
				H[i][j] = (f1 + f2 - 2 * f0) / (h ** 2)
			else:
				x[i] += h 
				x[j] += h 
				f1 = f(x)
				x[j] -= 2 * h 
				f2 = f(x)
				x[i] -= 2 * h 
				f4 = f(x)
				x[j] += 2 * h 
				f3 = f(x)
				x[i] += h 
				x[j] -= h 
				H[i][j] = (f1 - f2 - f3 + f4) / (4 * h ** 2)
	return H