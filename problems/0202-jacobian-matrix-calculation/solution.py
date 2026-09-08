import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	f0 = f(x)
	jacob = np.zeros([len(f0), len(x)])
	for i in range(len(x)):
		x[i] += h
		f1 = np.array(f(x))
		x[i] -= 2 * h
		f2 = np.array(f(x))
		x[i] += h
		d = (f1 - f2) / (2 * h)
		jacob[:, i] = d
	return jacob
