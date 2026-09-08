import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	max_v = 0
	for v in x:
		max_v = max(max_v, v)
	n = len(x)
	for i in range(n):
		x[i] -= max_v
	a = np.array(x)
	d = np.sum(np.exp(a))
	s = []
	for v in x:
		s.append(np.exp(v) / d)
	jacob = np.zeros([n, n])
	for i in range(n):
		for j in range(n):
			if i == j:
				jacob[i][j] = s[i] * (1 - s[i])
			else:
				jacob[i][j] = - s[i] * s[j]
	return jacob
