import math

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	def _sigmoid(x):
		return 1 / (1 + math.exp(-x))
	return {
		'sigmoid': _sigmoid(x) * (1 - _sigmoid(x)),
		'tanh': 1 - math.tanh(x) ** 2,
		'relu': 1 if x > 0 else 0
	}