import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	H, CE = 0, 0
	for p, q in zip(P, Q):
		H -= p * np.log(p) if p != 0 else 0
		CE -= p * np.log(q) if q != 0 else 0
	return (H, CE)
