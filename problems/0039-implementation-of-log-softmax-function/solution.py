import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	s = np.array(scores)
	s -= np.max(s)
	s = np.exp(s)
	return np.log(s / np.sum(s))