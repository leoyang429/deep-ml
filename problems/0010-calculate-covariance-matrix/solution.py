import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	V = np.array(vectors, dtype=np.float32)
	return np.cov(V).tolist()