import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	arr = np.array(matrix)
	return list(arr.mean(axis=1) if mode == 'row' else arr.mean(axis=0))