import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	arr_a = np.array(a)
	arr_b = np.array(b)
	n, m = arr_a.shape
	if arr_b.shape[0] != m:
		return -1
	return list(np.dot(arr_a, arr_b.T))
	