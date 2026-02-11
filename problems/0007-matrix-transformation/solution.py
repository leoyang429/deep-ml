import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	arr_a = np.array(A)
	arr_t = np.array(T)
	arr_s = np.array(S)
	try:
		t_inv = np.linalg.inv(arr_t)
		s_inv = np.linalg.inv(arr_s)
	except np.linalg.LinAlgError:
		return -1
	
	return t_inv @ arr_a @ arr_s