import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	try:
		arr_a = np.array(a)
		ret = list(arr_a.reshape(new_shape))
	except Exception as e:
		ret = []
	return ret