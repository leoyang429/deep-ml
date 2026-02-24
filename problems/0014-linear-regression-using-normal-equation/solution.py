import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X_np = np.array(X)
	y_np = np.array(y)
	X_np_T = X_np.T
	theta =  np.linalg.inv(X_np_T @ X_np) @ X_np_T @ y_np
	theta  = np.round(theta, 4)
	return theta