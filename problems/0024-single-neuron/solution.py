import math
import numpy as np

def sigmoid(z):
    # Your code here
    return 1 / (1 + np.exp(-z))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
    A = np.array(features)
    x = np.array(weights).reshape(-1, 1)
    p = sigmoid(A @ x + bias)
    l = np.array(labels).reshape(-1, 1)
    mse = np.sum((p - l) ** 2) / l.shape[0]
    return np.round(p.flatten(), 4).tolist(), np.round(mse, 4).tolist()