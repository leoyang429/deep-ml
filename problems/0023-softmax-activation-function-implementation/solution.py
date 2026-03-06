import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    s = np.array(scores)
    s = s - np.amax(s)
    exp_s = np.exp(s)
    return (exp_s / np.sum(exp_s)).tolist()