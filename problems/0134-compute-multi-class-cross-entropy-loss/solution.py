import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    L = 0
    for prob, label in zip(predicted_probs, true_labels):
        for p, l in zip(prob, label):
            L -= l * np.log(max(p, epsilon))
    return L / len(predicted_probs)