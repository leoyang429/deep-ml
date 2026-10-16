import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    eps = 1e-9
    x = np.clip(preds, eps, 1.0)
    N, C = x.shape
    y = np.zeros_like(preds, dtype=preds.dtype)
    y[np.arange(N), target] = 1.0
    loss = - np.log(x[np.arange(N), target])
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    grad = (x - y).sum(axis=0)
    return loss, grad