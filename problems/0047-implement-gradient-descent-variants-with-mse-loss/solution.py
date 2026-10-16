import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m, n = X.shape
    Y = y[:, None] # m, 1
    w = weights[:, None] # n, 1
    if method == 'batch':
        batch_size = m
    elif method == 'stochastic':
        batch_size = 1
    for epoch in range(n_epochs):
        for iter in range(m // batch_size):
            x = X[iter * batch_size:(iter + 1) * batch_size, ...] # bs, n
            y = Y[iter * batch_size:(iter + 1) * batch_size, ...] # bs, 1
            h = x @ w # bs, 1
            L = ((h - y) ** 2).sum(axis=0) / batch_size # 1,
            grad = 2 * x.T @ (h-y) / batch_size # n, 1
            w -= learning_rate * grad
    return w[:, 0]