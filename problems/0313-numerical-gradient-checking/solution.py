import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x[i] += epsilon
        f1 = f(x)
        x[i] -= epsilon * 2
        f2 = f(x)
        x[i] += epsilon
        grad[i] = (f1 - f2) / (2 * epsilon)
    norm = np.linalg.norm(grad) + np.linalg.norm(analytical_grad)
    return (
        grad,
        np.linalg.norm(grad - analytical_grad) / norm if np.abs(norm) > epsilon else 0
    )

