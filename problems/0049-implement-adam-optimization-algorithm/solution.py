import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    t, m, v = 0, 0, 0
    while t < num_iterations:
        t += 1
        g = grad(x0)
        m = (beta1 * m + (1 - beta1) * g)
        m_h = m / (1 - beta1 ** t)
        v = (beta2 * v + (1 - beta2) * (g ** 2))
        v_h = v / (1 - beta2 ** t)
        x0 -= learning_rate * m_h / (v_h ** 0.5 + epsilon)
    return x0