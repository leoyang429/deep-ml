import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    g = grad_output
    return (
        g @ W,
        g.T @ x,
        g.sum(0)
    )
