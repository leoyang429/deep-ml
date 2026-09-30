import torch

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    x = torch.tensor(x_value, dtype=torch.float32, requires_grad=True)
    f = x * x + 3 * x + 2
    f.backward()
    return x.grad.item()
