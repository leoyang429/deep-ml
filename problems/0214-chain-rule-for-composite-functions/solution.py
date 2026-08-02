import numpy as np

registry = {}

def register(func):
    registry[func.__name__] = func
    return func

@register
def square(x):
    return x ** 2

@register
def square_d(x):
    return 2 * x

@register
def sin(x):
    return np.sin(x)

@register
def sin_d(x):
    return np.cos(x)

@register
def exp(x):
    return np.exp(x)

@register
def exp_d(x):
    return np.exp(x)

@register
def log(x):
    return np.log(x)

@register
def log_d(x):
    return 1 / x

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    grad = 1.0
    for func in functions[::-1]:
        grad *= registry[f'{func}_d'](x)
        x = registry[f'{func}'](x)
    return grad