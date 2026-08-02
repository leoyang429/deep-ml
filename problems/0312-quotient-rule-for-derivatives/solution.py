import numpy as np

def evaluate(f: list, x: float) -> float:
    ret = 0
    for i, p in enumerate(f):
        ret += p * (x ** (len(f) - 1 - i))
    return ret

def derivative(f: list) -> list:
    if len(f) == 0:
        return [0]
    return [(len(f) - 1 - i) * f[i] for i in range(len(f)-1)]

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    return (evaluate(derivative(g_coeffs), x) * evaluate(h_coeffs, x) - 
            evaluate(derivative(h_coeffs), x) * evaluate(g_coeffs, x)) / \
           ((evaluate(h_coeffs, x)) ** 2)