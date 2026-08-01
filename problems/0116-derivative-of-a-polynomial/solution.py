def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    
    # f = c * x^n
    return c * n * (x ** (n-1))
