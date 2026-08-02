import numpy as np

registry = {}

def register(func):
    registry[func.__name__] = func
    return func

@register
def poly2d(point):
    x, y = point
    return (2 * x * y + y ** 2,
            x ** 2 + 2 * x * y)

@register
def exp_sum(point):
    x, y = point
    return (np.exp(x + y),
            np.exp(x + y))

@register
def product_sin(point):
    x, y = point
    return (np.sin(y),
            np.cos(y))

@register
def poly3d(point):
    x, y, z = point
    return (2 * x * y,
            x ** 2 + z ** 2,
            2 * z * y)

@register
def squared_error(point):
    x, y = point
    return (2 * (x - y),
            -2 * (x - y))

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	# Your code here
	return registry[func_name](point)