import numpy as np
from collections import defaultdict

def poly_dict_to_list(f_dict: defaultdict) -> list:
    ret = []
    for k, v in sorted(f_dict.items()):
        while k > len(ret):
            ret.append(0)
        ret.append(v)
    while len(ret) > 1:
        if ret[-1] == 0:
            ret.pop(-1)
        else:
            break
    return ret

def poly_list_to_dict(f: list) -> defaultdict:
    ret = defaultdict(lambda: 0)
    for i, x in enumerate(f):
        ret[i] += x
    return ret

def poly_mult(f: list, g: list) -> list:
    f_dict, g_dict = poly_list_to_dict(f), poly_list_to_dict(g)
    ret_dict = defaultdict(lambda: 0)
    for k_f, v_f in sorted(f_dict.items()):
        for k_g, v_g in sorted(g_dict.items()):
            ret_dict[k_f + k_g] += v_f * v_g
    return poly_dict_to_list(ret_dict)

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    # Your code here
    
    product = poly_mult(f_coeffs, g_coeffs)[1:]
    if len(product) == 0:
        return [0.0]
    coeffs = [x+1 for x in range(len(product))]
    return [round(product[i] * coeffs[i], 4) for i in range(len(product))]