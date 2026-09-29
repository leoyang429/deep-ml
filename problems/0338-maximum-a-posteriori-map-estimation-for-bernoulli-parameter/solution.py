import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    a = np.array(observations)
    n = np.sum(a)
    m = len(a) - n
    alpha += n
    beta += m 
    return round((alpha - 1) / (alpha + beta - 2), 4)