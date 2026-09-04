import numpy as np

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    if len(samples) == 0:
        return []
    samples = sorted(samples)
    p = 1 / len(samples)
    ret = []
    for x in samples:
        if len(ret) == 0 or x != ret[-1][0]:
            ret.append((x, p))
        else:
            x, prob = ret.pop(-1)
            ret.append((x, prob + p))
    return ret
