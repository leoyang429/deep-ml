import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    ret = {}
    data = sorted(data)
    n = len(data)
    ret['mean'] = float(np.sum(data) / n)
    ret['median'] = float(data[n // 2]) if n % 2 == 1 else (data[n // 2] + data[n // 2 - 1]) / 2.0
    ret['mode'] = 0
    ret['variance'] = float(np.var(data))
    ret['standard_deviation'] = ret['variance'] ** 0.5
    max_freq = 0
    prev_x = None
    freq = 0
    for i, x in enumerate(data):
        if prev_x is None:
            ret['mode'] = x
            max_freq = 1
            prev_x = x
            freq = 1
        elif x == prev_x:
            freq += 1
            if freq > max_freq:
                max_freq = freq
                ret['mode'] = x
        else:
            prev_x = x
            freq = 1
        if '25th_percentile' not in ret and i >= n * 0.25 - 1:
            ret['25th_percentile'] = x * 1.0
            ret['50th_percentile'] = ret['median']
        if '75th_percentile' not in ret and i >= n * 0.75 - 1:
            ret['75th_percentile'] = x * 1.0
    ret['interquartile_range'] = ret['75th_percentile'] - ret['25th_percentile']
    return ret
