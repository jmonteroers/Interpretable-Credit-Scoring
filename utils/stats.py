import numpy as np


def ecdf(data):
    """Compute ECDF for a one-dimensional quantitative variable"""
    n = len(data)
    
    # sorted data for the x-axis, ECDF
    x = np.sort(data)
    
    # y values, from 1/n to 1
    y = np.arange(1, n + 1) / n
    
    return x, y