"""
Implementation of Machine Learning problems
"""
import numpy as np
from math import factorial


def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    310. Compute Taylor series approximation for common functions.
    https://www.deep-ml.com/problems/310
    
    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of terms in the series
    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    if func_name == 'exp':
        res = sum([x ** n / factorial(n) for n in range(n_terms)])
    elif func_name == 'sin':
        res = sum([(-1) ** n * x ** (2 * n + 1) / factorial(2 * n + 1) for n in range(n_terms)])
    elif func_name == 'cos':
        res = sum([(-1) ** n * x ** (2 * n) / factorial(2 * n) for n in range(n_terms)])
    else:
        raise ValueError(f"Unsupported function: {func_name}")
    
    return round(res, 6)
  
