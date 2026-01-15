"""
Implementation of Machine Learning problems
"""
import numpy as np
from typing import Tuple


def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    141. Shift and scale values from their original range [min, max] to a target [c, d] range.
    https://www.deep-ml.com/problems/141
    """
    x = np.asarray(values, dtype=float)
    a, b = np.min(x), np.max(x)
    if a == b:
        return np.full_like(x, c, dtype=float)
    x = c + (d - c) / (b - a) * (x - a)
    return x
  
