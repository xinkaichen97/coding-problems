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


def label_encode_ordinal(values: list, order: list) -> list:
    """
    356. Encode ordinal categorical values to integers based on specified order.
    https://www.deep-ml.com/problems/356
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    Returns:
        List of integers representing the encoded values
    """
    mp = {cat: i for i, cat in enumerate(order)}
    return [mp.get(val, -1) for val in values]
    
