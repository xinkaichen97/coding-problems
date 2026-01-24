"""
Implementation of probability problems
"""
from typing import List, Tuple, Any


def conditional_probability(data: List[Tuple[Any, Any]], x: Any, y: Any) -> float:
    """
    168. Calculate Conditional Probability from Data
    https://www.deep-ml.com/problems/168
    
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    count_x, count_xy = 0, 0

    # manual counter is faster than list comprehension
    for xi, yi in data:
        if xi == x:
            count_x += 1
            if yi == y:
                count_xy += 1

    # # numpy
    # data = np.array(data)
    # x_mask = data[:, 0] == x
    # count_x = np.sum(x_mask)
    # y_mask = data[:, 1] == y
    # count_xy = np.sum(x_mask & y_mask)
  
    # handle special case
    if count_x == 0:
        return 0.0
    
    return round(count_xy / count_x, 4)
  
