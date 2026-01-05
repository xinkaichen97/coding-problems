"""
Implementation of Calculus problems
"""


def poly_term_derivative(c: float, x: float, n: float) -> float:
    """
    116. Derivative of a Polynomial
    https://www.deep-ml.com/problems/116
    """
    if n == 0.0:
        return 0.0
    else:
        return n * c * (x ** (n - 1))
      
