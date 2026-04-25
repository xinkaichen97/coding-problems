"""
Implementation of Calculus problems
"""
import numpy as np


def poly_term_derivative(c: float, x: float, n: float) -> float:
    """
    116. Derivative of a Polynomial
    https://www.deep-ml.com/problems/116
    """
    if n == 0.0:
        return 0.0
    else:
        return n * c * (x ** (n - 1))
      

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	220. Compute the derivative of cross-entropy loss with respect to logits.
    https://www.deep-ml.com/problems/220
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	logits = np.array(logits, dtype=float)
	z = logits - logits.max()  # stable softmax
	exp_z = np.exp(z)
	p = exp_z / exp_z.sum()

	grad = p.copy()
	grad[target] -= 1  # no need to construct one-hot

	return grad.tolist()
    
