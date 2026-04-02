"""
Implementation of Machine Learning problems
"""
import numpy as np
from collections import Counter


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  

def confusion_matrix(data):
    """
    75. Generate a Confusion Matrix for Binary Classification
    https://www.deep-ml.com/problems/75
    """
    # Count all (true, pred) pairs
    counts = Counter(tuple(pair) for pair in data)
    
    # Build confusion matrix
    return [
        [counts[(1, 1)], counts[(1, 0)]],  # TP, FN
        [counts[(0, 1)], counts[(0, 0)]]   # FP, TN
    ]


def confusion_matrix_np(data):
    """
    75. Generate a Confusion Matrix for Binary Classification
    https://www.deep-ml.com/problems/75
    """
    # Convert data to numpy array
    data_array = np.array(data)
    y_true = data_array[:, 0]
    y_pred = data_array[:, 1]
    
    # Initialize confusion matrix
    # cm[i][j] where i is the true label, j is the predicted label
    cm = np.zeros((2, 2), dtype=int)
    
    # Count occurrences for each combination
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label)][int(pred_label)] += 1
    
    # Convert to a list of lists and return
    return cm.tolist()


def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    104. Binary Classification with Logistic Regression
    https://www.deep-ml.com/problems/104
    
    Args:
      X: Input feature matrix (shape: N x D)
      weights: Model weights (shape: D)
      bias: Model bias
    
    Returns:
      Binary predictions (0 or 1)
    """
    z = X @ weights + bias
    z = np.clip(z, -500, 500)
    out = sigmoid(z)
    return np.where(out >= 0.5, 1, 0)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
	"""
	281. Compute the polynomial kernel between two vectors.
    https://www.deep-ml.com/problems/281
	
	Args:
		x: First input vector
		y: Second input vector
		degree: Degree of the polynomial (default: 3)
		gamma: Scaling factor (default: 1.0)
		coef0: Independent term in kernel function (default: 1.0)
	Returns:
		The polynomial kernel value as a float
	"""
	x, y = np.array(x), np.array(y)
	dot_product = np.dot(x, y)
	return (gamma * dot_product + coef0) ** degree
    

def svm_margin_width(w: np.ndarray) -> float:
    """
    282. Calculate the margin width of a linear SVM classifier.
	https://www.deep-ml.com/problems/282
    
    Parameters:
    	w : np.ndarray - weight vector defining the hyperplane
    Returns:
    	float - the total margin width
    """
    return 2 / np.linalg.norm(w)

	
