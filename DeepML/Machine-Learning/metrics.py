"""
Implementation of Machine Learning problems
"""
import numpy as np


def f_score(y_true, y_pred, beta):
	"""
	61. Implement F-Score Calculation for Binary Classification
	https://www.deep-ml.com/problems/61
	
	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	y_true = np.asarray(y_true).flatten()
	y_pred = np.asarray(y_pred).flatten()
	
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))
	
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
	
	return round(f_score, 3)
  

def calculate_f1_score(y_true, y_pred):
	"""
	91. Calculate F1 Score from Predicted and True Labels
	https://www.deep-ml.com/problems/91
	
	Args:
	y_true (list): True labels (ground truth).
	y_pred (list): Predicted labels.
	
	Returns:
	float: The F1 score rounded to three decimal places.
	"""
	y_true = np.asarray(y_true).flatten()
	y_pred = np.asarray(y_pred).flatten()
	
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))
	
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
	
	return round(f1, 3)


def rmse(y_true, y_pred):
	"""
	71. Calculate Root Mean Square Error (RMSE)
	https://www.deep-ml.com/problems/71
	"""
	rmse_res = np.sqrt(np.mean((y_true - y_pred) ** 2))
	return round(rmse_res, 3)


def jaccard_index(y_true, y_pred):
	"""
	72. Calculate Jaccard Index for Binary Classification
	https://www.deep-ml.com/problems/72
	"""
	y_true = np.asarray(y_true).flatten()
	y_pred = np.asarray(y_pred).flatten()
	
	num = np.sum((y_true == y_pred) & (y_pred == 1))
	if num == 0:
	  return 0.0
	
	if np.sum(y_true == 1) == 0 and np.sum(y_pred == 1) == 0:
	  return 0.0
	
	denom = np.sum(y_true + y_pred >= 1)
	res = num / denom
	return round(res, 3)
  

def dice_score(y_true, y_pred):
	"""
	73. Calculate Dice Score for Classification
	https://www.deep-ml.com/problems/73
	"""
	y_true = np.asarray(y_true).flatten()
	y_pred = np.asarray(y_pred).flatten()
	
	if len(y_true) != len(y_pred):
	  raise ValueError("Arrays must have the same length")
	  
	if y_true.sum() + y_pred.sum() == 0:
	  return 0.0
	
	intersection = np.logical_and(y_true, y_pred).sum()
	res = 2 * intersection / (y_true.sum() + y_pred.sum())
	return round(res, 3)


def mae(y_true, y_pred):
	"""
	93. Calculate Mean Absolute Error (MAE)
	https://www.deep-ml.com/problems/93
	
	Parameters:
	y_true (numpy.ndarray): Array of true values
	y_pred (numpy.ndarray): Array of predicted values
	
	Returns:
	float: Mean Absolute Error rounded to 3 decimal places
	"""
	val = np.mean(np.abs(y_true - y_pred))
	return round(val, 3)


def huber_loss(y_true, y_pred, delta=1.0):
	"""
    192. Compute the Huber Loss between true and predicted values.
	https://www.deep-ml.com/problems/192
    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior
    Returns:
        float: Average Huber loss
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
	# when diff <= delta, it's the MSE; otherwise, it's δ * (|r| - δ/2)
    huber = np.where(
        np.abs(diff) <= delta,
        0.5 * diff ** 2,
        delta * (np.abs(diff) - 0.5 * delta)
    )
    return np.mean(huber)
	

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """
    255. Compute Focal Loss for multi-class classification.
    https://www.deep-ml.com/problems/255
    Args:
        y_true: Ground truth labels as class indices (list or 1D array)
        y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, list or 1D array of length n_classes)
    Returns:
        float: Average focal loss
    """
    n_samples = len(y_true)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
	# clip y_pred to avoid log issues
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
	# get true class probabilities from y_true
    p_t = y_pred[np.arange(n_samples), y_true]
	
	# apply alpha weights (n_classes,)
    if alpha:
        alpha = np.array(alpha)
		 # get weights for each sample
        alpha_t = alpha[y_true]
    else:
        alpha_t = np.ones(n_samples)
		
    fl = - alpha_t * (1 - p_t) ** gamma * np.log(p_t)
    return np.mean(fl)
	
