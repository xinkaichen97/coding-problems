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
