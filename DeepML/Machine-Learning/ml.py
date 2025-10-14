"""
Implementation of Machine Learning problems
"""
import numpy as np


def rmse(y_true, y_pred):
  """
  71. Calculate Root Mean Square Error (RMSE)
  """
  rmse_res = np.sqrt(np.mean((y_true - y_pred) ** 2))
  return round(rmse_res,3)


def jaccard_index(y_true, y_pred):
  """
  72. Calculate Jaccard Index for Binary Classification
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
  
  Parameters:
  y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values
  
  Returns:
  float: Mean Absolute Error rounded to 3 decimal places
  """
  val = np.mean(np.abs(y_true - y_pred))
  return round(val,3)


class StepLRScheduler:
  """
  153. StepLR Learning Rate Scheduler
  """
  def __init__(self, initial_lr, step_size, gamma):
      # Initialize initial_lr, step_size, and gamma
      self.initial_lr = initial_lr
      self.step_size = step_size
      self.gamma = gamma
  
  def get_lr(self, epoch):
      # Calculate and return the learning rate for the given epoch
      lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
      return round(lr, 4)


class ExponentialLRScheduler:
  """
  154. ExponentialLR Learning Rate Scheduler
  """
  def __init__(self, initial_lr, gamma):
      # Initialize initial_lr and gamma
      self.initial_lr = initial_lr
      self.gamma = gamma
  
  def get_lr(self, epoch):
      # Calculate and return the learning rate for the given epoch
      lr = self.initial_lr * (self.gamma ** epoch)
      return round(lr, 4)
