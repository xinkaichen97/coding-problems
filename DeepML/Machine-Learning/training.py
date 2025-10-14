"""
Implementation of Machine Learning problems
"""
import numpy as np


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
