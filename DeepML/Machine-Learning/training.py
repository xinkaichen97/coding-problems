"""
Implementation of Machine Learning problems
"""
import numpy as np
from typing import Callable, List, Tuple


def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    """
    135. Implement Early Stopping Based on Validation Loss
    https://www.deep-ml.com/problems/135
    """
    best_epoch, count = 0, 0
    prev = val_losses[0]
    n = len(val_losses)
  
    for i in range(1, n):
        # if the loss doesn't increase more than min_delta
        # update count and return if already reaches patience
        if prev - val_losses[i] < min_delta:
            count += 1
            if count == patience:
                return (i, best_epoch)
        else:
            # if the loss decreases, reset the count and update the best epoch
            # also update the prev
            count = 0
            best_epoch = i
            prev = val_losses[i]
          
    return (n - 1, n - 1)


class StepLRScheduler:
  """
  153. StepLR Learning Rate Scheduler
  https://www.deep-ml.com/problems/153
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
  https://www.deep-ml.com/problems/154
  """
  def __init__(self, initial_lr, gamma):
      # Initialize initial_lr and gamma
      self.initial_lr = initial_lr
      self.gamma = gamma
  
  def get_lr(self, epoch):
      # Calculate and return the learning rate for the given epoch
      lr = self.initial_lr * (self.gamma ** epoch)
      return round(lr, 4)


def checkpoint_forward(funcs: List[Callable[[np.ndarray], np.ndarray], input_arr: np.ndarray) -> np.ndarray:
    """
    188. Gradient Checkpointing
    https://www.deep-ml.com/problems/188
    
    Applies a list of functions in sequence to the input array, simulating gradient checkpointing by not storing intermediates.
    Args:
        funcs (list of callables): List of functions to apply in sequence.
        input_arr (np.ndarray): Input numpy array.
    Returns:
        np.ndarray: The output after applying all functions, same shape as output of last function.
    """
    res = input_arr
    for func in funcs:
        res = func(res)
    return res


def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
    """
    198. Apply weight decay (L2 regularization) to parameters
    https://www.deep-ml.com/problems/198
    
    Args:
        parameters: List of parameter arrays
        gradients: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    Returns:
        Updated parameters
    """
    # convert to np arrays
    param = np.array(parameters)
    grad = np.array(gradients)
    mask = np.array(apply_to_all).reshape(-1, 1)
                           
    # update parameters with gradients (including bias)
    param -= lr * grad
    # only apply weight decay on the mask (excluding bias)
    param -= lr * weight_decay * param * mask
                           
    return param.tolist()


