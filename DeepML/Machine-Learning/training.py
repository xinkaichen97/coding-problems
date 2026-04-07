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


def early_stopping_2(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    """
    199. Determine at each epoch whether training should stop based on validation loss
    https://www.deep-ml.com/problems/199
    
    Args:
        val_losses: List of validation losses at each epoch
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    Returns:
        List of booleans indicating whether to stop at each epoch
    """
    # edge case
    if len(val_losses) == 0:
        return []
        
    # initialize 
    res = []
    count = 0
    best_loss = float("inf")
    
    for loss in val_losses:
        # Check if current loss is an improvement
        if loss < best_loss - min_delta:
            best_loss = loss
            count = 0
        else:
            count += 1
        
        # Stop if we've had patience epochs without improvement
        res.append(count >= patience)
        
    return res


def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    290. Implement He (Kaiming) weight initialization.
    https://www.deep-ml.com/problems/290
    
    Parameters:
        n_in: number of input units
        n_out: number of output units
        mode: 'fan_in' or 'fan_out'
        distribution: 'normal' or 'uniform'
        seed: random seed for reproducibility  
    Returns:
        numpy array of shape (n_in, n_out) with He-initialized weights
    """
    np.random.seed(seed)
    fan = n_in if mode == "fan_in" else n_out
    if distribution == "normal":
        weights = np.random.normal(0, np.sqrt(2 / fan), (n_in, n_out))
    else:
        weights = np.random.uniform(-np.sqrt(6 / fan), np.sqrt(6 / fan), (n_in, n_out))
    return weights
    

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	308. Calculate the magnitude and direction of a gradient vector.
    https://www.deep-ml.com/problems/308
	
	Args:
		gradient: A list representing the gradient vector
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	gradient = np.array(gradient)
	magnitude = np.linalg.norm(gradient)
	if magnitude == 0:
		direction = descent_direction = np.zeros(gradient.shape)
	else:
		direction = gradient / magnitude
		descent_direction = - direction
	return {
		'magnitude': magnitude, 'direction': direction.tolist(), 'descent_direction': descent_direction.tolist()
	}
    

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    369. Perform Xavier/Glorot weight initialization.
	https://www.deep-ml.com/problems/369

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.
    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    np.random.seed(seed)
    shape = [fan_in, fan_out]
    if mode == 'uniform':
        param = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-param, param, shape)
    elif mode == 'normal':
        param = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, param, shape)
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")
		
    return {
        'weights': np.round(weights, 4).tolist(), 'shape': shape, 'param': np.round(param, 4)
    }
	
