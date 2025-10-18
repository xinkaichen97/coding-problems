"""
Problems for Deep Learning
"""
import math
import numpy as np
from numpy.typing import NDArray


def sigmoid(z: float) -> float:
    """
    https://www.deep-ml.com/problems/22
    """
    result = 1 / (1 + math.exp(-z))
    # numpy: 1 / (1 + np.exp(-z))
    return round(result, 4)


def sigmoid_stable(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Numerically stable sigmoid implementation.
    Args:
        x: Input array of any shape, dtype should be float
    Returns:
        Sigmoid output with same shape as input, values in range (0, 1)
    """
    # For positive values, use the standard formula
    # For negative values, use exp(x)/(1 + exp(x)) to avoid overflow
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def softmax(scores: list[float]) -> list[float]:
    """
    https://www.deep-ml.com/problems/23
    """
  	sums = sum([math.exp(score) for score in scores])
  	probabilities = [round(math.exp(score) / sums, 4) for score in scores]
  	return probabilities


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax values using numpy.
    Args:
        x: Input array of shape (n_samples, n_features) or (n_features,)
    Returns:
        Softmax probabilities with same shape as input
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    
    # Divide by sum of exponentials
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def log_softmax(scores: list) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/39
    """
    x = np.asarray(scores)
  	x_max = np.max(x, axis=-1, keepdims=True)
    x_shifted = x - x_max
    
    # Compute log(sum(exp(x_shifted)))
    log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
    
    # log_softmax = x_shifted - log_sum_exp
    # (adding back x_max cancels out: x - x_max - log_sum_exp + x_max = x - log_sum_exp)
    return x_shifted - log_sum_exp


def relu(z: float) -> float
    """
    https://www.deep-ml.com/problems/42
    """
  	return max(0, z)


def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    """
    https://www.deep-ml.com/problems/44
    """
  	return max(0, z * alpha) 


def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    """
    https://www.deep-ml.com/problems/24
    """
    z = [sum([f * weight for f, weight in zip(feature, weights)]) + bias for feature in features]
    probabilities = [round(1 / (1 + math.exp(-z_i)), 4) for z_i in z]
    mse = sum([(pred - label) ** 2 for pred, label in zip(probabilities, labels)]) / len(features)
    mse = round(mse, 4)
    return probabilities, mse


def single_neuron_model_np(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> tuple[np.ndarray, float]:
  
    z = np.dot(features, weights) + bias  # Shape: (n_samples,)
    
    # Apply sigmoid activation
    predictions = 1 / (1 + np.exp(-z))
    
    # Compute mean squared error
    mse = np.mean((predictions - labels) ** 2)
    
    # Round to 4 decimal places
    predictions_rounded = np.round(predictions, 4)
    mse_rounded = round(float(mse), 4)
    
    return predictions_rounded, mse_rounded


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    """
    https://www.deep-ml.com/problems/25
    """
    # define variables
    weights = initial_weights.copy()
    bias = initial_bias
    n_samples = features.shape[0]
    mse_values = []

    # loop through each epoch
    for i in range(epochs):
        # forward pass
        z = np.dot(features, weights) + bias  
        predictions = 1 / (1 + np.exp(-z))
      
        # calculate mse and add to list
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))
      
        # calculate gradients
        sigmoid_grad = predictions * (1 - predictions) 
        delta = (predictions - labels) * sigmoid_grad
        weight_gradients = (2 / n_samples) * np.dot(features.T, delta)
        bias_gradient = (2 / n_samples) * np.sum(delta) 
      
        # gradient descent
        weights = weights - learning_rate * weight_gradients
        bias = bias - learning_rate * bias_gradient
    
    # rounding
    updated_weights = np.round(weights, 4).tolist()
    updated_bias = round(float(bias), 4)  
    return updated_weights, updated_bias, mse_values
