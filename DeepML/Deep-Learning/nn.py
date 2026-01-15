"""
Problems for Deep Learning
"""
import math
import numpy as np
from numpy.typing import NDArray
from scipy.special import erf


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


def hard_sigmoid(x: float) -> float:
	"""
    https://www.deep-ml.com/problems/96
	Implements the Hard Sigmoid activation function.
	Args:
		x (float): Input value
	Returns:
		float: The Hard Sigmoid of the input
	"""
	if x <= -2.5:
        return 0.0
    elif x < 2.5:
        return 0.2 * x + 0.5
    else:
        return 1.0
        

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
    # np: np.maximum(0, x)
  	return max(0, z)


def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    """
    https://www.deep-ml.com/problems/44
    """
  	return max(0, z * alpha) 


def elu(x: float, alpha: float = 1.0) -> float:
	"""
    https://www.deep-ml.com/problems/97
	Compute the ELU activation function.
	Args:
		x (float): Input value
		alpha (float): ELU parameter for negative values (default: 1.0)
	Returns:
		float: ELU activation value
	"""
    val = x if x > 0 else alpha * (math.exp(x) - 1)
	return round(float(val), 4)


def prelu(x: float, alpha: float = 0.25) -> float:
	"""
    https://www.deep-ml.com/problems/98
	Implements the PReLU (Parametric ReLU) activation function.
	Args:
		x: Input value
		alpha: Slope parameter for negative values (default: 0.25)
	Returns:
		float: PReLU activation value
	"""
	val = x if x > 0 else alpha * x
    return round(float(val), 4)


def selu(x: float) -> float:
	"""
    https://www.deep-ml.com/problems/103
	Implements the SELU (Scaled Exponential Linear Unit) activation function.
	Args:
		x: Input value
	Returns:
		SELU activation value
	"""
	alpha = 1.6732632423543772
	scale = 1.0507009873554804
	val = scale * x if x > 0 else scale * alpha * (math.exp(x) - 1)
    return round(float(val), 4)


def GeLU(x: np.ndarray) -> np.ndarray:
	"""
    https://www.deep-ml.com/problems/147
    GELU (Gaussian Error Linear Unit) is the default activation function in Transformer-based models
    """
    scores = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    ## tanh approximation (used in BERT, GPT):
    # scores = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    ## sigmoid approximation:
    # scores = x * (1 / (1 + np.exp(-1.702 * x)))
	return scores


def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/156
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    d = x.shape[1] // 2
    x1 = x[:, :d]
    x2 = x[:, d:]
    scores = x1 * x2 * sigmoid(x2)
    return np.round(scores, 4)


def tanh(x: np.ndarray) -> np.ndarray:
    # return np.tanh(x)
    # tanh(x) = 2 * sigmoid(2x) - 1
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    """
    https://www.deep-ml.com/problems/128
    """
    return np.round(gamma * tanh(alpha * x) + beta, 4)


def softplus(x: float) -> float:
	"""
    https://www.deep-ml.com/problems/99
	Compute the softplus activation function.
	Args:
		x: Input value
	Returns:
		The softplus value: log(1 + e^x)
	"""
    # For x > 20, exp(x) is huge, so ln(1 + exp(x)) ≈ x
    # np: return np.where(x > 20, x, np.log(1 + np.exp(x)))
	val = math.log(1 + math.exp(x))
	return round(val,4)


def softsign(x: float) -> float:
	"""
    https://www.deep-ml.com/problems/100
	Implements the Softsign activation function.
	Args:
		x (float): Input value
	Returns:
		float: The Softsign of the input
    """
	val = x / (1 + abs(x))
	return round(val, 4)


def swish(x: float) -> float:
	"""
    https://www.deep-ml.com/problems/102
	Implements the Swish activation function.
	Args:
		x: Input value
	Returns:
		The Swish activation value
	"""
    # np: x * sigmoid(beta * x)
    # beta: Scaling parameter (default=1.0)
	return x / (1 + math.exp(-x))
    

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


class Value:
    """
    https://www.deep-ml.com/problems/26
    reference: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

	def relu(self):
		out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

	def backward(self):
		# topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


def compute_efficiency(n_experts, k_active, d_in, d_out):
	"""
	123. Calculate Computational Efficiency of MoE
	https://www.deep-ml.com/problems/123

	Args:
		n_experts: Total number of experts
		k_active: Number of active experts (sparsity)
		d_in: Input dimension
		d_out: Output dimension

	Returns:
		Percentage savings in FLOPs
	"""
	flop_dense = n_experts * d_in * d_out
	flop_moe = k_active * d_in * d_out
	savings = (flop_dense - flop_moe) / flop_dense * 100
	return savings		
