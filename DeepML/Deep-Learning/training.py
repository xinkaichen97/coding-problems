"""
Problems for Deep Learning
"""
import numpy as np
from np.typing import NDArray


def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    """
    https://www.deep-ml.com/problems/49
    https://www.deep-ml.com/problems/87
    """
    # initialize parameters
    x = np.asarray(x0, dtype=np.float64)
    m = np.zeros_like(x) 
    v = np.zeros_like(x)
    
    for t in range(1, num_iterations + 1):
        # compute gradient
        g = np.asarray(grad(x), dtype=np.float64)
      
        # update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
      
        # compute bias-corrected first and second moments
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
      
        # update parameters
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
      
    return float(x) if np.isscalar(x0) else x


def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    https://www.deep-ml.com/problems/169
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w = w - lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * w)
    return w, m, v


def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    https://www.deep-ml.com/problems/134
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    # compute look-ahead param
    para_ahead = parameter - momentum * velocity
    # compute update velocity 
    velocity = momentum * velocity + learning_rate * grad_fn(para_ahead)
    # update param
    parameter = parameter - velocity
    
    return np.round(parameter, 5), np.round(velocity, 5)


def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    """
    https://www.deep-ml.com/problems/145
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        G: Accumulated squared gradients
        learning_rate: Learning rate (default=0.01)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_G)
    """
    # convert to array
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    # Step 1: Accumulate squared gradients
    G += grad ** 2
    # Step 2: Compute adaptive learning rate
    adaptive_lr = learning_rate / (np.sqrt(G) + epsilon)
    # Step 3: Update parameters
    parameter -= adaptive_lr * grad
  
    return np.round(parameter, 5), np.round(G, 5)


def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    https://www.deep-ml.com/problems/146
    Update parameters using the momentum optimizer.
    Uses momentum to accelerate learning in relevant directions and dampen oscillations.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        velocity: Current velocity/momentum term
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    # convert to array
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
  
    # update velocity and parameter
    velocity = momentum * velocity + learning_rate * grad
    parameter -= velocity
  
    return np.round(parameter, 5), np.round(velocity, 5)


def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    https://www.deep-ml.com/problems/148
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    It uses the maximum of past squared gradients instead of the exponential moving average.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        u: Infinity norm estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.002)
        beta1: First moment decay rate (default=0.9)
        beta2: Infinity norm decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_m, updated_u)
    """
    # convert to array
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
  
    # first moment estimate
    m = beta1 * m + (1 - beta1) * grad
    # second moment estimate (infinity norm)
    u = np.maximum(beta2 * u, np.abs(grad))
    # bias correction
    m_t = m / (1 - beta1 ** t)
    # parameter update
    parameter -= learning_rate * m_t / (u + epsilon)
  
    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)


def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    https://www.deep-ml.com/problems/149
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)
    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    # convert to array
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Step 1: Accumulate gradient (exponential moving average of squared gradients)
    u = rho * u + (1 - rho) * grad ** 2
    # Step 2: Compute parameter update (delta)
    delta = - np.sqrt(v + epsilon) / np.sqrt(u + epsilon) * grad
    # Step 3: Update parameters
    parameter += delta
    # Step 4: Accumulate parameter updates (exponential moving average of squared deltas)
    v = rho * v + (1 - rho) * delta ** 2
    
    return np.round(parameter, 5), np.round(u, 5), np.round(v, 5)


def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    """
    https://www.deep-ml.com/problems/134
    """
    # clip to avoid log(0)
    predicted_probs = np.clip(predicted_probs, epsilon, 1.0 - epsilon)
    mean_loss = -np.mean(np.sum(true_labels * np.log(predicted_probs), axis=-1))
    # make sure the return value is non-negative (e.g. -0.0)
    return float(np.maximum(mean_loss, 0.0))


def cross_entropy(predictions: NDArray[np.floating], targets: NDArray[np.floating]) -> float:
    """
    Cross-entropy loss.
    Measures the difference between predicted and true probability distributions.
    Args:
        predictions: Predicted probabilities, shape (n_samples, n_classes)
        targets: True labels (one-hot encoded), shape (n_samples, n_classes)
    Returns:
        Average cross-entropy loss
    """
    # Clip to avoid log(0)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(targets * np.log(predictions), axis=-1))


def softmax_cross_entropy_gradient(logits: NDArray[np.floating], targets: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Gradient of softmax + cross-entropy.
    Key insight: The gradient simplifies to (predictions - targets)!
    Args:
        logits: Input logits before softmax
        targets: True labels (one-hot encoded)
    Returns:
        Gradient with respect to logits
    """
    predictions = softmax(logits)
    return predictions - targets
