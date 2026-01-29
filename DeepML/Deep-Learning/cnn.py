"""
Problems for CNNs
"""
import numpy as np
from typing import Union

  
def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/41
    """
    # get shapes
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    
    # apply padding
    if padding > 0:
        padded_input = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)
    else:
        padded_input = input_matrix
    
    padded_height, padded_width = padded_input.shape
    
    # calculate output dimensions
    output_height = (padded_height - kernel_height) // stride + 1
    output_width = (padded_width - kernel_width) // stride + 1
    
    # initialize output
    output_matrix = np.zeros((output_height, output_width))
  
    # perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # calculate input region coordinates
            row_start = i * stride
            row_end = row_start + kernel_height
            col_start = j * stride
            col_end = col_start + kernel_width
            
            # extract region and perform element-wise multiplication and sum
            region = padded_input[row_start:row_end, col_start:col_end]
            output_matrix[i, j] = np.sum(region * kernel)
              
    return output_matrix


def calculate_brightness(img: Union[list, np.ndarray]) -> float:
    """
    https://www.deep-ml.com/problems/70
    """
    # Edge case 1: Empty
    if not img or len(img) == 0:
        return -1
    try:
        img_array = np.array(img)
        # Empty after conversion
        if img_array.size == 0:
            return -1
        # Edge case 2: Jagged (object dtype indicates inconsistent lengths)
        if img_array.dtype == object:
            return -1
        # Edge case 3: Invalid range
        if np.any(img_array < 0) or np.any(img_array > 255):
            return -1
        # Calculate brightness
        return round(float(np.mean(img_array)), 2)
    except:
        return -1


def calculate_contrast(img: np.ndarray) -> int:
    """
    https://www.deep-ml.com/problems/82
    Calculate the contrast of a grayscale image.
    Args:
      img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
    """
    return np.max(img) - np.min(img)
  

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/114
    Args:
      x: array of shape (height, width, channels)
    Returns
      a 1D array of shape (channels,)
    """
    return np.mean(x, axis=(0, 1))


def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/115
    """
    # check input shape
    if X.ndim != 4:
        raise ValueError(f"Expected 4D input (B, C, H, W), got {X.ndim}D")
    batch_size, channels, height, width = X.shape
    # reshape gamma and beta if needed
    if gamma.ndim == 1:
        gamma = gamma.reshape(1, channels, 1, 1)
    if beta.ndim == 1:
        beta = beta.reshape(1, channels, 1, 1)
    # compute mean across batch and spatial dimensions for each channel
    # Shape: (1, C, 1, 1)
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  
    variance = np.var(X, axis=(0, 2, 3), keepdims=True) 
    # normalize
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    # scale and shift
    output = gamma * X_normalized + beta
    return output


class DropoutLayer:
    """
    https://www.deep-ml.com/problems/151
    """
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        if not 0 <= p < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {p}")
        self.p = p
        self.mask = None  # Will store the dropout mask during forward pass
        self.scale = 1.0 / (1.0 - p) if p < 1 else 1.0  # Scaling factor

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.""" 
        # During inference or if p=0, no dropout
        if not training or self.p == 0:
            return x
        
        # Generate random binary mask
        # Each element has probability (1 - p) of being kept
        self.mask = np.random.binomial(1, 1 - self.p, size=x.shape).astype(x.dtype)
        
        # Apply mask and scale
        # Scaling by 1/(1-p) maintains expected value
        output = x * self.mask * self.scale
        
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        # If no mask (inference mode or p=0), gradient passes through unchanged
        if self.mask is None:
            return grad
        
        # Apply the same mask and scaling used in forward pass
        grad_output = grad * self.mask * self.scale
        
        return grad_output
  

def rgb_to_grayscale(image):
    """
    237. Convert an RGB image to grayscale using the luminosity method.
    https://www.deep-ml.com/problems/237
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    img_array = np.array(image)
    # check edge cases
    if img_array.ndim != 3 or img_array.shape[-1] != 3 or np.max(img_array)> 255:
        return -1
    # use matrix multiplication
    weights = np.array([0.299, 0.587, 0.114])
    grayscale = np.dot(img_array, weights)
    return np.round(grayscale).astype(int).tolist()
  
