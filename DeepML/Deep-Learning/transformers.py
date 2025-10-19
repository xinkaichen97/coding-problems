"""
Problems for Transformers
"""
import numpy as np


def softmax(values):
    weights = np.exp(values - np.max(values, axis=-1, keepdims=True)) 
    return weights / np.sum(weights, axis=-1, keepdims=True)


def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/53
    https://www.deep-ml.com/problems/107
    """
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores + (mask * -1e9)

    # softmax
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)

    # weighted sum
    attention_output = attention_weights @ V
    
    return attention_output


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return X @ W_q, X @ W_k, X @ W_v


def pattern_weaver(n: int, crystal_values: list[float], dimension: int) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/89
    """
    # reshape to be a column vector
    crystal_values = np.asarray(crystal_values).reshape(n, 1)
    # create n x n matrix
    attention_weights = crystal_values @ crystal_values.T / np.sqrt(dimension)
    attention_weights = softmax(attention_weights)
    # weighted sum of values
    x = attention_weights @ crystal_values
    return np.round(x, 3).flatten()


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/94
    """
    
    seq_len, d_model = Q.shape
    
    # Ensure d_model is divisible by n_heads
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    # Dimension of each head
    d_k = d_model // n_heads
    
    # Split Q, K, V into multiple heads
    # Reshape from (seq_len, d_model) to (seq_len, n_heads, d_k)
    Q_heads = Q.reshape(seq_len, n_heads, d_k)
    K_heads = K.reshape(seq_len, n_heads, d_k)
    V_heads = V.reshape(seq_len, n_heads, d_k)
    
    # Store outputs from each head
    head_outputs = []
    
    # Compute attention for each head
    for i in range(n_heads):
        Q_h = Q_heads[:, i, :]  # (seq_len, d_k)
        K_h = K_heads[:, i, :]  # (seq_len, d_k)
        V_h = V_heads[:, i, :]  # (seq_len, d_k)
        
        # Scaled dot-product attention
        scores = Q_h @ K_h.T / np.sqrt(d_k)
        
        # Softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Weighted sum of values
        head_output = attention_weights @ V_h  # (seq_len, d_k)
        head_outputs.append(head_output)
    
    # Concatenate all heads
    output = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
    
    return output


def sparse_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int, scale_factor=None) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/131
    """
    seq_len, d_k = Q.shape
    
    # Set scale factor
    if scale_factor is None:
        scale_factor = np.sqrt(d_k)
    
    # Compute full attention scores
    scores = Q @ K.T / scale_factor  # (seq_len, seq_len)
    
    # Create window mask
    # mask[i, j] = True if j is within window of i
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    
    for i in range(seq_len):
        # Define window boundaries
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = True
    
    # Apply mask: set positions outside window to large negative value
    scores_masked = np.where(mask, scores, -1e9)
    
    # Apply softmax
    attention_weights = np.exp(scores_masked - np.max(scores_masked, axis=-1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Zero out attention weights outside the window for true sparsity
    attention_weights = np.where(mask, attention_weights, 0)
    
    # Compute output
    output = attention_weights @ V
    
    return output


def pos_encoding(position: int, d_model: int) -> np.ndarray:
    """
    https://www.deep-ml.com/problems/85
    """
    # edge cases
    if position <= 0 or d_model <= 0:
          return -1
    
    # positions: (position, 1), dims: (1, d_model)
    pos = np.arange(position, dtype=np.float32)[:, None]
    i = np.arange(d_model, dtype=np.float32)[None, :]
  
    # angle rates per dimension (same for even/odd pairs)
    angle_rads = pos / np.power(10000.0, (2 * (i // 2)) / d_model)

    # create pos_encoding (position, d_model) and apply sin/cos functions
    pos_encoding = np.empty((position, d_model), dtype=np.float32)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])   # even indices
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])   # odd indices
  
    return np.float16(pos_encoding)
  
