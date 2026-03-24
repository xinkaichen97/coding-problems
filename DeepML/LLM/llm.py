"""
Implementation of LLM problems
"""
import numpy as np


def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    383. Apply top-p (nucleus) sampling to filter a probability distribution.
    https://www.deep-ml.com/problems/383
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    # apply softmax
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # get sorted probabilities
    # use stable kind to preserve order for ties
    sorted_indices = np.argsort(-probs, kind='stable')
    sorted_probs = probs[sorted_indices]

    # filter top-p
    cumsum = np.cumsum(sorted_probs)
    cumsum[-1] = 1.0  # force exact 1.0 to handle float precision
    idx = np.argmax(cumsum >= p)
    sorted_probs[idx + 1:] = 0.0

    # restore the original order and normalize
    filtered_probs = np.zeros_like(probs)
    filtered_probs[sorted_indices] = sorted_probs
    filtered_probs /= filtered_probs.sum()
    
    return np.round(filtered_probs, 4).tolist()
  
