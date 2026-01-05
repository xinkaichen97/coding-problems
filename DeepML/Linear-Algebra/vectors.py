"""
Implementation of some vector functions
"""
import numpy as np


def orthogonal_projection(v, L):
    """
    66. Compute the orthogonal projection of vector v onto line L.
    https://www.deep-ml.com/problems/66
    
    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    # Convert to numpy arrays
    v = np.array(v)
    L = np.array(L)
    
    # Compute the projection using the formula: proj_L(v) = ((v·L)/(L·L)) * L
    projection = (np.dot(v, L) / np.dot(L, L)) * L
    
    return projection


def cosine_similarity(v1, v2):
    """
    76. Calculate Cosine Similarity Between Vectors
    https://www.deep-ml.com/problems/76
    """
    # Convert to arrays
    v1 = np.asarray(v1, dtype=float).flatten()
    v2 = np.asarray(v2, dtype=float).flatten()
    
    # Check empty
    if v1.size == 0 or v2.size == 0:
        raise ValueError("Vectors cannot be empty")
    
    # Check shape match
    if v1.shape != v2.shape:
        raise ValueError(f"Shape mismatch: {v1.shape} vs {v2.shape}")
    
    # Calculate magnitudes
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Check zero magnitude
    if np.isclose(norm_v1, 0) or np.isclose(norm_v2, 0):
        raise ValueError("Vectors cannot have zero magnitude")
    
    # Calculate cosine similarity
    return round(np.dot(v1, v2) / (norm_v1 * norm_v2), 3)


def cross_product(a, b):
    """
    118. Compute the Cross Product of Two 3D Vectors
    https://www.deep-ml.com/problems/118
    """
    ## numpy's built-in function
    # a, b = np.array(a), np.array(b)
    # return np.cross(a, b)
    ans_1 = a[1] * b[2] - a[2] * b[1]
    ans_2 = a[2] * b[0] - a[0] * b[2]
    ans_3 = a[0] * b[1] - a[1] * b[0]
    return [ans_1, ans_2, ans_3]


def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    """
    121. Vector Element-wise Sum
    https://www.deep-ml.com/problems/121
    
    Return the element-wise sum of vectors 'a' and 'b'.
    If vectors have different lengths, return -1.
    """
    a, b = np.array(a), np.array(b)
    
    if a.shape != b.shape:
        return -1
    
    return (a + b).tolist()

