"""
Implementation of some matrix functions
"""
import numpy as np


def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    """
    1. Matrix-Vector Dot Product
    https://www.deep-ml.com/problems/1
    """
    c = []
    for row in a:
        if len(row) != len(b):
          return -1
    for row in a:
        c.append(sum([row[i] * b[i] for i in range(len(row))]))
      
    # # numpy
    # a, b = np.array(a), np.array(b)
    # if a.shape[1] != b.shape[0]:
    #     return -1
    # c = (a @ b).tolist()
  
    return c
  

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    2. Transpose of a Matrix
    https://www.deep-ml.com/problems/2
    """
    # # numpy
    # return np.array(a).T.tolist()
    return [list(i) for i in zip(*a)]


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    """
    8. Calculate 2x2 Matrix Inverse
    https://www.deep-ml.com/problems/8
    """
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    det = a * d - b * c
    if det == 0:
        return None
    inverse = [[d / det, - b / det], [- c / det, a / det]]
  
    # # numpy
    # matrix = np.array(matrix)
    # inverse = np.linalg.inv(matrix).tolist()
  
    return inverse


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    """
    27. Transformation Matrix from Basis B to C
    https://www.deep-ml.com/problems/27
    """
    # np.array(): Always creates a new copy of the data
    B = np.array(B) 
    C = np.array(C)
    P = np.linalg.inv(C).dot(B)
    return P


def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    195. Compute the determinant and trace of a square matrix
    https://www.deep-ml.com/problems/195
    
    Args:
      matrix: A square matrix (n x n) represented as list of lists
    Returns:
      Tuple of (determinant, trace)
    """
    matrix = np.array(matrix)
	  return (np.linalg.det(matrix), np.trace(matrix))
  
