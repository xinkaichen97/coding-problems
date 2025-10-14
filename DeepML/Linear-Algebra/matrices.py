"""
Implementation of some matrix functions
"""
import numpy as np


def matrix_dot_vector(a:list[list[int|float]], b:list[int|float])-> list[int|float]:
  """
  1. Matrix-Vector Dot Product
  """
  c = []
  for row in a:
    if len(row) != len(b):
      return -1
  for row in a:
    c.append(sum([row[i] * b[i] for i in range(len(row))]))
  return c
  

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
  """
  2. Transpose of a Matrix
  """
  return [list(i) for i in zip(*a)]


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
  """
  8. Calculate 2x2 Matrix Inverse
  """
  a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
  det = a * d - b * c
  if det == 0:
    return None
  inverse = [[d / det, - b / det], [- c / det, a / det]]
  return inverse


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
  """
  27. Transformation Matrix from Basis B to C
  """
  # np.array(): Always creates a new copy of the data
  B = np.array(B) 
  C = np.array(C)
  P = np.linalg.inv(C).dot(B)
  return P
  
