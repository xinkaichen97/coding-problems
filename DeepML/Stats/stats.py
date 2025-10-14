"""
Implementation of some statistics problems
"""
import numpy as np 
from scipy import stats


def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
  """
  10. Calculate Covariance Matrix
  """
  n = len(vectors)
  m = len(vectors[0])
  means = [sum(v) / m for v in vectors]
  cov_matrix = [[0 for _ in range(n)] for _ in range(n)]
  for i in range(n):
    for j in range(i, n):
      cov = sum([(vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(m)]) / (m - 1)
      cov_matrix[i][j] = cov_matrix[j][i] = cov
  return cov_matrix


def descriptive_statistics(data):
  """
  78. Descriptive Statistics Calculator
  """
  data = np.asarray(data).flatten()
  mean = np.mean(data)
  median = np.median(data)
  mode = stats.mode(data)[0]
  variance = np.var(data)
  std_dev = np.std(data)
  percentiles = [np.percentile(data, 25), np.percentile(data, 50), np.percentile(data, 75)]
  iqr = percentiles[2] - percentiles[0]
  stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance,4),
        "standard_deviation": np.round(std_dev,4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }
    return stats_dict
