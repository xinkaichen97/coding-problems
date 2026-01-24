"""
Implementation of probability problems
"""
from typing import List, Tuple, Any, Literal


def conditional_probability(data: List[Tuple[Any, Any]], x: Any, y: Any) -> float:
    """
    168. Calculate Conditional Probability from Data
    https://www.deep-ml.com/problems/168
    
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    count_x, count_xy = 0, 0

    # manual counter is faster than list comprehension
    for xi, yi in data:
        if xi == x:
            count_x += 1
            if yi == y:
                count_xy += 1

    # # numpy
    # data = np.array(data)
    # x_mask = data[:, 0] == x
    # count_x = np.sum(x_mask)
    # y_mask = data[:, 1] == y
    # count_xy = np.sum(x_mask & y_mask)
  
    # handle special case
    if count_x == 0:
        return 0.0
    
    return round(count_xy / count_x, 4)
  

def dice_statistics(n: int) -> Tuple[float, float]:
	"""
    179. Expected Value and Variance of an n-Sided Die
    https://www.deep-ml.com/problems/179
    
	Args:
		n (int): Number of sides of the die
	Returns:
		tuple: (expected_value, variance)
	"""
    ex = (n + 1) / 2
    var = (n ** 2 - 1) / 12
	return (ex, var)


def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    181. Sampling Distribution of the Mean
    https://www.deep-ml.com/problems/181
    
    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)
    Returns:
        Mean of the sample means (float)
    """
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, (num_samples, sample_size))
    else:  # exponential
        samples = np.random.exponential(1, (num_samples, sample_size))
        
    # calculate the mean of each independent sample, and then calculate the mean again
    sample_means = np.mean(samples, axis=1)
    return float(np.mean(sample_means))


def simulate_clt_torch(num_samples: int, sample_size: int, distribution: Literal['uniform', 'exponential'] = 'uniform') -> float:
    """PyTorch version"""
    if distribution == 'uniform':
        samples = torch.rand(num_samples, sample_size)
    else:  # exponential
        samples = torch.distributions.Exponential(1).sample((num_samples, sample_size))
    
    sample_means = torch.mean(samples, dim=1)
    return float(torch.mean(sample_means))
    
