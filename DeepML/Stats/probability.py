"""
Implementation of probability problems
"""
import math
import numpy as np
import torch
from collections import Counter
from typing import List, Tuple, Any, Literal


def binomial_probability(n: int, k: int, p: float) -> float:
    """
    79. Calculate the probability of exactly k successes in n Bernoulli trials.
    https://www.deep-ml.com/problems/79
	
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    Returns:
        Probability of k successes
    """
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
	

def normal_pdf(x, mean, std_dev):
	"""
	80. Calculate the probability density function (PDF) of the normal distribution.
	https://www.deep-ml.com/problems/80
	
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	val = math.e ** (-(x - mean) ** 2 / (2 * std_dev ** 2)) / math.sqrt(2 * math.pi * std_dev ** 2)
	return round(val, 5)


def poisson_probability(k, lam):
	"""
	81. Calculate the probability of observing exactly k events in a fixed interval, given the mean rate of events lam, using the Poisson distribution formula.
	https://www.deep-ml.com/problems/81
	
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	val = (lam ** k) * math.exp(-lam) / math.factorial(k)
	return round(val, 5)


def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    136. Computes the KL divergence between two multivariate Gaussian distributions.
	https://www.deep-ml.com/problems/136
    
    Parameters:
	    mu_p: mean vector of the first distribution
	    Cov_p: covariance matrix of the first distribution
	    mu_q: mean vector of the second distribution
	    Cov_q: covariance matrix of the second distribution
    Returns:
    	KL divergence as a float
    """
    k = len(mu_p)
    covq_inv = np.linalg.inv(Cov_q)
    diff = mu_p - mu_q
    log_det = np.log(np.linalg.det(Cov_q) / np.linalg.det(Cov_p))
    trace_term = np.trace(covq_inv @ Cov_p)
    mahal = diff.T @ covq_inv @ diff
    return 0.5 * (log_det - k + trace_term + mahal)


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


def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    182. Simulate the Central Limit Theorem.
	https://www.deep-ml.com/problems/182

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.
    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    sample_means = np.zeros(runs)

	# different sample mean and variance for each distribution
    if distribution == 'uniform':
        mu, sigma = 0.5, np.sqrt(1/12)
    elif distribution == 'exponential':
        mu, sigma = 1.0, 1.0
    elif distribution == 'bernoulli':
        mu, sigma = 0.3, np.sqrt(0.3 * 0.7)

    for i in range(runs):
        if distribution == 'uniform':
            out = np.random.uniform(low=0, high=1, size=n)
        elif distribution == 'exponential':
            out = np.random.exponential(scale=1.0, size=n)
        elif distribution == 'bernoulli':
            # using this would cause the seed to be different: out = np.random.binomial(n=1, p=0.3, size=n)
            out = (np.random.uniform(size=n) < 0.3).astype(int)
        sample_means[i] = np.mean(out)

    standardized = (sample_means - mu) / (sigma / np.sqrt(n))
    return {
        'mean': np.mean(standardized),
        'std': np.std(standardized)
    }
	

def simulate_clt_torch(num_samples: int, sample_size: int, distribution: Literal['uniform', 'exponential'] = 'uniform') -> float:
    """PyTorch version"""
    if distribution == 'uniform':
        samples = torch.rand(num_samples, sample_size)
    else:  # exponential
        samples = torch.distributions.Exponential(1).sample((num_samples, sample_size))
    
    sample_means = torch.mean(samples, dim=1)
    return float(torch.mean(sample_means))
    

def empirical_pmf(samples):
	"""
    184. Given an iterable of integer samples, return a list of (value, probability) pairs sorted by value ascending.
	https://www.deep-ml.com/problems/184
    """
    counts = Counter(samples)
    total = sum(counts.values())
    return [(val, counts[val] / total) for val in sorted(counts)]


def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    243. Compute the covariance of X and Y from their joint PMF.
	https://www.deep-ml.com/problems/243
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    Returns:
        Covariance of X and Y as a float
    """
    X, Y = np.array(x_values), np.array(y_values)
    P = np.array(joint_pmf)

	# sum rows/columns in PMF to get probability of each value
    p_x = P.sum(axis=1)  # shape (len(X),)
    p_y = P.sum(axis=0)  # shape (len(Y),)
	# compute E(X), E(Y), and E(XY) - ΣᵢΣⱼ xᵢyⱼ P(xᵢ, yⱼ)
    E_X  = X @ p_x
    E_Y  = Y @ p_y
    E_XY = np.sum(np.outer(X, Y) * P)
    
    return E_XY - E_X * E_Y


def law_of_total_probability(priors: dict, conditionals: dict) -> float:
    """
    244. Compute P(A) using the Law of Total Probability.
    https://www.deep-ml.com/problems/244
	
    Args:
        priors: Dictionary mapping partition event names to P(Bi)
        conditionals: Dictionary mapping partition event names to P(A|Bi)
    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    P_A = 0
    for key in conditionals:
        P_A += conditionals[key] * priors[key]
    return P_A


def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    245. Calculate the PMF of the hypergeometric distribution.
    https://www.deep-ml.com/problems/244
    
    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes
    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    return round(math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n), 4)


def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	336. Calculate posterior probabilities using Bayes' Theorem.
    https://www.deep-ml.com/problems/336
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	joint = [p * l for p, l in zip(priors, likelihoods)]
	evidence = sum(joint)
	posteriors = [p / evidence for p in joint]
	return posteriors
	
