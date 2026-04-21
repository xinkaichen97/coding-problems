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


def phi_corr(x: list[int], y: list[int]) -> float:
  	"""
  	95. Calculate the Phi coefficient between two binary variables.
  	https://www.deep-ml.com/problems/95
  
  	Args:
  	x (list[int]): A list of binary values (0 or 1).
  	y (list[int]): A list of binary values (0 or 1).
  
  	Returns:
  	float: The Phi coefficient rounded to 4 decimal places.
  	"""
    x, y = np.array(x), np.array(y)
    x_00 = np.sum((x == 0) & (y == 0))
    x_01 = np.sum((x == 0) & (y == 1))
    x_10 = np.sum((x == 1) & (y == 0))
    x_11 = np.sum((x == 1) & (y == 1))
    
    val = (x_00 * x_11 - x_01 * x_10) / np.sqrt((x_00 + x_01) * (x_10 + x_11) * (x_00 + x_10) * (x_01 + x_11))
  	return round(val, 4)


def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    """
    120. Bhattacharyya Distance Between Two Distributions
    https://www.deep-ml.com/problems/120
    """
    if not p or not q or len(p) != len(q):
        return 0.0
        
    p = np.array(p)
    q = np.array(q)
    bc = np.sum(np.sqrt(p * q))
    bd = -np.log(bc)
    
    return round(bd, 4)


def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    183. Calculate the variance of a portfolio
    https://www.deep-ml.com/problems/183

    Args:
        cov_matrix (list[list[float]]): Covariance matrix of asset returns.
        weights (list[float]): Portfolio weights.
    Returns:
        float: Portfolio variance.
    """
    # Convert to numpy arrays
    cov = np.array(cov_matrix)
    w = np.array(weights)
    
    # Calculate portfolio variance: w^T @ cov @ w
    portfolio_var = w.T @ cov @ w

    # # torch version
    # # Convert inputs to tensors
    # cov = torch.tensor(cov_matrix, dtype=torch.float)
    # w = torch.tensor(weights, dtype=torch.float)

    # # Calculate portfolio variance: w^T @ cov @ w
    # portfolio_var = w @ cov @ w
    
    return float(portfolio_var)
    

def two_sample_t_test(sample1: list[float], sample2: list[float], alpha: float = 0.05) -> dict:
	"""
	211. Perform a two-sample independent t-test (Welch's t-test).
    https://www.deep-ml.com/problems/211
	
	Args:
		sample1: First sample data
		sample2: Second sample data
		alpha: Significance level (default 0.05)
	Returns:
		Dictionary containing:
		- t_statistic: The calculated t-statistic
		- p_value: Two-tailed p-value
		- degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
		- reject_null: Boolean, whether to reject null hypothesis
		- cohens_d: Effect size (Cohen's d)
	"""
    # calculate mean and variance
	n1, n2 = len(sample1), len(sample2)
	sample1, sample2 = np.array(sample1), np.array(sample2)
	mean1, mean2 = np.mean(sample1), np.mean(sample2)
	var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    # calculate standard error and t-value
	SE = np.sqrt(var1 / n1 + var2 / n2)
	t_statistic = (mean1 - mean2) / SE

    # calculate df (Welch-Satterthwaite)
	degrees_of_freedom = (var1 / n1 + var2 / n2) ** 2 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

    # calculate p-value using 1 - CDF or SF, two-tailed -> *2
	p_value = 2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
	reject_null = (p_value < alpha)

    # calculate Cohen's d
    # pooled SD: Student's -> df-weighted, Welch's -> average
	# s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
	s_pooled = np.sqrt((var1 + var2) / 2)
	cohens_d = (mean1 - mean2) / s_pooled

	return {
		't_statistic': t_statistic,
		'p_value': p_value,
		'degrees_of_freedom': degrees_of_freedom,
		'reject_null': reject_null,
		'cohens_d': cohens_d
	}


def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    296. Calculate statistical power for a two-sample z-test.
    https://www.deep-ml.com/problems/296
    
    Parameters:
        effect_size: Cohen's d (standardized effect size)
        sample_size_per_group: Number of observations per group
        alpha: Significance level (default 0.05)
        two_tailed: Whether the test is two-tailed (default True)
    Returns:
        Statistical power as a float rounded to 4 decimal places
    """
    # use built-in functions
    try:
        from math import erfinv
    except ImportError:
        from scipy.special import erfinv

    # or use the approximate function
    def erfinv(x: float) -> float:
        """ Approximate erfinv via Newton's method. """
        # initial guess using rational approximation
        a = 0.147
        ln = math.log(1 - x * x)
        t = 2 / (math.pi * a) + ln / 2
        result = math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)
        
        # Newton refinement: erfinv(x) = root of erf(y) - x = 0
        for _ in range(3):
            result -= (math.erf(result) - x) / (2 / math.sqrt(math.pi) * math.exp(-result**2))
        
        return result

    # non-centrality parameter
    ncp = effect_size * math.sqrt(sample_size_per_group / 2)

    # inverse CDF of normal using erfinv
    def norm_ppf(p):
        return math.sqrt(2) * math.erfinv(2 * p - 1)

    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    if two_tailed:
        z_critical = norm_ppf(1 - alpha / 2)
        power = 1 - norm_cdf(z_critical - ncp) - norm_cdf(-z_critical - ncp)
    else:
        z_critical = norm_ppf(1 - alpha)
        power = 1 - norm_cdf(z_critical - ncp)
    
    return round(power, 4)


def gaussian_mle(data: np.ndarray) -> tuple:
    """
    337. Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    https://www.deep-ml.com/problems/337
    
    Args:
        data: 1D numpy array of observations
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    data = np.array(data)
    mean_mle = np.mean(data)
    variance_mle = np.mean(np.square(data - mean_mle))
    return (mean_mle, variance_mle)


def exponential_distribution(x: list, lam: float) -> dict:
    """
    340. Compute exponential distribution properties.
    https://www.deep-ml.com/problems/340
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution  
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    if lam <= 0:
        return {
            'pdf': None, 'cdf': None, 'mean': None, 'variance': None
        }
    x = np.array(x)
    pdf = np.where(x < 0, 0, lam * np.exp(-lam * x))
    cdf = np.where(x < 0, 0, 1 - np.exp(-lam * x))
    
    return {
        'pdf': np.round(pdf, 4).tolist(), 'cdf': np.round(cdf, 4).tolist(), 'mean': round(1 / lam, 4), 'variance': round(1 / lam ** 2, 4)
    }
    

def law_of_large_numbers(n_samples: int, population_mean: float, population_std: float) -> float:
    """
    342. Demonstrate the Law of Large Numbers by computing the sample mean.
    https://www.deep-ml.com/problems/342
    
    Args:
        n_samples: Total number of samples to draw from the distribution
        population_mean: The true mean of the population distribution
        population_std: The true standard deviation of the population distribution
    Returns:
        The sample mean
    """
    samples = np.random.normal(loc=population_mean, scale=population_std, size=n_samples)
    return np.mean(samples)
    
