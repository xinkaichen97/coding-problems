"""
Implementation of RL problems
"""
import numpy as np
from typing import Union, List


def incremental_mean(Q_prev: float, k: int, R: float) -> float: 
    """
    159. Incremental Mean for Online Reward Estimation
    https://www.deep-ml.com/problems/159
    
    Q_prev: previous mean estimate (float)
    k: number of times the action has been selected (int)
    R: new observed reward (float)
    Returns: new mean estimate (float)
    """
    return Q_prev + (R - Q_prev) / k


def exp_weighted_average(Q1: float, rewards: Union[List, np.ndarray], alpha: float) -> float:
    """
    161. Exponential Weighted Average of Rewards
    https://www.deep-ml.com/problems/161
    
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    k = len(rewards)
    res = (1 - alpha) ** k * Q1 
    for i in range(k):
        # i goes from 0 to k-1, same with the exponential
        res += alpha * (1 - alpha) ** (k - 1 - i) * rewards[i]
    return res


def ucb_action(counts: np.ndarray, values: np.ndarray, t: int, c: float) -> int:
    """
    162. Upper Confidence Bound (UCB) Action Selection
    https://www.deep-ml.com/problems/162
    
    Args:
      counts (np.ndarray): Number of times each action has been chosen
      values (np.ndarray): Average reward of each action
      t (int): Current timestep (starts from 1)
      c (float): Exploration coefficient
    Returns:
      int: Index of action to select
    """
    ucb = values + c * np.sqrt(np.log(t) / counts)
    return np.argmax(ucb)


def discounted_return(rewards: Union[List[float], np.ndarray], gamma: float) -> float:
    """
    165/167. Compute the total discounted return for a sequence of rewards.
    https://www.deep-ml.com/problems/165
    https://www.deep-ml.com/problems/167
    
    Args:
        rewards (list or np.ndarray): List or array of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return
    """
    rewards = np.array(rewards)
    discount_factors = gamma ** np.arange(len(rewards))
    return float(np.sum(rewards * discount_factors))


def compute_group_relative_advantage(rewards: list[float]) -> list[float]:
    """
    224. Compute the Group Relative Advantage for GRPO
    https://www.deep-ml.com/problems/224
    
    For each reward r_i in a group, compute:
    A_i = (r_i - mean(rewards)) / std(rewards)
    If all rewards are identical (std=0), return zeros.
    Args:
        rewards: List of rewards for a group of outputs from the same prompt
    Returns:
        List of normalized advantages
    """
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    sd = np.std(rewards)
    res = (rewards - mean) / sd if sd != 0 else np.zeros_like(rewards) # handle zero
    return res.tolist()


def kl_divergence_estimator(pi_theta: np.ndarray, pi_ref: np.ndarray) -> np.ndarray:
    """
    225. Compute the unbiased KL divergence estimator used in GRPO
    https://www.deep-ml.com/problems/225
    
    Formula: D_KL = (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
    Args:
        pi_theta: Current policy probabilities for each sample
        pi_ref: Reference policy probabilities for each sample
    Returns:
        Array of KL divergence estimates (one per sample)
    """
    kl = (pi_ref / pi_theta) - np.log(pi_ref / pi_theta) - 1
    return kl

