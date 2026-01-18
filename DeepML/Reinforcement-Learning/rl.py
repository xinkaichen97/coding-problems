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

