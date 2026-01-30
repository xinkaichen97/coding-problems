"""
Implementation of MLOps problems
"""
import numpy as np


def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    248. Calculate inference statistics for model monitoring.
    https://www.deep-ml.com/problems/248
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    # edge case, return empty dictionary
    if not latencies_ms:
        return {}
      
    latencies_arr = np.array(latencies_ms)
    avg_latency_ms = np.round(np.mean(latencies_arr), 2)
    throughput_per_sec = np.round(1000 / avg_latency_ms, 2)
    # use np.percentile or np.quantile
    p50_ms, p95_ms, p99_ms = np.round(np.percentile(latencies_arr, [50, 95, 99]), 2)

    return {'throughput_per_sec': throughput_per_sec, 'avg_latency_ms': avg_latency_ms, 'p50_ms': p50_ms, 'p95_ms': p95_ms, 'p99_ms': p99_ms}
  
