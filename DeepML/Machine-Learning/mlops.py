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
  

def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    249. Calculate health metrics for a batch prediction job.
    https://www.deep-ml.com/problems/249
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    # edge case: empty input
    if not predictions:
        return {}

    # get all confidences and convert to an array
    confidences = np.array([pred['confidence'] for pred in predictions if 'confidence' in pred])

    # handle case where the denominator is zero
    if len(confidences) > 0:
        success_rate = len(confidences) / len(predictions) * 100
        avg_confidence = np.mean(confidences) * 100
        low_confidence_rate = np.sum(confidences < confidence_threshold) / len(confidences) * 100
    else:
        success_rate = 0.0
        avg_confidence = 0.0
        low_confidence_rate = 0.0

    return {
        'success_rate': np.round(success_rate, 2),
        'avg_confidence': np.round(avg_confidence, 2),
        'low_confidence_rate': np.round(low_confidence_rate, 2)
    }
    
