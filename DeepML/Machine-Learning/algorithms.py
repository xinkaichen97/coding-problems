"""
Implementation of common ML algorithms
"""
import numpy as np


def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	
    # convert to numpy arrays for easier computation
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # step 1: assign each point to the nearest centroid
        # compute distances from each point to each centroid
        # shape: (n_points, k)

		    # add new axis to points for broadcasting: (n_points, n_dims) → (n_points, 1, n_dims)
        points_expanded = points[:, np.newaxis]

		    # compute the difference between each point and each centroid
        # broadcasting: (n_points, 1, n_dims) - (k, n_dims) → (n_points, k, n_dims)
        squared_differences = (points_expanded - centroids) ** 2

		    # sum across dimensions to get squared Euclidean distance
        # shape: (n_points, k, n_dims) → (n_points, k)
        distances = np.sqrt(squared_differences.sum(axis=2))
        
        # get cluster assignment for each point
        assignments = np.argmin(distances, axis=1)
        
        # step 2: update centroids
        new_centroids = np.array([
            points[assignments == i].mean(axis=0) if np.any(assignments == i) else centroids[i]
            for i in range(k)
        ])
        
        # check for convergence (centroids don't change)
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
            
        centroids = new_centroids
    
    # convert back to a list of tuples
    final_centroids = [tuple(centroid) for centroid in centroids]
    
    return final_centroids
  
  
