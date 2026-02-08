"""
Implementation of common ML algorithms
"""
import numpy as np


def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	"""
	17. K-Means Clustering
	https://www.deep-ml.com/problems/17
	"""
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
  
  
def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    18. Generate train/test index splits for k-fold cross-validation.
	https://www.deep-ml.com/problems/18
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # create indices array
    indices = np.arange(n_samples)
    
    # shuffle in-place if needed
    if shuffle:
        np.random.shuffle(indices)
    
    # split indices into k folds
    folds = np.array_split(indices, k)

    # generate train/test splits
    res = []
    for i in range(k):
        # test fold is the i-th fold
        test = folds[i].tolist()
        # train folds are all other folds concatenated
        train = np.concatenate(folds[:i] + folds[i + 1:]).tolist()

        res.append((train, test))

    return res
	

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    19. Perform PCA and return the top k principal components.
	https://www.deep-ml.com/problems/19
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
	# standardize data
	mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1  # prevent division by zero
    centered_data = (data - mean) / std

	# compute covariance matrix and eigenvalues/eigenvectors
    n_samples = centered_data.shape[0]
    cov_matrix = (centered_data.T @ centered_data) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

	# find the top-k eigenvectors (each col is an eigenvector)
	# no need to do argsort when using np.linalg.eigh (eigenvalues are in ascending order)
    # sorted_indices = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[sorted_indices]
    # eigenvectors = eigenvectors[:, sorted_indices]
	eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    top_k_eigenvectors = eigenvectors[:, :k]

	# fix sign convention: first non-zero element should be positive
    for i in range(k):
        eigenvector = top_k_eigenvectors[:, i]
        non_zero_indices = np.where(np.abs(eigenvector) > 1e-10)[0]
        if len(non_zero_indices) > 0:
            first_non_zero_idx = non_zero_indices[0]
            if eigenvector[first_non_zero_idx] < 0:
                top_k_eigenvectors[:, i] = -eigenvector

    principal_components = np.round(top_k_eigenvectors, 4)
    
    return principal_components
	

def k_nearest_neighbors(points, query_point, k):
    """
    173. Find k nearest neighbors to a query point
	https://www.deep-ml.com/problems/173
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    # convert to numpy arrays
    points_array = np.array(points)
    query_array = np.array(query_point)
    
    # compute Euclidean distances
    distances = np.sqrt(((points_array - query_array) ** 2).sum(axis=1))
    
    # get indices sorted by distance (stable sort preserves order for ties)
    sorted_indices = np.argsort(distances, kind='stable')
    
    # select top k indices
    top_k_indices = sorted_indices[:k]
    
    # return corresponding points as a list of tuples
    k_nearest = [points[i] for i in top_k_indices]
    
    return k_nearest
