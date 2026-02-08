"""
Implementation of Trees
"""
import numpy as np


def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """
    20. Build a decision tree using the ID3 algorithm with NumPy optimization.
    https://www.deep-ml.com/problems/20
    """
    
    # Extract labels as a numpy array
    labels = np.array([example[target_attr] for example in examples])
    
    # Base case 1: All examples have the same class
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return str(unique_labels[0])
    
    # Base case 2: No more attributes to split on
    if not attributes:
        return majority_class(examples, target_attr)
    
    # Find the best attribute to split on (maximum information gain)
    # Calculate gains for all attributes at once
    gains = np.array([calculate_information_gain(examples, attr, target_attr) 
                      for attr in attributes])
    
    # Use argmax to get the first attribute with maximum gain
    best_attr_idx = np.argmax(gains)
    best_attr = attributes[best_attr_idx]
    
    # Create the tree node for the best attribute
    tree = {}
    branches = {}
    
    # Get all unique values for the best attribute and sort them
    attr_values = np.array([example[best_attr] for example in examples])
    sorted_values = np.sort(np.unique(attr_values))
    
    # Recursively build subtrees for each value
    remaining_attributes = [attr for attr in attributes if attr != best_attr]
    
    for value in sorted_values:
        # Get subset of examples with this value
        subset = [ex for ex in examples if ex[best_attr] == value]
        
        if not subset:
            # No examples with this value - use majority class of parent
            branches[str(value)] = majority_class(examples, target_attr)
        else:
            # Recursively build subtree
            branches[str(value)] = learn_decision_tree(subset, remaining_attributes, target_attr)
    
    tree[best_attr] = branches
    
    return tree


def calculate_entropy(labels: np.ndarray) -> float:
    """Calculate the entropy of a numpy array of labels."""
    if len(labels) == 0:
        return 0.0
    
    # Get unique labels and their counts
    unique, counts = np.unique(labels, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(labels)
    
    # Calculate entropy: -sum(p_i * log2(p_i))
    # Filter out zero probabilities before log
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy)


def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    # Extract labels as numpy array
    labels = np.array([example[target_attr] for example in examples])
    
    # Calculate entropy of the current set
    total_entropy = calculate_entropy(labels)
    
    # Extract attribute values as numpy array
    attr_values = np.array([example[attr] for example in examples])
    
    # Get unique values and calculate weighted entropy
    unique_values = np.unique(attr_values)
    total_examples = len(examples)
    weighted_entropy = 0.0
    
    for value in unique_values:
        # Create boolean mask for this value
        mask = attr_values == value
        
        # Get subset of labels
        subset_labels = labels[mask]
        
        # Calculate weighted contribution
        weight = len(subset_labels) / total_examples
        subset_entropy = calculate_entropy(subset_labels)
        weighted_entropy += weight * subset_entropy
    
    # Information gain
    information_gain = total_entropy - weighted_entropy
    
    return float(information_gain)


def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    if not examples:
        return None
    
    # Extract labels as numpy array
    labels = np.array([example[target_attr] for example in examples])
    
    # Count class occurrences
    unique, counts = np.unique(labels, return_counts=True)
    
    # Find maximum count
    max_count = np.max(counts)
    
    # Get all classes with max count
    tied_classes = unique[counts == max_count]
    
    # Sort alphabetically and return first
    tied_classes_sorted = np.sort(tied_classes)
    
    return str(tied_classes_sorted[0])

