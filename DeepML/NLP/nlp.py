"""
Problems for Natural Language Processing
"""
import string


def unigram_probability(corpus: str, word: str) -> float:
    """
    129. Calculate Unigram Probability from Corpus
    https://www.deep-ml.com/problems/129
    """
    words = corpus.split()
    cnt, total = 0, 0
    for w in words:
        total += 1
        if w == word:
            cnt += 1
    return round(cnt / total, 4)
  

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    320. Calculate Perplexity for Language Models
    https://www.deep-ml.com/problems/320
    Args:
        probabilities: List of probabilities P(token_i | context) for each token in the sequence, where each probability is in (0, 1]
    Returns:
        Perplexity value as a float
    """
    probs = np.array(probabilities)
    pp = np.exp(-np.mean(np.log(probs)))
    return pp
    

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    325. Calculate the exact match score between predictions and references.
    https://www.deep-ml.com/problems/325
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    Returns:
        Exact match score as a float between 0 and 1
    """
    if (not predictions and not references) or (len(predictions) != len(references)):
        return 0.0

    def normalize(s: str) -> str:
        s = s.lower()
        s = "".join(c for c in s if c not in string.punctuation) # remove punctuation first
        s = " ".join(s.split())
        return s.strip()

    matched = sum(normalize(pred) == normalize(ref) for pred, ref in zip(predictions, references))
    return matched / len(predictions)
    
