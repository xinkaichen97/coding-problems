"""
Problems for Natural Language Processing
"""


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
    
