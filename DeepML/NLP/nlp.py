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
  
