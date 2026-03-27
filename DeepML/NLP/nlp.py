"""
Problems for Natural Language Processing
"""
import numpy as np
import string


def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    110. Calculate METEOR score for machine translation evaluation.
    https://www.deep-ml.com/problems/110
    
    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)
    Returns:
        METEOR score between 0 and 1
    """
    ref_list = reference.strip().lower().split()
    cand_list = candidate.strip().lower().split()
    ref_available = ref_list.copy()

    matches = 0
    match_idx = []
    matched = False
    
    for i, word in enumerate(cand_list):
        if word in ref_available:
            ri = ref_available.index(word)
            ref_available[ri] = None
            match_idx.append(ri)
            matches += 1
    
    if len(match_idx) > 0:
        chunks = 1
        for i in range(len(match_idx) - 1):
            if match_idx[i + 1] != match_idx[i] + 1:
                chunks += 1
    else:
        chunks = 0

    precision = matches / len(cand_list)
    recall = matches / len(ref_list)
    F_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

    penalty = gamma * (chunks / matches) ** beta
    meteor = F_mean * (1 - penalty)
    return meteor


def meteor_score_np(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    110. Calculate METEOR score for machine translation evaluation.
    https://www.deep-ml.com/problems/110
    """
    # numpy version
    ref_tokens = np.array(reference.lower().split())
    cand_tokens = np.array(candidate.lower().split())

    # --- 1. Greedy exact-match alignment ---
    ref_matched  = np.zeros(len(ref_tokens),  dtype=bool)
    cand_matched = np.zeros(len(cand_tokens), dtype=bool)

    for ci, cw in enumerate(cand_tokens):
        hits = np.where((ref_tokens == cw) & ~ref_matched)[0]
        if hits.size:
            ref_matched[hits[0]]  = True
            cand_matched[ci]      = True

    m = cand_matched.sum()          # number of matched unigrams
    if m == 0:
        return 0.0

    # --- 2. Precision & Recall ---
    P = m / len(cand_tokens)
    R = m / len(ref_tokens)

    # --- 3. Harmonic F-mean (alpha controls P vs R weight) ---
    F = P * R / (alpha * P + (1.0 - alpha) * R)

    # --- 4. Fragmentation penalty ---
    # Count contiguous chunks in the CANDIDATE that are matched
    cand_flags = cand_matched.astype(int)
    # A new chunk starts wherever a matched token is preceded by an unmatched one
    # (or is the very first matched token)
    chunks = int(np.sum(np.diff(np.concatenate(([0], cand_flags, [0]))) == 1))

    penalty = gamma * (chunks / m) ** beta

    return float(F * (1.0 - penalty))


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
    
