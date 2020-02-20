import torch
import torch.nn as nn


# inspired from cn  numberbatch
def read_men3000(filename):
    """
    Parses the MEN test collection. MEN is a collection of 3000 english word
    pairs, each with a relatedness rating between 0 and 50. The relatedness of
    a pair of words was determined by the number of times the pair was selected
    as more related compared to another randomly chosen pair.
    """
    with open(filename, 'r') as file:
        for line in file:
            parts = line.rstrip().split()
            term1 = parts[0].split('-')[0]  # remove part of speech
            term2 = parts[1].split('-')[0]
            gold_score = float(parts[2])
            yield term1, term2, gold_score


def cosine_similarity(word_vector_1, word_vector_2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(word_vector_1, word_vector_2)


def men_evaluation(filename, word_to_idx, embedding):
    from scipy.stats import spearmanr
    import numpy as np

    gold_scores, computed_scores = [], []
    for term_1, term_2, gold_score in read_men3000(filename):
        UNK_idx = word_to_idx["UNK"]
        idx_1, idx_2 = word_to_idx.get(term_1, UNK_idx), word_to_idx.get(term_2, UNK_idx)
        idx_1, idx_2 = torch.tensor(idx_1.index, device=torch.device("cuda")), \
                       torch.tensor(idx_2.index, device=torch.device("cuda"))
        vector_1, vector_2 = embedding(idx_1), embedding(idx_2)
        computed_score = cosine_similarity(vector_1, vector_2)
        gold_scores.append(gold_score)
        computed_scores.append(computed_score)
    return spearmanr(np.array(gold_scores), np.array(computed_scores))[0]



