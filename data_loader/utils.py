def create_vocab(vocab_file):
    with open(vocab_file, "r") as f:
        vocab = set(f.read().splitlines())
    word_to_idx = {word: i+2 for i, word in enumerate(sorted(vocab))}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx


def load_pretrained(embedding_files):
    import numpy as np
    import torch
    y = []
    for vec_model_path in embedding_files:
        with open(vec_model_path, "r") as file:
            weight = np.array([line.strip().split()[1:] for line in file.readlines()]).astype('float64')
            weight = np.insert(weight, 0, np.random.rand(300), axis=0)  # <PAD>
            weight = np.insert(weight, 1, np.random.rand(300), axis=0)  # <UNK>
            weight = torch.FloatTensor(weight)
            y.append(weight)
    y = torch.cat(y, dim=1)
    return y
