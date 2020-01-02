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


def prepare_generator(x, y, vocab_size, config):
    from torch.utils import data
    from data_loader.dataset import EmbeddingDataset

    xy = list(((idx, emb) for idx, emb in zip(x, y)))

    full_dataset = EmbeddingDataset(xy)

    train_size, valid_size = int(config.train_prop * vocab_size), int(config.valid_prop * vocab_size)
    test_size = vocab_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = data.random_split(full_dataset, [train_size, valid_size, test_size])

    train_generator = data.DataLoader(train_dataset, **config.params_dataset)
    valid_generator = data.DataLoader(valid_dataset, **config.params_dataset)
    test_generator = data.DataLoader(test_dataset, **config.params_dataset)

    return train_generator, valid_generator, test_generator


def get_one_hot(idx, vocab_size):
    import numpy as np

    one_hot = np.zeros(vocab_size)
    one_hot[idx] = 1
    return one_hot
