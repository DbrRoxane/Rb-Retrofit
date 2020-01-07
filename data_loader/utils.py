def create_vocab(vocab_file):
    with open(vocab_file, "r") as f:
        vocab = set(f.read().splitlines())
    word_to_idx = {word: i+2 for i, word in enumerate(sorted(vocab))}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx


def load_pretrained(embedding_files):

    from gensim.models import KeyedVectors
    import torch
    import torch.nn as nn

    vec_model = KeyedVectors.load_word2vec_format(embedding_files)
    weights = torch.FloatTensor(vec_model.vectors)
    embedding = nn.Embedding.from_pretrained(weights)

    return embedding


def load_graph(graph_file, word_to_idx, rel_to_idx=None):
    triples = []
    cnt_all, cnt_ok = 0, 0
    with open(graph_file, "r") as f:
        for triple in f.readlines():
            cnt_all += 1
            triple = triple.strip().split('  ,  ')
            if len(triple) == 3:
                head_word, rel_word, tail_word = triple
            else:
                continue
            head_idx, tail_idx = word_to_idx.get(head_word, False), word_to_idx.get(tail_word, False) # 1 is UNK
            if head_idx and tail_idx:
                #rel_idx = rel_to_idx.get(rel_word, 1)
                cnt_ok += 1
                triples.append({'head': head_idx, 'rel': None, 'tail': tail_idx})
    print(cnt_all, cnt_ok)
    return triples


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


def prepare_generator_graph(x, nb_false):
    from torch.utils import data
    from data_loader.dataset import TriplesDataset
    import config

    dataset = TriplesDataset(x, nb_false)
    train_size, valid_size = int(config.train_prop * len(x)), int(config.valid_prop * len(x))
    test_size = len(x) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = data.random_split(dataset, [train_size, valid_size, test_size])

    train_generator = data.DataLoader(train_dataset, **config.params_dataset)
    valid_generator = data.DataLoader(valid_dataset, **config.params_dataset)
    test_generator = data.DataLoader(test_dataset, **config.params_dataset)

    return train_generator, valid_generator, test_generator


def get_one_hot(idx, vocab_size):
    import numpy as np
    import torch

    one_hot = np.zeros(vocab_size)
    one_hot[idx] = 1
    return torch.tensor(one_hot).float()
