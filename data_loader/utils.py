def load_pretrained(embedding_files):

    from gensim.models import KeyedVectors
    import torch
    import torch.nn as nn

    vec_model = KeyedVectors.load_word2vec_format(embedding_files)
    weights = torch.FloatTensor(vec_model.vectors)
    embedding = nn.Embedding.from_pretrained(weights)

    return embedding


def load_graph(graph_file, vec_model, rel_to_idx=None, neg_sample=1, num_class=0):
    import torch
    triples = []
    vocab_size = len(vec_model.index2word)
    with open(graph_file, "r") as f:
        for triple in f.readlines():
            triple = triple.strip().split('  ,  ')
            if len(triple) == 3:
                head_idx = vec_model.vocab.get(triple[0], None)
                tail_idx = vec_model.vocab.get(triple[2], None)
            else:
                continue
            if head_idx and tail_idx:
                # positive has class 0 and negative class 1
                triple = ({'head': head_idx.index, 'rel': torch.tensor(0).to("cuda"), 'tail': tail_idx.index}, torch.tensor(num_class)) # 1 car positive
                triples.append(triple)
                for neg in range(neg_sample):
                    triples.append((generate_negative_sample(triple, vocab_size), torch.tensor(1)))
    return triples


def load_anto_syn_graph(syn_file, anto_file, vec_model, rel_to_idx=None, neg_sample=1):
    antonyms = load_graph(anto_file, vec_model, rel_to_idx=None, neg_sample=0, num_class=1)
    synonyms = load_graph(syn_file, vec_model, rel_to_idx=None, neg_sample=0, num_class=0)
    antosyn = antonyms+synonyms
    return antosyn


def compute_weight(class_repartition):
    import torch
    baseline = float(class_repartition[0])
    weights = [baseline/nb for nb in class_repartition]
    return torch.tensor(weights)


def generate_negative_sample(true_fact, vocab_size):
    import torch
    import random
    head_or_tail = torch.tensor(random.getrandbits(1)).to("cuda")
    rand_word_idx = torch.tensor(random.randint(0, vocab_size-1), dtype=torch.long).to("cuda")
    if head_or_tail:
        return {'head': rand_word_idx, 'rel': true_fact[0]['rel'], 'tail': true_fact[0]['tail']}
    return {'head': true_fact[0]['head'], 'rel': true_fact[0]['rel'], 'tail': rand_word_idx}


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


def prepare_generator_graph(x):
    from torch.utils import data
    from data_loader.triples_dataset import TriplesDataset
    import config

    dataset = TriplesDataset(x)
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
