from torch.utils.data import Dataset
import torch
import random
from .utils import get_one_hot


class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        idx, emb = self.dataset[index]
        one_hot = get_one_hot(idx, self.__len__())
        return one_hot, emb


class TriplesDataset(Dataset):
    def __init__(self, dataset, nb_false):
        self.dataset = dataset
        self.nb_false = nb_false

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        head, rel, tail = self.dataset[index]
        true = {'head': head, 'rel': rel, 'tail': tail}
        falses = self.generate_negative_facts(true)
        return true, falses

    def generate_negative_facts(self, true_fact):
        falses = []
        for _ in range(self.nb_false):
            head_or_tail = random.getrandbits(1)
            word_idx = random.randint(0, self.__len__())
            falses.append({'head': word_idx, 'rel': true_fact['rel'], 'tail':true_fact['tail']}) \
                if head_or_tail \
                else falses.append({'head': true_fact['head'], 'rel': true_fact['rel'], 'tail':word_idx})
        return falses
