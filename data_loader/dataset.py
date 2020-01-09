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
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        triple, label = self.dataset[index]
        return triple, label
