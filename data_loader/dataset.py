from torch.utils.data import Dataset
import torch
import numpy as np


class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        idx, emb = self.dataset[index]
        one_hot = torch.tensor(self.get_one_hot(idx)).float()
        return one_hot, emb

    def get_one_hot(self, idx):
        one_hot = np.zeros(self.vocab_size)
        one_hot[idx] = 1
        return one_hot
