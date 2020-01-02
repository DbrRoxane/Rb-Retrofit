from torch.utils.data import Dataset
import torch
from .utils import get_one_hot

class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.vocab_size = self.__len__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        idx, emb = self.dataset[index]
        one_hot = torch.tensor(get_one_hot(idx, self.vocab_size)).float()
        return one_hot, emb


