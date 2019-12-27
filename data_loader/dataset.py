from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        idx, emb = self.dataset[index]
        return idx, emb
