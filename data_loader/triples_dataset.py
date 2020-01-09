from torch.utils.data import Dataset


class TriplesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        triple, label = self.dataset[index]
        return triple, label
