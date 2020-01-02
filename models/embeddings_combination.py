import torch.nn as nn


class CombinePreTrainedEmbs(nn.Module):
    def __init__(self, vocab_size, embedding_dim, number_models):
        super(CombinePreTrainedEmbs, self).__init__()
        self.encode = nn.Linear(vocab_size, embedding_dim)
        self.decode = nn.Linear(embedding_dim, number_models*embedding_dim)

    def forward(self, x):
        encoded = self.encode(x)
        original_refined = self.decode(encoded)
        return original_refined
