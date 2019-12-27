import numpy as np
import torch
import torch.nn as nn

class CombinePreTrainedEmbs(nn.Module):
    def __init__(self, word_to_idx, embedding_dim, number_models):
        super(CombinePreTrainedEmbs, self).__init__()
        self.combined = nn.Embedding(len(word_to_idx), embedding_dim, padding_idx=0)
        self.combined.weight.data.uniform_(-2, 2)
        self.fc = nn.Linear(embedding_dim, number_models*embedding_dim)

    def forward(self, x):
        #lookup_tensor = torch.tensor(x, dtype=torch.long)
        combined = self.combined(x)
        original_refined = self.fc(combined)
        return original_refined
