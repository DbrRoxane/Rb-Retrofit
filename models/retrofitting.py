import torch.nn as nn
import torch


class Retrofit(nn.Module):
    def __init__(self, vec_model, word_to_idx, weight):
        super(Retrofit, self).__init__()
        self.word_to_idx = word_to_idx
        weights = torch.FloatTensor(vec_model.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights, max_norm=2)
        self.embedding.weight.requires_grad = True
        self.fc1 = nn.Linear(self.embedding.embedding_dim*2, self.embedding.embedding_dim)
        self.fc2 = nn.Linear(self.embedding.embedding_dim, 2)
        self.loss_function = nn.CrossEntropyLoss(weight=weight)

    def forward(self, x):
        head_embedding = self.embedding(x['head'].cuda())
        tail_embedding = self.embedding(x['tail'].cuda())
        embedded = torch.cat((head_embedding, tail_embedding), 1)
        hidden = nn.functional.tanh(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

