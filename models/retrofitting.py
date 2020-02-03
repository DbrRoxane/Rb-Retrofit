import torch.nn as nn
import torch
from data_loader.utils import set_word_to_idx


class Retrofit(nn.Module):
    """
    Retrofitting from Faruqi
    Adapted for neural net
    """
    def __init__(self, vec_model):
        super(Retrofit, self).__init__()
        self.word_to_idx = set_word_to_idx(vec_model)
        weights = torch.FloatTensor(vec_model.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.embedding.weight.requires_grad = True
        self.fc = nn.Linear(self.embedding.embedding_dim*2, 2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        head_embedding = self.embedding(x['head'].cuda())
        tail_embedding = self.embedding(x['tail'].cuda())
        embeddings = torch.cat((head_embedding, tail_embedding), 1)
        output = self.fc(embeddings)
        return output

