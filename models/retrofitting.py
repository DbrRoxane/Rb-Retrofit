import torch.nn as nn
import torch


class Retrofit(nn.Module):
    """
    Retrofitting from Faruqi
    Adapted for neural net
    """
    def __init__(self, dim, size, ent_pre_trained, score_func):
        self.entity_layer = nn.Embedding(dim, size)
        self.entity_layer.from_pretrained(ent_pre_trained)
        self.score_func = score_func

    def forward(self, x):
        head_embedding = self.entity_layer(torch.tensor(x['head']))
        tail_embedding = self.entity_layer(torch.tensor(x['tail']))
        distance = torch.norm(head_embedding - tail_embedding, 2)
        return distance

