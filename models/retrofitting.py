import torch.nn as nn
import torch


class Retrofit(nn.Module):
    """
    Retrofitting from Faruqi
    Adapted for neural net
    """
    def __init__(self, ent_pre_trained):
        super(Retrofit, self).__init__()
        self.embedding_layer = ent_pre_trained

    def forward(self, x):
        head_embedding = self.embedding_layer[x['head']]
        tail_embedding = self.embedding_layer[x['tail']]
        distance = torch.norm(head_embedding - tail_embedding, 2)
        return distance

