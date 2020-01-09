import torch.nn as nn
import torch


class RbRetrofit(nn.Module):
    def __init__(self, dim, size, entity_weights, rel_weights, score_func):
        self.entity_layer = nn.Embedding(dim, size)
        self.entity_layer.from_pretrained(entity_weights)
        self.rel_layer = nn.Embedding(dim, size)
        #self.rel_layer.from_pretrained(entity_weights)
        self.score_func = score_func

    def forward(self, x):
        for triple in x:
            head_embedding = self.entity_layer(torch.tensor(triple['head'], dtype=torch.long))
            rel_embedding = self.rel_layer(torch.tensor(triple['rel'], dtype=torch.long))
            tail_embedding = self.entity_layer(torch.tensor(triple['tail'], dtype=torch.long))
            self.score_func(head_embedding, rel_embedding, tail_embedding)
        # return score
        pass

    def transE_score(self, head_embedding, rel_embedding, tail_embedding):
        return torch.norm(head_embedding + rel_embedding - tail_embedding, 2)

    def no_rel_score(self, head_embedding, rel_embedding, tail_embedding):
        return torch.norm(head_embedding - tail_embedding, 2)

