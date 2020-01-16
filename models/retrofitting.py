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
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        head_embedding = self.embedding(x['head'])
        tail_embedding = self.embedding(x['tail'])
        return head_embedding, tail_embedding

    def loss_function(self, prediction, target):
        head_pred, tail_pred = prediction
        head_target, tail_target = target(
            ux a
        )
        distance_pairs = 1 - self.cos(head_pred, tail_pred)
        distance_initial_head = 1 - self.cos(head_pred, head_target)
        distance_initial_tail = 1 - self.cos(tail_pred, tail_target)
        return distance_pairs + distance_initial_head/2. + distance_initial_tail/2.
