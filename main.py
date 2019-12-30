import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
from poutyne.framework import Experiment, Model

from models.embeddings_combination import CombinePreTrainedEmbs
from data_loader.utils import create_vocab, load_pretrained
from data_loader.dataset import EmbeddingDataset
import config


# LOAD DATA
# 1 - get the vocab from pre-trained and aligned
entity_to_idx = create_vocab(config.entities_file)
idx_to_entity = {v: k for k, v in entity_to_idx.items()}
vocab_size = len(entity_to_idx.keys())

x = list(idx_to_entity.keys())
y = load_pretrained(config.pretrained_embs)
y_emb = nn.Embedding.from_pretrained(y)
xy = list(((idx, emb) for idx, emb in zip(x, y)))

full_dataset = EmbeddingDataset(xy)

train_size, valid_size = int(0.7*vocab_size), int(0.15*vocab_size)
test_size = vocab_size - train_size - valid_size

train_dataset, valid_dataset, test_dataset = data.random_split(full_dataset, [train_size, valid_size, test_size])

train_generator = data.DataLoader(train_dataset, **config.params_dataset)
valid_generator = data.DataLoader(valid_dataset, **config.params_dataset)
test_generator  = data.DataLoader(test_dataset, **config.params_dataset)

device = torch.device('cuda:%d' % config.device if torch.cuda.is_available() else 'cpu')

network = CombinePreTrainedEmbs(entity_to_idx, **config.params_network)
optimizer = optim.SGD(network.parameters(), **config.params_optimizer)
criterion = nn.MSELoss()


exp = Experiment('./experiment_3',
                 network,
                 device=device,
                 optimizer=optimizer,
                 loss_function=criterion,
                 batch_metrics=['mse'])

exp.train(train_generator, valid_generator, epochs=config.epoch)
exp.test(test_generator)



# ÉIl faut que je mélange �a de la mme mani�re entre train, valid, test
# I just give as input to my forward a word, that it convert into embedding <
# TODO write forward as  if x is just one word
# TODO transfoorm in with lookup and themn process
# input = Y_i initialisé au hasard
# self.fc1 = nn.Linear(300, 300 * number of models)
# W = self.fc1(input) -- relu?? sigmoid??
# loss_fn = nn.MSELoss(reduction='mean')
# loss = loss_fn(y, y_hat)
# mettre eta2 pour weight decay dans l'optimiser

#combine_net = pretrained_embeddings_combination()

# 3 get the KG
#    triples with entities converted in word embeddings and relations in relation embeddings
#    also need the scores
#   only keep edges with known entities
#   create embedding layer for relations, which has require_grad=True

# 4 Create batches of edges

# FOR EACH BATCH
# 5 Create n_neg for each edges by picking randomly a word embedding and put it in head or tail
# create a module where the forward takes positive edges and will generate the negatives randomly into the forward
# positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
# Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id
#         heads = triplets[:, 0]
#         relations = triplets[:, 1]
#         tails = triplets[:, 2]

# 6 calculate score for each pos and negative edges
#   score = ||y_u + q_r - y_v||
#   (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,
#                                                                                                           dim=1)
# 7 Calculate loss
#   L1 = weight edge * max (margin + score positive - score negative, 0)
#   target = torch.tensor([-1], dtype=torch.long, device=self.device)
#   return self.criterion(positive_distances, negative_distances, target)
#   read nice explaination https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7


