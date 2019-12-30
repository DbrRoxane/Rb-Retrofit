import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
from poutyne.framework import Experiment, Model

from models.embeddings_combination import CombinePreTrainedEmbs
from data_loader.utils import create_vocab
from data_loader.dataset import EmbeddingDataset


# INPUT = pre-trained embedding models AND KG
# OUTPUT = word embedding updated and relation embeddings

pretrained_embs = ['./data/wgc_w_zip.txt', './data/wgc_g_zip.txt']
# graphs triplets [e,r,h]  score
graphs = ['./data/cnet_graph_score.txt', './data/wordnet_graph_score.txt', './data/ppdb_graph_score.txt']
# liste des mots
entities_file = './data/vocab_Ins_wgc.txt'
# listes de relation
relations_file = ['./data/cnetrellist.txt', './data/wordnetrellist.txt', './data/ppdbrellist.txt']


# LOAD DATA
# 1 - get the vocab from pre-trained and aligned
entity_to_idx = create_vocab(entities_file)
idx_to_entity = {v: k for k, v in entity_to_idx.items()}
vocab_size = len(entity_to_idx.keys())

x = list(idx_to_entity.keys())

# 2 - get the pre-trained vectors
y = []
for vec_model_path in pretrained_embs:
    with open(vec_model_path, "r") as file:
        weight = np.array([line.strip().split()[1:] for line in file.readlines()]).astype('float64')
        weight = np.insert(weight, 0, np.random.rand(300), axis=0) # <PAD>
        weight = np.insert(weight, 1, np.random.rand(300), axis=0) # <UNK>
        weight = torch.FloatTensor(weight)
        y.append(weight)

# 3 - concat the vectors from models
y = torch.cat(y, dim=1)

# 4 - store it in an embedding layer
y_emb = nn.Embedding.from_pretrained(y)

# 5 - Prepare the data

xy = list(((idx, emb) for idx, emb in zip(x, y)))

full_dataset = EmbeddingDataset(xy)

train_size, valid_size = int(0.7*vocab_size), int(0.15*vocab_size)
test_size = vocab_size - train_size - valid_size

train_dataset, valid_dataset, test_dataset = data.random_split(full_dataset, [train_size, valid_size, test_size])

params = {'batch_size': 1024,
          'shuffle': True,
          'num_workers': 0}

train_generator = data.DataLoader(train_dataset, **params)
valid_generator = data.DataLoader(valid_dataset, **params)
test_generator  = data.DataLoader(test_dataset,  **params)

# 6 - Model parameter

device = 0
device = torch.device('cuda:%d' % device if torch.cuda.is_available() else 'cpu')

network = CombinePreTrainedEmbs(entity_to_idx, embedding_dim=300, number_models=len(pretrained_embs))
optimizer = optim.SGD(network.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)
criterion = nn.MSELoss()

epoch = 45

exp = Experiment('./experiment_3',
                 network,
                 device=device,
                 optimizer=optimizer,
                 loss_function=criterion,
                 batch_metrics=['mse'])

exp.train(train_generator, valid_generator, epochs=epoch)
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


