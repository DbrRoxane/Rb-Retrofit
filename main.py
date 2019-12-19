
from gensim.models import KeyedVectors
from models import pretrained_embeddings_combination
from data_loader.utils import  create_vocab


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
# 1 get the pre-trained word embedding
entity_to_idx = create_vocab(entities_file)
rel_to_idx = create_vocab(relations_file)

en

vec_models = []
for vec_model_path in pretrained_embs:
    vec_models.append(KeyedVectors.load_word2vec_format(vec_model_path))

# 2 PRE-TRAINED WORD EMBEDDING ACCUMULATION
# input = Y_i initialisé au hasard
# self.fc1 = nn.Linear(300, 300 * number of models)
# W = self.fc1(input) -- relu?? sigmoid??
# loss_fn = nn.MSELoss(reduction='mean')
# loss = loss_fn(y, y_hat)
# mettre eta2 pour weight decay dans l'optimiser
combine_net = pretrained_embeddings_combination()


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


