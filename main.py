import torch
import torch.nn as nn
import torch.optim as optim
from poutyne.framework import Experiment

from models.embeddings_combination import CombinePreTrainedEmbs
from data_loader.utils import create_vocab, load_graph, prepare_generator_graph
from test_emb.redimensionality_learning import LearningVisualizer
import config


def main():
    entity_to_idx = create_vocab(config.entities_file)
    #relation_to_idx = create_vocab(config.)
    idx_to_entity = {v: k for k, v in entity_to_idx.items()}
    vocab_size = len(entity_to_idx.keys())

    #x = list(idx_to_entity.keys())
    #y = load_pretrained(config.pretrained_embs)
    #train_generator, valid_generator, test_generator = prepare_generator(x, y, vocab_size, config)

    x = load_graph(config.graphs[0], entity_to_idx)
    train_generator, valid_generator, test_generator = prepare_generator_graph(x, config.nb_false)

    device = torch.device('cuda:%d' % config.device if torch.cuda.is_available() else 'cpu')

    """
    network = CombinePreTrainedEmbs(vocab_size, **config.params_network)
    optimizer = optim.SGD(network.parameters(), **config.params_optimizer)
    criterion = nn.MSELoss()

    exp = Experiment(config.dir_experiment,
                     network,
                     device=device,
                     optimizer=optimizer,
                     loss_function=criterion,
                     batch_metrics=['mse'])

    exp.train(train_generator, valid_generator, epochs=config.epoch)
    exp.test(test_generator)

    learning_visualizer = LearningVisualizer(idx_to_entity, config.params_network['embedding_dim'], exp, config.epoch)
    learning_visualizer.visualize_learning()"""




if __name__ == '__main__':
    main()




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


