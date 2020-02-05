import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import KeyedVectors
from poutyne.framework import Experiment, StepLR

from models.retrofitting import Retrofit
from data_loader.utils import load_graph, prepare_generator_graph, set_word_to_idx
import config
from evaluation.similarity import men_evaluation
from evaluation.test_emb.redimensionality_learning import LearningVisualizer


def main():
    vec_model = KeyedVectors.load_word2vec_format(config.pretrained_embs[0], limit=500000)

    vec_model_initial = KeyedVectors.load_word2vec_format(config.pretrained_embs[0], limit=500000)
    original_weights = torch.FloatTensor(vec_model_initial.vectors)
    original_weights.to("cuda")
    original_embs = nn.Embedding.from_pretrained(original_weights)
    original_embs.cuda()
    original_embs.weight.requires_grad = False

    word_to_idx = set_word_to_idx(vec_model)
    print("Breakpoint 1")

    x = load_graph(config.graphs[0], vec_model, vec_model_initial, word_to_idx)

    print("Breakpoint 2")
    train_generator, valid_generator, test_generator = prepare_generator_graph(x)

    print("Breakpoint 3")
    device = torch.device('cuda:%d' % config.device if torch.cuda.is_available() else 'cpu')

    network = Retrofit(vec_model, word_to_idx)
    optimizer = optim.Adam(network.parameters(), **config.params_optimizer)
    #scheduler = StepLR(step_size=1, gamma=0.3)
    #callbacks = [scheduler]

    exp = Experiment(config.dir_experiment,
                     network,
                     device=device,
                     optimizer=optimizer,
                     loss_function=None,
                     batch_metrics=['acc']
                     )

    exp.train(train_generator, valid_generator, epochs=config.epoch) #, lr_schedulers=callbacks)
    exp.test(test_generator)

    learning_visualizer = LearningVisualizer(exp, config.epoch)
    learning_visualizer.visualize_learning()

    exp._load_best_checkpoint()
    exp.model.model.embedding.weight.requires_grad = False

    print(men_evaluation('./data/evaluation/MEN/MEN_dataset_lemma_form.test',
                         word_to_idx,
                         exp.model.model.embedding))



    print(men_evaluation('./data/evaluation/MEN/MEN_dataset_lemma_form.test', word_to_idx, original_embs))


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


