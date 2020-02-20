import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from poutyne.framework.callbacks.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
from gensim.models import KeyedVectors
from poutyne.framework import Experiment

from models.retrofitting import Retrofit
from data_loader.utils import load_anto_syn_graph, prepare_generator_graph, compute_weight
import config
from evaluation.similarity import men_evaluation
from evaluation.test_emb.redimensionality_learning import LearningVisualizer


def lambda_lr_embedding(current_epoch):
    if current_epoch <= 4:
        return 0
    elif current_epoch <= 10:
        return 1e-2
    elif 4 < current_epoch <= 20:
        return 3e-3
    else:
        return 1e-3


def lambda_lr_other(current_epoch):
    if current_epoch <= 4:
        return 1
    elif 4 < current_epoch <= 10:
        return 1e-1
    elif current_epoch <= 20:
        return 1e-2
    else:
        return 1e-3


def main():
    vec_model = KeyedVectors.load_word2vec_format(config.pretrained_embs[0], limit=500000)
    print("Breakpoint 1")

    x = load_anto_syn_graph(config.synonyms_graph[0], config.antonyms_graph[0],
                            vec_model, neg_sample=config.nb_false)

    weight = compute_weight(x)

    print("Breakpoint 2")
    train_generator, valid_generator, test_generator = prepare_generator_graph(x)
    print("Breakpoint 3")
    device = torch.device('cuda:%d' % config.device if torch.cuda.is_available() else 'cpu')

    network = Retrofit(vec_model, weight)

    embeddings_param_set = set(network.embedding.parameters())
    other_params_list = [p for p in network.parameters() if p not in embeddings_param_set]
    optimizer = optim.SGD([{'params': other_params_list, **config.optimizer_other_params},
                           {'params': network.embedding.parameters(), **config.optimizer_embeddings_params}])

    #scheduler = LambdaLR(lr_lambda=[lambda_lr_other, lambda_lr_embedding])
    #scheduler = StepLR(step_size=8, gamma=0.1)
    scheduler = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, verbose=True)
    callbacks = [scheduler]

    exp = Experiment(config.dir_experiment,
                     network,
                     device=device,
                     optimizer=optimizer,
                     loss_function=None,
                     batch_metrics=['bin_acc']
                )

    exp.train(train_generator, valid_generator, epochs=config.epoch, lr_schedulers=callbacks)
    exp.test(test_generator)

    steps = len(test_generator)
    test_loss, test_metrics, pred_y, true_y = exp.model.evaluate_generator(test_generator,
                                                                           return_pred=True,
                                                                           return_ground_truth=True,
                                                                           steps=steps)

    pred_y = np.argmax(np.concatenate(pred_y), 1)
    true_y = np.concatenate(true_y)
    true_syn, false_syn, false_anto, true_anto = confusion_matrix(true_y, pred_y).ravel()
    print(true_syn, false_syn, false_anto, true_anto)

    learning_visualizer = LearningVisualizer(exp, config.epoch)
    learning_visualizer.visualize_learning()

    exp._load_best_checkpoint()
    exp.model.model.embedding.weight.requires_grad = False

    print(men_evaluation('./data/evaluation/MEN/MEN_dataset_lemma_form.test',
                         vec_model.vocab,
                         exp.model.model.embedding))

    vec_model_initial = KeyedVectors.load_word2vec_format(config.pretrained_embs[0], limit=500000)
    original_weights = torch.FloatTensor(vec_model_initial.vectors)
    original_weights.to("cuda")
    original_embs = nn.Embedding.from_pretrained(original_weights)
    original_embs.cuda()
    original_embs.weight.requires_grad = False

    print(men_evaluation('./data/evaluation/MEN/MEN_dataset_lemma_form.test', vec_model.vocab, original_embs))


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


