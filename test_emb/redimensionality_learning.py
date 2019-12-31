from poutyne.framework import Experiment
import matplotlib.pyplot as plt
import torch
import numpy as np

def display_emb_evolution(idx_to_word, dim_embedding, experiment, last_epoch):
    # piger 5 traits random entre 0 et 299
    selected_features = np.random.randint(low=0, high=dim_embedding, size=5)
    selected_word = torch.tensor(np.random.choice(list(idx_to_word.keys()), 1))
    feature_evolution = {feature: [] for feature in selected_features}

    x = []

    for epoch in range(last_epoch):
        try :
            experiment.load_checkpoint(epoch)
            x.append(epoch)
        except:
            continue
        for feature in selected_features:
            feature_evolution[feature].append(float(experiment.model.model.combined(selected_word)[0][feature]))

    fig, ax = plt.subplots()
    for k, v, in feature_evolution.items():
        ax.plot(x, v)
    plt.show()
    return None