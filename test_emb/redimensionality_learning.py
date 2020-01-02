import matplotlib.pyplot as plt
import torch
import numpy as np
from data_loader.utils import get_one_hot


class LearningVisualizer():
    def __init__(self, idx_to_word, dim_embedding, experiment, last_epoch):
        self.idx_to_word= idx_to_word
        self.dim_embedding = dim_embedding
        self.experiment = experiment
        self.select_random_features()
        self.selected_word = self.select_random_word()
        self.selected_features = self.select_random_features()
        self.feature_evolution = {feature: [] for feature in self.selected_features}
        self.recorded_epoch = self.get_recorded_epoch(last_epoch)

    def select_random_word(self, nb_words=1):
        idx = torch.tensor(np.random.choice(list(self.idx_to_word.keys()), nb_words)[0])
        one_hot = torch.tensor(get_one_hot(idx, len(self.idx_to_word))).float()
        return one_hot

    def select_random_features(self, nb_features=5):
        return np.random.randint(low=0, high=self.dim_embedding, size=nb_features)

    def generate_data_evolution(self):
        for epoch in self.recorded_epoch:
            self.experiment.load_checkpoint(epoch)
            for feature in self.selected_features:
                self.feature_evolution[feature].append(
                    float(self.experiment.model.model.encode(self.selected_word)[feature]))

    def get_recorded_epoch(self, last_epoch):
        recorded_epoch = []
        for epoch in range(last_epoch):
            try:
                self.experiment.load_checkpoint(epoch)
                recorded_epoch.append(epoch)
            except ValueError:
                continue
        return recorded_epoch

    def visualize_learning(self):
        self.generate_data_evolution()

        fig, ax = plt.subplots()
        for k, v, in self.feature_evolution.items():
            ax.plot(self.recorded_epoch, v)

        ax.set_title('Word {0} for features {1}'.format(self.selected_word, self.selected_features))
        ax.set_xlabel('evolution through epochs')
        ax.set_ylabel('feature value')

        plt.show()
