from torch import nn


class EmbeddingLayer(nn.Embedding):
    def __init__(self, word_to_idx, embedding_dim):
        super(MyEmbeddings, self).__init__(len(word_to_idx), embedding_dim, padding_idx=0)
        self.embedding_dim = embedding_dim
        self.vocab_size = len(word_to_idx)
        self.word_to_idx = word_to_idx

    def set_item_embedding(self, idx, embedding):
        self.weight.data[idx] = torch.FloatTensor(embedding)

    def load_words_embeddings(self, vec_model):
        for word in vec_model.index2word:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                embedding = vec_model[word]
                self.set_item_embedding(idx, embedding)