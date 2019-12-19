def create_vocab(vocab_file):
    with open(vocab_file, "r") as f:
        vocab = set(f.read().splitlines())
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word in sorted(vocab):
        word_to_idx[word] = len(word_to_idx)
    return word_to_idx
