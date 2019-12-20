def create_vocab(vocab_file):
    with open(vocab_file, "r") as f:
        vocab = set(f.read().splitlines())
    word_to_idx = {word: i+2 for i, word in enumerate(sorted(vocab))}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1
    return word_to_idx
