def load_vocab(vocab_file):
    with open(vocab_file, encoding='utf-8') as f:
        vocab = f.readlines()

    return [v.strip() for v in vocab if v.strip()]


