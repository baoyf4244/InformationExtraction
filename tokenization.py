import os


class CharTokenizer:
    def __int__(self, vocab_file, max_freq=None, min_freq=10):
        self.vocab_file = vocab_file

    def load_vocab(self):
        vocab = []
        with open(self.vocab_file, encoding='utf-8') as f:
            for line in f:
                vocab.append(line.strip())
        return vocab
