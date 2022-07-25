import sys
from collections import Counter


class WhiteSpaceTokenizer:
    def __int__(self, vocab_file, data_file=None, max_freq=None, min_freq=10):
        self.vocab_file = vocab_file
        self.data_file = data_file
        self.max_freq = sys.maxsize if max_freq is None else max_freq
        self.min_freq = min_freq
        self.vocab = self.load_vocab()
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

    def load_vocab(self):
        counter = Counter()
        vocab = []
        try:
            with open(self.vocab_file, encoding='utf-8') as f:
                for line in f:
                    vocab.append(line.strip())
        except Exception:
            with open(self.data_file, encoding='utf-8') as f:
                for line in f:
                    counter.update(self.tokenize(line))

            vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
            for word, count in counter.items():
                if self.min_freq < count < self.max_freq:
                    vocab.append(word)

            with open(self.vocab_file, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(vocab))

        return vocab

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            token_ids.append(self.word2idx[token] if token in self.word2idx else self.word2idx['<UNK>'])
        return token_ids

    def get_pad_id(self):
        return self.word2idx['<PAD>']

    def get_unk_id(self):
        return self.word2idx['<UNK>']

    def get_eos_id(self):
        return self.word2idx['<EOS>']

    def get_vocab_size(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        return text.strip().split()


class CharTokenizer:
    def __int__(self, vocab_file, data_file=None, max_freq=None, min_freq=0):
        self.vocab_file = vocab_file
        self.data_file = data_file
        self.max_freq = sys.maxsize if max_freq is None else max_freq
        self.min_freq = min_freq
        self.vocab = self.load_vocab()
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}

    def load_vocab(self):
        counter = Counter()
        vocab = []
        try:
            with open(self.vocab_file, encoding='utf-8') as f:
                for line in f:
                    vocab.append(line.strip())
        except Exception:
            with open(self.data_file, encoding='utf-8') as f:
                for line in f:
                    counter.update(self.tokenize(line))

            vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
            for word, count in counter.items():
                if self.min_freq < count < self.max_freq:
                    vocab.append(word)

            with open(self.vocab_file, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(vocab))

        return vocab

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            token_ids.append(self.char2idx[token] if token in self.char2idx else self.char2idx['<UNK>'])
        return token_ids

    @staticmethod
    def tokenize(text):
        chars = []
        tokens = text.strip().split()
        for token in tokens:
            chars.extend(token.strip())

        return chars
