import json
import sys
from collections import Counter


class PTRTokenizer:
    def __init__(self, vocab_file, data_file=None, max_freq=None, min_freq=1):
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.start_token = '[CLS]'
        self.end_token = '[SEP]'
        self.vocab_file = vocab_file
        self.data_file = data_file
        self.max_freq = sys.maxsize if max_freq is None else max_freq
        self.min_freq = min_freq
        self.vocab = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.word2idx = None
        self.init()

    def init(self):
        self.load_vocab()
        self.add_special_tokens()
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

    def add_special_tokens(self):
        pass

    def load_vocab(self):
        try:
            with open(self.vocab_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip() not in self.vocab[: 4]:
                        self.vocab.append(line.strip())
        except Exception:
            assert self.data_file is not None, 'vocab_file 和 data_file 必须有一个可用'
            counter = Counter()
            with open(self.data_file, encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    tokens = self.tokenize(line['text'])
                    for token in tokens:
                        if isinstance(token, list):
                            counter.update(token)
                        else:
                            counter[token] += 1

            for word, count in counter.items():
                if self.min_freq < count < self.max_freq and word not in self.vocab[: 4]:
                    self.vocab.append(word)

            with open(self.vocab_file, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(self.vocab))

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            token_ids.append(self.convert_token_to_id(token))
        return token_ids

    def convert_token_to_id(self, token):
        return self.word2idx[token] if token in self.word2idx else self.word2idx[self.unk_token]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[idx] for idx in ids]

    def convert_id_to_token(self, idx):
        return self.vocab[idx]

    def get_pad_id(self):
        return self.word2idx[self.pad_token]

    def get_pad_token(self):
        return self.pad_token

    def get_unk_id(self):
        return self.word2idx[self.unk_token]

    def get_start_id(self):
        return self.word2idx[self.start_token]

    def get_start_token(self):
        return self.start_token

    def get_end_id(self):
        return self.word2idx[self.end_token]

    def get_end_token(self):
        return self.end_token

    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def get_word2idx(self):
        return self.word2idx

    @staticmethod
    def tokenize(text):
        return text.strip().split()


if __name__ == '__main__':
    PTRTokenizer('data/nre/vocab.txt', 'data/nre/train.txt')