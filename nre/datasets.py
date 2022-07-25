from torch.utils.data import Dataset, DataLoader
from tokenization import CharTokenizer, WhiteSpaceTokenizer


class NREDataSet(Dataset):
    def __init__(self, sent_file, relations_file, tokenizer: WhiteSpaceTokenizer,
                 char_tokenizer: CharTokenizer = None, tuples_file=None):
        super(NREDataSet, self).__init__()
        self.sent_file = sent_file
        self.tuples_file = tuples_file
        self.relations_file = relations_file
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.relations = self.get_relations()
        self.datasets = self.make_dataset()

    def get_relations(self):
        relations = []
        with open(self.relations_file, encoding='utf-8') as f:
            for line in f:
                relations.append(line.strip())
        return relations

    def make_dataset(self):
        datasets = []
        with open(self.sent_file, encoding='utf-8') as f:
            for line in f:
                data = {}
                tokens = self.tokenizer.tokenize(line)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                data['token_ids'] = token_ids
                data['masks'] = [0] * len(token_ids)

                vocab_masks = [1] * len(self.tokenizer.get_vocab_size())
                for token_id in token_ids:
                    vocab_masks[token_id] = 0

                vocab_masks[self.tokenizer.get_eos_id()] = 0
                for _ in [';', '|']:
                    vocab_masks.append(0)

                data['vocab_masks'] = vocab_masks
                if self.char_tokenizer:
                    chars = self.char_tokenizer.tokenize(line)
                    char_ids = self.char_tokenizer.convert_tokens_to_ids(chars)
                    data['char_ids'] = char_ids

            datasets.append(char_ids)
        return datasets

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def collocate_fn(batch):
        max_seq_len = max([len(data['token_ids']) for data in batch])
        token_ids = [data['token_ids'] + [0] * (max_seq_len - len(data['token_ids'])) for data in batch]
        masks = [data['masks'] + [1] * (max_seq_len - len(data['masks'])) for data in batch]
        vocab_masks = [data['vocab_masks'] for data in batch]

