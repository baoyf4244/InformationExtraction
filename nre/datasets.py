import os
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
from tokenization import ChineseCharTokenizer, EnglishLabelTokenizer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningDataModule


class NREDataSet(Dataset):
    def __init__(self, sent_file, relations_file, tokenizer: EnglishLabelTokenizer,
                 char_tokenizer: ChineseCharTokenizer = None, triples_file=None):
        super(NREDataSet, self).__init__()
        self.sent_file = sent_file
        self.triples_file = triples_file
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

    def get_vocab_masks(self, token_ids):
        vocab_masks = [1] * self.tokenizer.get_vocab_size()
        for token_id in token_ids:
            vocab_masks[token_id] = 0

        vocab_masks[self.tokenizer.get_end_id()] = 0
        vocab_masks[self.tokenizer.get_ele_sep_id()] = 0
        vocab_masks[self.tokenizer.get_triple_sep_id()] = 0
        for relation in self.relations:
            vocab_masks[self.tokenizer.convert_token_to_id(relation)] = 0

        return vocab_masks

    def make_dataset(self):
        datasets = []
        with open(self.sent_file, encoding='utf-8') as f:
            sentences = f.readlines()

        with open(self.triples_file, encoding='utf-8') as f:
            triples = f.readlines()

        for sentence, triple in zip(sentences, triples):
            data = {}
            tokens = self.tokenizer.tokenize(sentence)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            data['tokens'] = tokens
            data['token_ids'] = token_ids
            data['masks'] = [0] * len(token_ids)
            data['vocab_masks'] = self.get_vocab_masks(token_ids)

            targets = self.tokenizer.tokenize(triple) + [self.tokenizer.get_end_token()]
            target_ids = self.tokenizer.convert_tokens_to_ids(targets)
            data['targets'] = targets
            data['target_ids'] = target_ids

            if self.char_tokenizer:
                chars = self.char_tokenizer.tokenize(sentence)
                char_ids = self.char_tokenizer.convert_tokens_to_ids(chars)
                data['char_ids'] = char_ids

            datasets.append(data)
        return datasets

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def collocate_fn(batch):
        src_max_seq_len = max([len(data['token_ids']) for data in batch])
        token_ids = [data['token_ids'] + [0] * (src_max_seq_len - len(data['token_ids'])) for data in batch]
        masks = [data['masks'] + [1] * (src_max_seq_len - len(data['masks'])) for data in batch]
        vocab_masks = [data['vocab_masks'] for data in batch]

        tgt_max_seq_len = max(len(data['target_ids']) for data in batch)
        target_ids = [data['target_ids'] + [0] * (tgt_max_seq_len - len(data['target_ids'])) for data in batch]
        tokens = [data['tokens'] for data in batch]
        targets = [data['targets'] for data in batch]

        return tokens, targets, torch.LongTensor(token_ids), torch.LongTensor(masks), \
               torch.LongTensor(vocab_masks), torch.LongTensor(target_ids)


class PTRNREDataSet(Dataset):
    def __init__(self, sent_file, relations_file, tokenizer: EnglishLabelTokenizer,
                 char_tokenizer: ChineseCharTokenizer = None, triples_file=None):
        super(PTRNREDataSet, self).__init__()
        self.sent_file = sent_file
        self.triples_file = triples_file
        self.relations_file = relations_file
        self.tokenizer = tokenizer
        self.char_tokenizer = char_tokenizer
        self.relations = self.get_relations()
        self.relation2idx = {relation: idx for idx, relation in enumerate(self.relations)}
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
            sentences = f.readlines()

        with open(self.triples_file, encoding='utf-8') as f:
            triples = f.readlines()

        for sentence, triple in zip(sentences, triples):
            data = {}
            tokens = self.tokenizer.tokenize(sentence)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            data['tokens'] = tokens
            data['token_ids'] = token_ids
            data['masks'] = [0] * len(token_ids)

            targets = triple.split('|')

            head_start_ids, head_end_ids, tail_start_ids, tail_end_ids, relation_ids = [], [], [], [], []
            for target in targets:
                elements = target.strip().split()
                head_start_ids.append(int(elements[0]))
                head_end_ids.append(int(elements[1]))
                tail_start_ids.append(int(elements[2]))
                tail_end_ids.append(int(elements[3]))
                relation_ids.append(self.relation2idx[elements[4]])

            data['head_start_ids'] = head_start_ids + [-1]
            data['head_end_ids'] = head_end_ids + [-1]
            data['tail_start_ids'] = tail_start_ids + [-1]
            data['tail_end_ids'] = tail_end_ids + [-1]
            data['relation_ids'] = relation_ids + [self.relation2idx['<SOS>']]
            data['target_masks'] = [0] * len(relation_ids)

            if self.char_tokenizer:
                chars = self.char_tokenizer.tokenize(sentence)
                char_ids = self.char_tokenizer.convert_tokens_to_ids(chars)
                data['char_ids'] = char_ids

            datasets.append(data)
        return datasets

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    @staticmethod
    def collocate_fn(batch):
        src_max_seq_len = max([len(data['token_ids']) for data in batch])
        token_ids = [data['token_ids'] + [0] * (src_max_seq_len - len(data['token_ids'])) for data in batch]
        masks = [data['masks'] + [1] * (src_max_seq_len - len(data['masks'])) for data in batch]

        tgt_max_seq_len = max(len(data['relation_ids']) for data in batch)
        head_start_ids = [data['head_start_ids'] + [0] * (tgt_max_seq_len - len(data['head_start_ids'])) for data in batch]
        head_end_ids = [data['head_end_ids'] + [0] * (tgt_max_seq_len - len(data['head_end_ids'])) for data in batch]
        tail_start_ids = [data['tail_start_ids'] + [0] * (tgt_max_seq_len - len(data['tail_start_ids'])) for data in batch]
        tail_end_ids = [data['tail_end_ids'] + [0] * (tgt_max_seq_len - len(data['tail_end_ids'])) for data in batch]
        target_ids = [data['relation_ids'] + [0] * (tgt_max_seq_len - len(data['relation_ids'])) for data in batch]
        target_masks = [data['target_masks'] + [1] * (tgt_max_seq_len - len(data['target_masks'])) for data in batch]
        # tokens = [data['tokens'] for data in batch]
        # targets = [data['targets'] for data in batch]

        return torch.LongTensor(token_ids), torch.LongTensor(masks), torch.LongTensor(head_start_ids), \
               torch.LongTensor(head_end_ids), torch.LongTensor(tail_start_ids), torch.LongTensor(tail_end_ids), \
               torch.LongTensor(target_ids), torch.LongTensor(target_masks)


class NREDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/nre',
                 max_len: int = 200,
                 batch_size: int = 16,
                 relation_file: str = 'data/nre/relations.txt',
                 vocab_file: str = 'data/nre/vocab.txt'):
        """
        Args:
            data_dir:
            max_len:
            batch_size:
            relation_file:
            vocab_file:
        """
        super(NREDataModule, self).__init__()
        self.relation_file = relation_file
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.relation_file = relation_file
        self.tokenizer = EnglishLabelTokenizer(vocab_file)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = PTRNREDataSet(os.path.join(self.data_dir, 'train.sent'),
                                            os.path.join(self.data_dir, 'relations.txt'),
                                            self.tokenizer,
                                            triples_file=os.path.join(self.data_dir, 'train.pointer'))
            if os.path.isfile(os.path.join(self.data_dir, 'dev.sent')):
                self.val_dataset = PTRNREDataSet(os.path.join(self.data_dir, 'dev.sent'),
                                              os.path.join(self.data_dir, 'relations.txt'),
                                              self.tokenizer,
                                              triples_file=os.path.join(self.data_dir, 'dev.pointer'))
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.val_dataset = PTRNREDataSet(os.path.join(self.data_dir, 'test.sent'),
                                          os.path.join(self.data_dir, 'relations.txt'),
                                          self.tokenizer,
                                          triples_file=os.path.join(self.data_dir, 'test.pointer'))

        # if stage == 'predict' or stage is None:
        #     self.predict_dataset = self.dataset_class(predict_file, tag_file, self.tokenizer, True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=PTRNREDataSet.collocate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=PTRNREDataSet.collocate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=PTRNREDataSet.collocate_fn)

    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.predict_dataset, self.batch_size, collate_fn=NREDataSet.collocate_fn)


if __name__ == '__main__':
    vocab_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/vocab.txt'
    data_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/dev.sent'
    relations = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/relations.txt'
    triples_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/dev.tup'
    tokenizer = EnglishLabelTokenizer(vocab_file, data_file, relations)
    dataset = NREDataSet(data_file, relations, tokenizer, triples_file=triples_file)
    print(dataset[0])
    print(NREDataSet.collocate_fn(dataset[: 10]))
