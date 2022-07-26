import os
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
from tokenization import CharTokenizer, WhiteSpaceTokenizer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningDataModule


class NREDataSet(Dataset):
    def __init__(self, sent_file, relations_file, tokenizer: WhiteSpaceTokenizer,
                 char_tokenizer: CharTokenizer = None, triples_file=None):
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

        vocab_masks[self.tokenizer.get_eos_id()] = 0
        vocab_masks[self.tokenizer.get_ele_sep_id()] = 0
        vocab_masks[self.tokenizer.get_triple_sep_id()] = 0

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

            targets = self.tokenizer.tokenize(triple)
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
        self.tokenizer = WhiteSpaceTokenizer(vocab_file)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = NREDataSet(os.path.join(self.data_dir, 'train.sent'),
                                            os.path.join(self.data_dir, 'relations.txt'),
                                            self.tokenizer,
                                            triples_file=os.path.join(self.data_dir, 'train.tup'))
            if os.path.isfile(os.path.join(self.data_dir, 'dev.sent')):
                self.val_dataset = NREDataSet(os.path.join(self.data_dir, 'dev.sent'),
                                              os.path.join(self.data_dir, 'relations.txt'),
                                              self.tokenizer,
                                              triples_file=os.path.join(self.data_dir, 'dev.tup'))
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.val_dataset = NREDataSet(os.path.join(self.data_dir, 'test.sent'),
                                          os.path.join(self.data_dir, 'relations.txt'),
                                          self.tokenizer,
                                          triples_file=os.path.join(self.data_dir, 'test.tup'))

        # if stage == 'predict' or stage is None:
        #     self.predict_dataset = self.dataset_class(predict_file, tag_file, self.tokenizer, True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=NREDataSet.collocate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=NREDataSet.collocate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=NREDataSet.collocate_fn)

    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.predict_dataset, self.batch_size, collate_fn=NREDataSet.collocate_fn)


if __name__ == '__main__':
    vocab_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/vocab.txt'
    data_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/dev.sent'
    relations = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/relations.txt'
    triples_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/dev.tup'
    tokenizer = WhiteSpaceTokenizer(vocab_file, data_file, relations)
    dataset = NREDataSet(data_file, relations, tokenizer, triples_file=triples_file)
    print(dataset[0])
    print(NREDataSet.collocate_fn(dataset[: 10]))
