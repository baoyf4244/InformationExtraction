import json
import os
import torch
from typing import Optional
from tokenization_ptr import PTRTokenizer, WDTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningDataModule


class NREDataSet(Dataset):
    def __init__(self, data_file, relations_file, tokenizer: WDTokenizer):
        super(NREDataSet, self).__init__()
        self.data_file = data_file
        self.relations_file = relations_file
        self.tokenizer = tokenizer
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

        with open(self.data_file, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                data = {}
                tokens = self.tokenizer.tokenize(line['text'])
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                data['tokens'] = tokens
                data['token_ids'] = token_ids
                data['masks'] = [0] * len(token_ids)
                data['vocab_masks'] = self.get_vocab_masks(token_ids)

                triples = line['label']
                targets = []

                for triple in triples:
                    eles = triple.split()
                    head = tokens[int(eles[0]): int(eles[1]) + 1]
                    tail = tokens[int(eles[2]): int(eles[3]) + 1]
                    target = head + [self.tokenizer.get_ele_sep_token()] + tail + [self.tokenizer.get_ele_sep_token()] + [eles[4]]
                    targets.extend(target)
                    targets.append(self.tokenizer.get_triple_sep_token())

                targets[-1] = self.tokenizer.get_end_token()
                target_ids = self.tokenizer.convert_tokens_to_ids(targets)
                data['targets'] = targets
                data['target_ids'] = target_ids

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
    def __init__(self, data_file, relations_file, tokenizer: PTRTokenizer):
        super(PTRNREDataSet, self).__init__()
        self.data_file = data_file
        self.relations_file = relations_file
        self.tokenizer = tokenizer
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
        with open(self.data_file, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                data = {}
                tokens = self.tokenizer.tokenize(line['text'])
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                data['tokens'] = tokens
                data['token_ids'] = token_ids
                data['masks'] = [0] * len(token_ids)

                targets = line['label']

                head_start_offsets, head_end_offsets, tail_start_offsets, tail_end_offsets, relation_ids = [], [], [], [], []
                for target in targets:
                    elements = target.strip().split()
                    head_start_offsets.append(int(elements[0]))
                    head_end_offsets.append(int(elements[1]))
                    tail_start_offsets.append(int(elements[2]))
                    tail_end_offsets.append(int(elements[3]))
                    relation_ids.append(self.relation2idx[elements[4]])

                data['head_start_offsets'] = head_start_offsets + [-1]
                data['head_end_offsets'] = head_end_offsets + [-1]
                data['tail_start_offsets'] = tail_start_offsets + [-1]
                data['tail_end_offsets'] = tail_end_offsets + [-1]
                data['relation_ids'] = relation_ids + [self.relation2idx[self.tokenizer.get_end_token()]]
                data['target_masks'] = [0] * len(relation_ids)

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
        head_start_offsets = [data['head_start_offsets'] + [0] * (tgt_max_seq_len - len(data['head_start_offsets'])) for
                              data in batch]
        head_end_offsets = [data['head_end_offsets'] + [0] * (tgt_max_seq_len - len(data['head_end_offsets'])) for data
                            in batch]
        tail_start_offsets = [data['tail_start_offsets'] + [0] * (tgt_max_seq_len - len(data['tail_start_offsets'])) for
                              data in batch]
        tail_end_offsets = [data['tail_end_offsets'] + [0] * (tgt_max_seq_len - len(data['tail_end_offsets'])) for data
                            in batch]
        target_ids = [data['relation_ids'] + [0] * (tgt_max_seq_len - len(data['relation_ids'])) for data in batch]
        target_masks = [data['target_masks'] + [1] * (tgt_max_seq_len - len(data['target_masks'])) for data in batch]

        return torch.LongTensor(token_ids), torch.LongTensor(masks), torch.LongTensor(head_start_offsets), \
               torch.LongTensor(head_end_offsets), torch.LongTensor(tail_start_offsets), torch.LongTensor(
            tail_end_offsets), \
               torch.LongTensor(target_ids), torch.LongTensor(target_masks)


class NREDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/nre',
                 max_len: int = 200,
                 batch_size: int = 16,
                 relation_file: str = 'data/nre/relations.txt',
                 vocab_file: str = 'data/nre/vocab.txt',
                 model: str = 'prr'):
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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        if model == 'ptr':
            self.tokenizer = PTRTokenizer(vocab_file)
            self.data_class = PTRNREDataSet
        elif model == 'wd':
            self.tokenizer = WDTokenizer(vocab_file=vocab_file, label_file=relation_file)
            self.data_class = NREDataSet
        else:
            raise NotImplementedError
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = self.data_class(os.path.join(self.data_dir, 'train.txt'),
                                                 self.relation_file,
                                                 self.tokenizer)
            if os.path.isfile(os.path.join(self.data_dir, 'dev.txt')):
                self.val_dataset = self.data_class(os.path.join(self.data_dir, 'dev.txt'),
                                                   self.relation_file,
                                                   self.tokenizer)
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.val_dataset = self.data_class(os.path.join(self.data_dir, 'test.txt'),
                                               self.relation_file, self.tokenizer)

        # if stage == 'predict' or stage is None:
        #     self.predict_dataset = self.dataset_class(predict_file, tag_file, self.tokenizer, True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=self.data_class.collocate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=self.data_class.collocate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=self.data_class.collocate_fn)

    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.predict_dataset, self.batch_size, collate_fn=NREDataSet.collocate_fn)


if __name__ == '__main__':
    vocab_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/vocab.txt'
    data_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/train.txt'
    relations = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/relations.txt'
    tokenizer = WDTokenizer(vocab_file, data_file, relations)
    dataset = NREDataSet(data_file, relations, tokenizer)
    print(dataset[0])
    print(PTRNREDataSet.collocate_fn(dataset[: 10]))
