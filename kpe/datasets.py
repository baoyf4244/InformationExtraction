import json
import os
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
from tokenization import KPEChineseCharTokenizer
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning import LightningDataModule


class KPEDataSet(Dataset):
    def __init__(self, data_file, tokenizer: KPEChineseCharTokenizer):
        super(KPEDataSet, self).__init__()
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.datasets = []
        self.init()

    def init(self):
        self.make_dataset()

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

    def make_dataset(self):
        with open(self.data_file, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                tokens = self.tokenizer.tokenize(line['text'])
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                masks = [1] * len(token_ids)

                input_vocab_ids = []
                oov2idx = {}
                for token, token_id in zip(tokens, token_ids):
                    if token_id == self.tokenizer.get_unk_id():
                        if token not in oov2idx:
                            oov2idx[token] = self.tokenizer.get_vocab_size() + len(oov2idx)
                        input_vocab_ids.append(oov2idx[token])
                    else:
                        input_vocab_ids.append(token_id)

                targets = self.tokenizer.tokenize(self.tokenizer.get_sep_token().join(line['keywords']))
                target_ids = self.tokenizer.convert_tokens_to_ids(targets) + [self.tokenizer.get_end_id()]

                data = {
                    'id': line['id'],
                    'masks': masks,
                    'targets': targets,
                    'input_ids': token_ids,
                    'target_ids': target_ids,
                    'target_masks': [1] * len(target_ids),
                    'input_vocab_ids': input_vocab_ids,
                    'oov2idx': oov2idx
                }
                self.datasets.append(data)

    @staticmethod
    def pad(seqs):
        max_len = max(len(seq) for seq in seqs)
        padded_seqs = []
        for seq in seqs:
            padded_seqs.append(seq + [0] * (max_len - len(seq)))
        return padded_seqs

    @classmethod
    def collocate_fn(cls, batch):
        ids = [b['id'] for b in batch]
        targets = [b['targets'] for b in batch]
        masks = torch.LongTensor(cls.pad([b['masks'] for b in batch]))
        input_ids = torch.LongTensor(cls.pad([b['input_ids'] for b in batch]))
        target_masks = torch.LongTensor(cls.pad([b['target_masks'] for b in batch]))
        target_ids = torch.LongTensor(cls.pad([b['target_ids'] for b in batch]))
        input_vocab_ids = torch.LongTensor(cls.pad([b['input_vocab_ids'] for b in batch]))
        oov2idx = [b['oov2idx'] for b in batch]
        oov_ids = [list(oov.values()) for oov in oov2idx]
        oov_id_masks = [[1] * len(oov_id) for oov_id in oov_ids]
        oov_ids = torch.LongTensor(cls.pad(oov_ids))
        oov_id_masks = torch.LongTensor(cls.pad(oov_id_masks))
        return ids, targets, input_ids, masks, target_ids, target_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx


class KPEDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data/kpe',
                 max_len: int = 200,
                 batch_size: int = 16,
                 vocab_file: str = 'data/kpe/vocab.txt',
                 min_freq: int = 5):
        """
        Args:
            data_dir:
            max_len:
            batch_size:
            vocab_file:
        """
        super(KPEDataModule, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_file = os.path.join(data_dir, 'train.txt')
        self.val_file = os.path.join(data_dir, 'val.txt')
        self.test_file = os.path.join(data_dir, 'test.txt')
        self.predict_file = os.path.join(data_dir, 'predict.txt')
        self.tokenizer = KPEChineseCharTokenizer(vocab_file, self.train_file, min_freq=min_freq)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = KPEDataSet(self.train_file, self.tokenizer)
            if os.path.isfile(self.val_file):
                self.val_dataset = KPEDataSet(self.val_file, self.tokenizer)
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = KPEDataSet(self.test_file, self.tokenizer)

        if stage == 'predict' or stage is None:
            self.predict_dataset = KPEDataSet(self.predict_file, self.tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=KPEDataSet.collocate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=KPEDataSet.collocate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=KPEDataSet.collocate_fn)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset, self.batch_size, collate_fn=KPEDataSet.collocate_fn)


if __name__ == '__main__':
    vocab_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/kpe/vocab.txt'
    data_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/kpe/dev.txt'

    tokenizer = KPEChineseCharTokenizer(vocab_file, data_file, min_freq=10)
    dataset = KPEDataSet(data_file, tokenizer)
    print(dataset[0])
    print(KPEDataSet.collocate_fn(dataset[: 10]))
