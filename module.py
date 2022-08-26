import json
import os
import utils
import torch
from enum import Enum
from typing import Optional
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningModule, LightningDataModule


class IEModule(LightningModule):
    def get_f1_stats(self, preds, targets, masks=None):
        raise NotImplementedError

    @staticmethod
    def get_f1_score(tp, fp, fn):
        recall = tp / (tp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        f1 = 2 * recall * precision / (recall + precision + 1e-10)
        return recall, precision, f1

    def get_training_outputs(self, batch):
        input_ids, targets, masks = batch
        preds, loss = self(input_ids, targets, masks)
        return preds, targets, masks, loss

    def compute_step_states(self, batch, stage):
        preds, targets, masks, loss = self.get_training_outputs(batch)
        tp, fp, fn = self.get_f1_stats(preds, targets, masks)
        recall, precision, f1 = self.get_f1_score(tp, fp, fn)

        logs = {
            stage + '_tp': tp,
            stage + '_fp': fp,
            stage + '_fn': fn,
            stage + '_loss': loss,
            stage + '_f1_score': f1,
            stage + '_recall': recall,
            stage + '_precision': precision
        }

        if stage == 'train':
            logs['loss'] = logs['train_loss']
            logs['lr'] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict(logs, prog_bar=True)

        return logs

    def compute_epoch_states(self, outputs, stage='val'):
        tps = torch.Tensor([output[stage + '_tp'] for output in outputs]).sum()
        fps = torch.Tensor([output[stage + '_fp'] for output in outputs]).sum()
        fns = torch.Tensor([output[stage + '_fn'] for output in outputs]).sum()
        loss = torch.stack([output[stage + '_loss'] for output in outputs]).mean()

        recall, precision, f1 = self.get_f1_score(tps, fps, fns)

        logs = {
            stage + '_tp': tps,
            stage + '_fp': fps,
            stage + '_fn': fns,
            stage + '_loss': loss,
            stage + '_f1_score': f1,
            stage + '_recall': recall,
            stage + '_precision': precision
        }

        self.log_dict(logs)

    def training_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch, stage='train')
        return logs

    def training_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, stage='train')

    def validation_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch, stage='val')
        return logs

    def validation_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, stage='val')

    def test_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch, stage='test')
        return logs

    def test_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, stage='test')


class IEDataModule(LightningDataModule):
    def __init__(
            self,
            max_len: int = 200,
            batch_size: int = 16,
            data_dir: str = 'data'
    ):
        """
        Args:
            max_len:
            batch_size:
            data_dir:
        """
        super(IEDataModule, self).__init__()
        self.max_len = max_len
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.save_hyperparameters()

    def get_dataset(self, data_file, is_predict=False):
        raise NotImplementedError

    def collocate_fn(self, batch):
        raise NotImplementedError

    @staticmethod
    def pad_batch(batch, pad_token):
        max_len = max(len(b) for b in batch)
        batch = [b + [pad_token] * (max_len - len(b)) for b in batch]
        return torch.LongTensor(batch)

    @staticmethod
    def get_data_by_name(batch, name):
        return [b[name] for b in batch]

    def setup(self, stage: Optional[str] = None) -> None:
        train_file = os.path.join(self.data_dir, 'train.txt')
        val_file = os.path.join(self.data_dir, 'dev.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        predict_file = os.path.join(self.data_dir, 'predict.txt')
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset(train_file)
            if os.path.isfile(val_file):
                self.val_dataset = self.get_dataset(val_file)
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            if os.path.isfile(test_file):
                self.test_dataset = self.get_dataset(test_file)

        if stage == 'predict' or stage is None:
            self.predict_dataset = self.get_dataset(predict_file, True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collocate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=self.collocate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collocate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, self.batch_size, collate_fn=self.collocate_fn)


class IEDataSet(Dataset):
    def __init__(self, data_file, max_len=200, is_predict=False, *args, **kwargs):
        super(IEDataSet, self).__init__()
        self.data_file = data_file
        self.max_len = max_len
        self.is_predict = is_predict
        self.dataset = []

    def get_data(self, line):
        raise NotImplementedError

    def make_dataset(self):
        with open(self.data_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                data = self.get_data(line)

                if isinstance(data, list):
                    self.dataset.extend(data)
                else:
                    self.dataset.append(data)
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class Vocab:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.vocab = self.load_vocab()
        self.word2idx = {v: idx for idx, v in enumerate(self.vocab)}

    def load_vocab(self):
        with open(self.vocab_file, encoding='utf-8') as f:
            vocab = f.readlines()
        return [v.strip() for v in vocab if v.strip()]

    def convert_token_to_id(self, token):
        return self.word2idx[token] if token in self.word2idx else self.word2idx[SpecialTokens.UNK.value]

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[idx] for idx in ids]

    def convert_id_to_token(self, idx):
        return self.vocab[idx]

    def get_pad_id(self):
        return self.word2idx[SpecialTokens.PAD.value]

    @staticmethod
    def get_pad_token():
        return SpecialTokens.PAD.value

    def get_unk_id(self):
        return self.word2idx[SpecialTokens.UNK.value]

    @staticmethod
    def get_unk_token():
        return SpecialTokens.UNK.value

    def get_start_id(self):
        return self.word2idx[SpecialTokens.SOS.value]

    @staticmethod
    def get_start_token():
        return SpecialTokens.SOS.value

    def get_end_id(self):
        return self.word2idx[SpecialTokens.EOS.value]

    @staticmethod
    def get_end_token(self):
        return SpecialTokens.EOS.value

    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab


class SpecialTokens(Enum):
    PAD = '[PAD]'
    UNK = '[UNK]'
    SOS = '[CLS]'
    EOS = '[SEP]'
    NON_ENTITY = 'O'

