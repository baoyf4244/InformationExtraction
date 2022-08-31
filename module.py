import os
import json
import torch
import transformers
from enum import Enum
from typing import Optional
from collections import Counter
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
        raise NotImplementedError

    def compute_step_stats(self, batch, stage):
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

    def compute_epoch_stats(self, outputs, stage='val'):
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
        logs = self.compute_step_stats(batch, stage='train')
        return logs

    def training_epoch_end(self, outputs):
        self.compute_epoch_stats(outputs, stage='train')

    def validation_step(self, batch, batch_idx):
        logs = self.compute_step_stats(batch, stage='val')
        return logs

    def validation_epoch_end(self, outputs):
        self.compute_epoch_stats(outputs, stage='val')

    def test_step(self, batch, batch_idx):
        logs = self.compute_step_stats(batch, stage='test')
        return logs

    def test_epoch_end(self, outputs):
        self.compute_epoch_stats(outputs, stage='test')


class PreTrainBasedModule(IEModule):
    def __init__(
            self,
            warmup_steps: int = 1000,
            num_total_steps: int = 270000,
            pretrained_model_name: str = 'bert-base-chinese'
    ):
        super(PreTrainBasedModule, self).__init__()
        self.warmup_steps = warmup_steps
        self.num_total_steps = num_total_steps
        self.config = self.get_config(pretrained_model_name)
        self.pretrained_model = self.get_pretrained_model(pretrained_model_name)

    @staticmethod
    def get_config(pretrained_model_name):
        return transformers.AutoConfig.from_pretrained(pretrained_model_name)

    @staticmethod
    def get_pretrained_model(pretrained_model_name):
        return transformers.AutoModel.from_pretrained(pretrained_model_name)

    def get_f1_stats(self, preds, targets, masks=None):
        pass

    def get_training_outputs(self, batch):
        pass

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optim = torch.optim.AdamW(optimizer_grouped_parameters,
                                  2e-5,
                                  (0.9, 0.98),
                                  1e-8,
                                  0.01)

        lr = transformers.get_polynomial_decay_schedule_with_warmup(optim, self.warmup_steps,
                                                                    self.num_total_steps, lr_end=2e-5 / 5)
        return [optim], [lr]


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
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.val_file = os.path.join(self.data_dir, 'dev.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.predict_file = os.path.join(self.data_dir, 'predict.txt')
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
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset(self.train_file)
            if os.path.isfile(self.val_file):
                self.val_dataset = self.get_dataset(self.val_file)
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            if os.path.isfile(self.test_file):
                self.test_dataset = self.get_dataset(self.test_file)

        if stage == 'predict' or stage is None:
            self.predict_dataset = self.get_dataset(self.predict_file, True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collocate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=self.collocate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collocate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, self.batch_size, collate_fn=self.collocate_fn)


class IEDataSet(Dataset):
    def __init__(self, data_file, max_len=200, is_predict=False):
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


class LabelVocab:
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.special_tokens = self.get_special_tokens()
        self.vocab = self.load_vocab()
        self.word2idx = {v: idx for idx, v in enumerate(self.vocab)}

    def load_vocab(self):
        with open(self.vocab_file, encoding='utf-8') as f:
            vocab = f.readlines()
        vocab = [v.strip() for v in vocab if v.strip()]
        special_vocab = []
        for v in vocab:
            if v not in self.special_tokens:
                special_vocab.append(v)

        return self.special_tokens + special_vocab

    @staticmethod
    def get_special_tokens():
        return []

    def convert_token_to_id(self, token):
        return self.word2idx[token]

    def convert_tokens_to_ids(self, tokens):
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[idx] for idx in ids]

    def convert_id_to_token(self, idx):
        return self.vocab[idx]

    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab


class Vocab(LabelVocab):
    def __init__(self, data_file=None, min_freq=10, do_lower=False, *args, **kwargs):
        self.data_file = data_file
        self.min_freq = min_freq
        self.do_lower = do_lower
        super(Vocab, self).__init__(*args, **kwargs)

    def load_vocab(self):
        try:
            return super(Vocab, self).load_vocab()
        except Exception:
            assert self.data_file is not None, 'vocab.txt不存在或不完整时，数据文件不能为空'
            counter = Counter()
            with open(self.data_file, encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    tokens = self.tokenize(line['text'])
                    counter.update(tokens)

            vocab = []
            with open(self.vocab_file, 'w', encoding='utf-8') as f:
                for key, freq in counter.items():
                    if freq >= self.min_freq and key not in self.get_special_tokens():
                        vocab.append(key)
                        f.write(key + '\n')

            vocab = self.get_special_tokens() + vocab
            return vocab

    @staticmethod
    def get_special_tokens():
        return [SpecialTokens.PAD.value, SpecialTokens.UNK.value]

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

    def tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self.is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output).strip().split()

    @staticmethod
    def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((0x4E00 <= cp <= 0x9FFF) or  #
                (0x3400 <= cp <= 0x4DBF) or  #
                (0x20000 <= cp <= 0x2A6DF) or  #
                (0x2A700 <= cp <= 0x2B73F) or  #
                (0x2B740 <= cp <= 0x2B81F) or  #
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or  #
                (0x2F800 <= cp <= 0x2FA1F)):  #
            return True

        return False

    def tokenize(self, text):
        if self.do_lower:
            text = text.lower()
        return self.tokenize_chinese_chars(text)

    def convert_token_to_id(self, token):
        return self.word2idx[token] if token in self.word2idx else self.word2idx[SpecialTokens.UNK.value]


class Seq2SeqVocab(Vocab):
    @staticmethod
    def get_special_tokens():
        return Vocab.get_special_tokens() + [SpecialTokens.SOS.value, SpecialTokens.EOS.value]

    def get_start_id(self):
        return self.word2idx[SpecialTokens.SOS.value]

    @staticmethod
    def get_start_token():
        return SpecialTokens.SOS.value

    def get_end_id(self):
        return self.word2idx[SpecialTokens.EOS.value]

    @staticmethod
    def get_end_token():
        return SpecialTokens.EOS.value


class SpecialTokens(Enum):
    PAD = '[PAD]'
    UNK = '[UNK]'
    SOS = '[CLS]'
    EOS = '[SEP]'
    VERTICAL = '|'
    SEMICOLON = ';'
    NON_ENTITY = 'O'


if __name__ == '__main__':
    vocab = Vocab(vocab_file='data/kpe/vocab.txt', data_file='data/kpe/train.txt')
    tokens = vocab.tokenize('then terrorism struck again , this time in the indonesia capital of jakarta .')
    print(tokens)