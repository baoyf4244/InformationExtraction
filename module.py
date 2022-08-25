import os
import torch
from typing import Optional
from torch.utils.data import DataLoader, random_split
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
            data_dir: str = 'data',
            model_name: str = 'BiLSTM-LAN',
            tag_file: str = 'data/idx2tag_question.json'
    ):
        """

        :param data_dir: 数据存放目录
        :param max_len: 数据保留的最大长度
        :param batch_size:
        :param pretrained_model_name:
        """
        super(IEDataModule, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.dataset_class = self.get_dataset(model_name)
        self.tag_file = tag_file
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.save_hyperparameters()

    @staticmethod
    def get_dataset(model_name):
        raise NotImplementedError

    def get_tokenizer(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        train_file = os.path.join(self.data_dir, 'train.txt')
        val_file = os.path.join(self.data_dir, 'dev.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        predict_file = os.path.join(self.data_dir, 'predict.txt')
        tokenizer = self.get_tokenizer()
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(train_file, self.tag_file, tokenizer)
            if os.path.isfile(val_file):
                self.val_dataset = self.dataset_class(val_file, self.tag_file, tokenizer)
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            if os.path.isfile(test_file):
                self.test_dataset = self.dataset_class(test_file, self.tag_file, tokenizer)

        if stage == 'predict' or stage is None:
            self.predict_dataset = self.dataset_class(predict_file, self.tag_file, tokenizer, True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=self.dataset_class.collocate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=self.dataset_class.collocate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=self.dataset_class.collocate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, self.batch_size, collate_fn=self.dataset_class.collocate_fn)

