import json

import torch
import transformers

import layers
import metrics
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl
from torchmetrics import F1Score, Precision, Recall
from layers import MultiNonLinearLayer
from transformers import AutoModel, AutoConfig


optimizer_params = {
    'adamw': {
        '--beta1': 0.9,
        '--beta2': 0.98,
        '--eps': 1e-8,
        '--weight_decay': 0.01
    }
}


class BertBasedModule(pl.LightningModule):
    def __init__(self,
                 pretrained_model_name: str = 'bert-base-chinese',
                 warmup_steps: int = 1000,
                 num_total_steps: int = 270000,
                 *args, **kwargs
                 ):
        super(BertBasedModule, self).__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps
        self.num_total_steps = num_total_steps
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.bert_config = AutoConfig.from_pretrained(pretrained_model_name)

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


class MRCNERModule(BertBasedModule):
    def __init__(self, *args, **kwargs):
        super(MRCNERModule, self).__init__(*args, **kwargs)
        self.start = nn.Linear(self.bert_config.hidden_size, 1)
        self.end = nn.Linear(self.bert_config.hidden_size, 1)
        self.cls = MultiNonLinearLayer(self.bert_config.hidden_size * 2, self.bert_config.intermediate_size, 1, 2)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs, _ = self.bert(input_ids, attention_mask, token_type_ids, return_dict=False)
        batch_size, seq_len, hidden_size = bert_outputs.size()

        start_logits = self.start(bert_outputs)
        end_logits = self.end(bert_outputs)

        start_inputs = torch.unsqueeze(bert_outputs, 2).expand(-1, -1, seq_len, -1)
        end_inputs = torch.unsqueeze(bert_outputs, 1).expand(-1, seq_len, -1, -1)

        cls_inputs = torch.cat([start_inputs, end_inputs], -1)
        cls_outputs = self.cls(cls_inputs).squeeze(-1)

        return start_logits, end_logits, cls_outputs

    @staticmethod
    def compute_loss(logits, targets, masks):
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = (loss * masks).sum() / masks.sum()
        return loss

    def compute_step_states(self, batch, validation=True):
        input_ids, attention_mask, token_type_ids, start_labels, end_labels, span_labels = batch
        start_logits, end_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        span_logits = span_logits.squeeze(-1)

        masks = torch.logical_and(attention_mask, token_type_ids)
        start_loss = self.compute_loss(start_logits, start_labels, masks)
        end_loss = self.compute_loss(end_logits, end_labels, masks)

        span_masks = torch.logical_and(masks.unsqueeze(2).expand(-1, -1, masks.size(1)),
                                       masks.unsqueeze(1).expand(-1, -1, masks.size(1)))
        span_masks = torch.triu(span_masks, diagonal=0)
        span_loss = self.compute_loss(span_logits, span_labels, span_masks)

        loss = start_loss + end_loss + span_loss

        tp, fp, fn = metrics.mrc_span_f1(start_logits, end_logits, span_logits, masks, masks, span_labels)
        recall, precision, f1 = metrics.get_f1_score(tp, fp, fn)

        logs = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'f1_score': f1,
            'recall': recall,
            'precision': precision,
            'loss': loss,
            'start_loss': start_loss,
            'end_loss': end_loss
        }
        if validation:
            logs = {'val_' + key: value for key, value in logs.items()}

        return logs

    def compute_epoch_states(self, outputs, validation=True):
        if validation:
            tps = torch.Tensor([output['val_tp'] for output in outputs]).sum()
            fps = torch.Tensor([output['val_fp'] for output in outputs]).sum()
            fns = torch.Tensor([output['val_fn'] for output in outputs]).sum()
            loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        else:
            tps = torch.Tensor([output['tp'] for output in outputs]).sum()
            fps = torch.Tensor([output['fp'] for output in outputs]).sum()
            fns = torch.Tensor([output['fn'] for output in outputs]).sum()
            loss = torch.stack([output['loss'] for output in outputs]).mean()
        recall, precision, f1 = metrics.get_f1_score(tps, fps, fns)

        logs = {
            'tp': tps,
            'fp': fps,
            'fn': fns,
            'loss': loss,
            'f1_score': f1,
            'recall': recall,
            'precision': precision
        }
        if validation:
            logs = {'val_' + key: value for key, value in logs.items()}

        self.log_dict(logs)

    def training_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch, validation=False)
        logs['lr'] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict(logs, prog_bar=True, on_epoch=True)
        return logs

    def training_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, False)

    def validation_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch)
        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_epoch_end(self, outputs):
        self.compute_epoch_states(outputs)


class BiLSTMLanNERModule(pl.LightningModule):
    def __init__(self,
                 embedding_size: int = 128,
                 hidden_size: int = 128,
                 vocab_size: int = 21128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 tag_file: str = 'data/tags.json'
                 ):
        super(BiLSTMLanNERModule, self).__init__()
        with open(tag_file, encoding='utf-8') as f:
            self.idx2tag = json.load(f)
            self.idx2tag = {int(key): value for key, value in self.idx2tag.items()}
        self.num_labels = len(self.idx2tag)
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.label_embeddings = nn.Embedding(len(self.idx2tag), hidden_size)
        label_ids = torch.arange(0, len(self.idx2tag), dtype=torch.int64).unsqueeze(0)
        self.register_buffer('label_ids', label_ids)
        # label_embeddings = torch.FloatTensor(self.num_labels, hidden_size)
        # init.xavier_normal_(label_embeddings)
        # self.register_buffer('label_embeddings', nn.Parameter(label_embeddings))

        self.models = nn.ModuleList([layers.BiLSTMLan(embedding_size, hidden_size, num_heads)])
        for i in range(0, num_layers - 2):
            self.models.append(layers.BiLSTMLan(hidden_size * 2, hidden_size, num_heads))
        self.models.append(layers.BiLSTMLan(hidden_size * 2, hidden_size, 1))
        self.save_hyperparameters()

    def forward(self, input_ids, masks=None):
        inputs = self.embeddings(input_ids)
        label_ids = self.label_ids.expand(input_ids.size(0), -1)
        for layer in self.models[: -1]:
            label_embeddings = self.label_embeddings(label_ids)
            # label_embeddings = torch.unsqueeze(self.label_embeddings, 0).expand([inputs.size(0), -1, -1])
            inputs = layer(inputs, label_embeddings, masks)

        # label_embeddings = torch.unsqueeze(self.label_embeddings, 0).expand([inputs.size(0), -1, -1])
        label_embeddings = self.label_embeddings(label_ids)
        outputs = self.models[-1](inputs, label_embeddings, masks, True)
        return outputs

    def compute_step_states(self, batch, validation=True):
        input_ids, targets, masks = batch
        outputs = self(input_ids, masks)
        loss = F.cross_entropy(outputs.view(-1, self.num_labels), targets.view(-1), reduction='none', ignore_index=0) * masks.view(-1)
        loss = loss.sum() / masks.sum()

        preds = outputs.argmax(-1)
        tp, fp, fn = metrics.flat_ner_stats(preds, targets, masks, self.idx2tag)
        precision, recall, f1 = metrics.get_f1_score(tp, fp, fn)

        logs = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'loss': loss,
            'f1_score': f1,
            'recall': recall,
            'precision': precision
        }
        if validation:
            logs = {'val_' + key: value for key, value in logs.items()}

        return logs

    def compute_epoch_states(self, outputs, validation=True):
        if validation:
            tps = torch.Tensor([output['val_tp'] for output in outputs]).sum()
            fps = torch.Tensor([output['val_fp'] for output in outputs]).sum()
            fns = torch.Tensor([output['val_fn'] for output in outputs]).sum()
            loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        else:
            tps = torch.Tensor([output['tp'] for output in outputs]).sum()
            fps = torch.Tensor([output['fp'] for output in outputs]).sum()
            fns = torch.Tensor([output['fn'] for output in outputs]).sum()
            loss = torch.stack([output['loss'] for output in outputs]).mean()
        recall, precision, f1 = metrics.get_f1_score(tps, fps, fns)

        logs = {
            'tp': tps,
            'fp': fps,
            'fn': fns,
            'loss': loss,
            'f1_score': f1,
            'recall': recall,
            'precision': precision
        }
        if validation:
            logs = {'val_' + key: value for key, value in logs.items()}

        self.log_dict(logs)

    def training_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch, validation=False)
        logs['lr'] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict(logs, on_epoch=True, prog_bar=True)
        return logs

    def training_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, validation=False)

    def validation_step(self, batch, batch_idx):
        logs = self.compute_step_states(batch)
        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_epoch_end(self, outputs):
        self.compute_epoch_states(outputs)