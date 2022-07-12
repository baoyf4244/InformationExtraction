import torch
import transformers

import metrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
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

    def compute_total_loss(self, batch, tf_board_logs, val=False):
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
        tf_board_logs['start_loss'] = start_loss
        tf_board_logs['end_loss'] = end_loss
        tf_board_logs['span_loss'] = span_loss
        tf_board_logs['val_loss' if val else 'train_loss'] = loss

        tp, fp, fn = metrics.mrc_span_f1(start_logits, end_logits, span_logits, masks, masks, span_labels)
        recall, precision, f1 = metrics.get_f1_score(tp, fp, fn)

        tf_board_logs['tp'] = tp
        tf_board_logs['fp'] = fp
        tf_board_logs['fn'] = fn
        tf_board_logs['recall'] = recall
        tf_board_logs['precision'] = precision
        tf_board_logs['f1'] = f1

        return tf_board_logs

    def training_step(self, batch, batch_idx):
        tf_board_logs = {
            'lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        self.compute_total_loss(batch, tf_board_logs)
        self.log_dict(tf_board_logs, prog_bar=True)
        return tf_board_logs['train_loss']

    def validation_step(self, batch, batch_idx):
        tf_board_logs = {}
        self.compute_total_loss(batch, tf_board_logs, val=True)
        self.log_dict(tf_board_logs)
        return tf_board_logs

    def validation_epoch_end(self, outputs):
        tps = torch.stack([output['tp'] for output in outputs]).sum()
        fps = torch.stack([output['fp'] for output in outputs]).sum()
        fns = torch.stack([output['fn'] for output in outputs]).sum()
        recall, precision, f1 = metrics.get_f1_score(tps, fps, fns)
        val_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        tf_board_logs = {
            'val_loss': val_loss,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        self.log_dict(tf_board_logs, prog_bar=True)
        return tf_board_logs


