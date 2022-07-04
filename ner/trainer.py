import torch
import metrics
import torch.nn.functional as F
import pytorch_lightning as pl
from ner.models import BertMRC

pl.seed_everything(124)


class SpanNERTrainer(pl.LightningModule):
    def __init__(self):
        super(SpanNERTrainer, self).__init__()
        self.model = BertMRC.from_pretrained('bert-base-chinese')

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask, token_type_ids)

    @staticmethod
    def compute_loss(logits, targets, masks):
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = (loss * masks).sum() / masks.sum()
        return loss

    def compute_total_loss(self, batch, tf_board_logs, val=False):
        input_ids, attention_mask, token_type_ids, start_labels, end_labels, span_labels = batch
        start_logits, end_logits, span_logits = self.model(input_ids, attention_mask, token_type_ids)

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
        recall = tp / (tp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        f1 = 2 * recall * precision / (recall + precision + 1e-10)

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
        return torch.stack([tf_board_logs['tp'], tf_board_logs['fp'], tf_board_logs['fn']])

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optim = torch.optim.AdamW(optimizer_grouped_parameters, 2e-5, (0.9, 0.98), 1e-8, 0.01)
        lr = torch.optim.lr_scheduler.StepLR(optim, 1000)
        return [optim], [lr]
