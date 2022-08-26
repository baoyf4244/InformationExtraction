import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiNonLinearLayer
from ner.modeling_base import BertBasedModule
from torch.utils.data import Dataset, DataLoader


class MRCNERModule(BertBasedModule):
    def __init__(self, *args, **kwargs):
        super(MRCNERModule, self).__init__(*args, **kwargs)
        self.start = nn.Linear(self.config.hidden_size, 1)
        self.end = nn.Linear(self.config.hidden_size, 1)
        self.cls = MultiNonLinearLayer(self.config.hidden_size * 2, self.config.intermediate_size, 1, 2)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs, _ = self.pretrained_model(input_ids, attention_mask, token_type_ids, return_dict=False)
        batch_size, seq_len, hidden_size = bert_outputs.size()

        start_logits = self.start(bert_outputs)
        end_logits = self.end(bert_outputs)

        start_inputs = torch.unsqueeze(bert_outputs, 2).expand(-1, -1, seq_len, -1)
        end_inputs = torch.unsqueeze(bert_outputs, 1).expand(-1, seq_len, -1, -1)

        cls_inputs = torch.cat([start_inputs, end_inputs], -1)
        cls_outputs = self.cls(cls_inputs).squeeze(-1)

        return start_logits, end_logits, cls_outputs

    @staticmethod
    def get_loss(logits, targets, masks):
        targets = targets.float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = (loss * masks).sum() / masks.sum()
        return loss

    @staticmethod
    def get_spans(start_logits, end_logits, span_logits, masks):
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        span_preds = span_logits > 0

        bsz, seq_len = masks.size()

        start_masks = masks.bool()
        end_masks = masks.bool()

        start_preds = start_preds.bool()
        end_preds = end_preds.bool()

        span_preds = (span_preds
                      & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                      & end_preds.unsqueeze(1).expand(-1, seq_len, -1))
        span_label_mask = (start_masks.unsqueeze(-1).expand(-1, -1, seq_len)
                           & end_masks.unsqueeze(1).expand(-1, seq_len, -1))
        span_label_mask = torch.triu(span_label_mask, 0)  # start should be less or equal to end
        span_preds = span_label_mask & span_preds
        return span_preds

    def get_f1_stats(self, span_preds, span_labels, masks=None):
        """
        Compute span f1 according to query-based model output
        Args:
            span_preds: [bsz, seq_len, seq_len]
            masks: [bsz, seq_len]
            span_labels: [bsz, seq_len, seq_len]
        Returns:
            span-f1 counts, tensor of shape [3]: tp, fp, fn
        """
        tp = (span_labels & span_preds).long().sum()
        fp = (~span_labels & span_preds).long().sum()
        fn = (span_labels & ~span_preds).long().sum()
        return tp, fp, fn

    def get_training_outputs(self, batch):
        input_ids, attention_mask, token_type_ids, start_labels, end_labels, span_labels = batch
        masks = torch.logical_and(attention_mask, token_type_ids)

        start_logits, end_logits, span_logits = self(input_ids, attention_mask, token_type_ids)
        span_preds = self.get_spans(start_logits, end_logits, span_logits, masks)
        return span_preds, span_labels, masks, start_logits, end_logits, span_logits, start_labels, end_labels

    def compute_step_states(self, batch, stage):
        span_preds, span_labels, masks, start_logits, end_logits, span_logits, start_labels, end_labels = self.get_training_outputs(batch)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        span_logits = span_logits.squeeze(-1)

        start_loss = self.get_loss(start_logits, start_labels, masks)
        end_loss = self.get_loss(end_logits, end_labels, masks)

        span_masks = torch.logical_and(masks.unsqueeze(2).expand(-1, -1, masks.size(1)),
                                       masks.unsqueeze(1).expand(-1, -1, masks.size(1)))
        span_masks = torch.triu(span_masks, diagonal=0)
        span_loss = self.get_loss(span_logits, span_labels, span_masks)

        loss = start_loss + end_loss + span_loss

        tp, fp, fn = self.get_f1_stats(span_preds, span_labels, masks)
        recall, precision, f1 = self.get_f1_score(tp, fp, fn)

        logs = {
            stage + '_tp': tp,
            stage + '_fp': fp,
            stage + '_fn': fn,
            stage + '_f1_score': f1,
            stage + '_recall': recall,
            stage + '_precision': precision,
            stage + '_loss': loss,
            stage + '_start_loss': start_loss,
            stage + '_end_loss': end_loss
        }

        if stage == 'train':
            logs['loss'] = logs['train_loss']

        self.log_dict(logs, prog_bar=True)

        return logs
