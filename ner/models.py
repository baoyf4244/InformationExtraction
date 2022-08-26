import json
import torch
import layers
import metrics
import transformers
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchcrf import CRF
from module import IEModule
from collections import defaultdict
from layers import MultiNonLinearLayer
from transformers import AutoModel, AutoConfig


class NERModule(IEModule):
    def get_f1_stats(self, preds, targets, masks=None):
        tp, fp, fn = 0, 0, 0
        pred_chunks = self.get_batch_ner_chunks(preds, masks)
        target_chunks = self.get_ner_chunks(targets, masks)
        for pred_chunk, target_chunk in zip(pred_chunks, target_chunks):
            for label, indices in target_chunk.items():
                pred_indices = pred_chunk[label]
                tp += len(pred_indices.intersection(indices))

                for idx in pred_indices:
                    if idx not in indices:
                        fp += 1

                for idx in indices:
                    if idx not in pred_indices:
                        fn += 1
        return tp, fp, fn

    def get_batch_ner_chunks(self, tag_ids, masks):
        chunks = []
        for tag_id, mask in zip(tag_ids, masks):
            chunks.append(self.get_ner_chunks(tag_id, mask))
        return chunks

    def get_ner_chunks(self, tag_ids, masks):
        """
        根据标签ID返回标签名称对应的索引位置，标签采用BIO模式
        :param masks:
        :param tag_ids: 标签ID
        :return: dict，标签名称及相应的索引列表，索引列表为二元组，元组第一个元素为实体起始位置，第二个索引为实体结束位置+1（方便取值）
        """
        if isinstance(tag_ids, torch.Tensor):
            tag_ids = tag_ids.cpu().numpy().tolist()

        if isinstance(masks, torch.Tensor):
            seq_len = masks.sum()
        else:
            seq_len = sum(masks)
        tags = [self.idx2tag[tag_id] for tag_id in tag_ids]
        chunks = defaultdict(set)
        i = 0
        while i < seq_len:
            if tags[i].startswith('B-'):
                label = tags[i].split('-')[1]
                start = i
                i += 1
                while i < seq_len and tags[i].startswith('I-') and tags[i].split('-')[1] == label:
                    i += 1
                chunks[label].add((start, i))
            else:
                i += 1

        return chunks


class BertBasedModule(NERModule):
    def __init__(
            self,
            pretrained_model_name: str = 'bert-base-chinese',
            warmup_steps: int = 1000,
            num_total_steps: int = 270000
    ):
        super(BertBasedModule, self).__init__()
        self.warmup_steps = warmup_steps
        self.num_total_steps = num_total_steps
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name)

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

    def compute_step_states(self, batch, stage):
        input_ids, attention_mask, token_type_ids, start_labels, end_labels, span_labels = batch
        masks = torch.logical_and(attention_mask, token_type_ids)

        start_logits, end_logits, span_logits = self(input_ids, attention_mask, token_type_ids)
        span_preds = self.get_spans(start_logits, end_logits, span_logits, masks)

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


class BiLSTMLanNERModule(NERModule):
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
        self.models = nn.ModuleList([layers.BiLSTMLan(embedding_size, hidden_size, num_heads)])
        for i in range(0, num_layers - 2):
            self.models.append(layers.BiLSTMLan(hidden_size * 2, hidden_size, num_heads))
        self.models.append(layers.BiLSTMLan(hidden_size * 2, hidden_size, 1))
        self.loss = nn.NLLLoss(ignore_index=0, reduction='sum')
        self.save_hyperparameters()

    def forward(self, input_ids, tag_ids=None, masks=None):
        inputs = self.embeddings(input_ids)
        label_ids = self.label_ids.expand(input_ids.size(0), -1)
        label_embeddings = self.label_embeddings(label_ids)
        for layer in self.models[: -1]:
            inputs = layer(inputs, label_embeddings, masks)

        outputs = self.models[-1](inputs, label_embeddings, masks, True)
        if tag_ids is None:
            return outputs.argmax(-1)

        loss = self.get_loss(outputs, tag_ids, masks)
        return outputs.argmax(-1), loss

    def get_loss(self, outputs, targets, masks=None):
        batch_size = outputs.size(0)
        outputs = outputs.view(-1, self.num_labels)
        outputs = F.log_softmax(outputs, dim=-1)
        targets = targets.view(-1)
        loss = self.loss(outputs, targets)
        return loss / batch_size


class BiLSTMCrfNERModule(NERModule):
    def __init__(self,
                 embedding_size: int = 128,
                 hidden_size: int = 128,
                 vocab_size: int = 21128,
                 num_layers: int = 2,
                 tag_file: str = 'data/tags.json'
                 ):
        super(BiLSTMCrfNERModule, self).__init__()
        with open(tag_file, encoding='utf-8') as f:
            self.idx2tag = json.load(f)
            self.idx2tag = {int(key): value for key, value in self.idx2tag.items()}
        self.num_labels = len(self.idx2tag)
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, len(self.idx2tag))
        self.crf = CRF(len(self.idx2tag), batch_first=True)
        self.save_hyperparameters()

    def forward(self, input_ids, tag_ids=None, masks=None):
        inputs = self.embeddings(input_ids)
        lstm_outputs, _ = self.lstm(inputs)
        linear_outputs = self.linear(lstm_outputs)
        if masks is not None:
            masks = masks.bool()

        outputs = self.crf.decode(linear_outputs, masks)
        if tag_ids is None:
            return outputs
        loss = self.crf(linear_outputs, tag_ids, masks)
        loss = loss / input_ids.size(0)
        return outputs, -loss

