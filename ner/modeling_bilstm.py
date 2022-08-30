import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Vocab, LabelVocab
from layers import MultiHeadAttention
from ner.modeling_base import NERModule
from torchcrf import CRF


class BiLSTMLan(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(BiLSTMLan, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.lstm = nn.LSTM(input_size, hidden_size // 2, 1, batch_first=True, bidirectional=True)
        self.lan = MultiHeadAttention(num_heads, hidden_size)

    def forward(self, inputs, label_inputs, masks=None, inference=False):
        lstm_outputs, _ = self.lstm(inputs)
        outputs = self.lan(lstm_outputs, label_inputs, label_inputs, masks, inference=inference)
        if inference:
            return outputs
        return torch.cat([outputs, lstm_outputs], dim=-1)


class BiLSTMLanNERModule(NERModule):
    def __init__(
            self,
            embedding_size: int = 128,
            hidden_size: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            *args, **kwargs
    ):
        super(BiLSTMLanNERModule, self).__init__(*args, **kwargs)
        self.embeddings = nn.Embedding(self.vocab_size, embedding_size)
        self.label_embeddings = nn.Embedding(self.num_labels, hidden_size)
        label_ids = torch.arange(0, self.num_labels, dtype=torch.int64).unsqueeze(0)
        self.register_buffer('label_ids', label_ids)
        self.models = nn.ModuleList([BiLSTMLan(embedding_size, hidden_size, num_heads)])
        for i in range(0, num_layers - 2):
            self.models.append(BiLSTMLan(hidden_size * 2, hidden_size, num_heads))
        self.models.append(BiLSTMLan(hidden_size * 2, hidden_size, 1))
        self.loss = nn.NLLLoss(ignore_index=self.label_vocab.get_pad_id(), reduction='sum')
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
    def __init__(
            self,
            embedding_size: int = 128,
            hidden_size: int = 128,
            num_layers: int = 2,
            *args, **kwargs
    ):
        super(BiLSTMCrfNERModule, self).__init__(*args, **kwargs)
        self.embeddings = nn.Embedding(self.vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
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