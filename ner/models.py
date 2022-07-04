import torch
import torch.nn as nn
from layers import MultiNonLinearLayer
from transformers import BertModel, BertPreTrainedModel, BertConfig


class BertMRC(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(BertMRC, self).__init__(config)
        self.bert = BertModel(config)
        self.start = nn.Linear(config.hidden_size, 1)
        self.end = nn.Linear(config.hidden_size, 1)
        self.cls = MultiNonLinearLayer(config.hidden_size * 2, config.intermediate_size, 1, 2)

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

