import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import factory


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, act_fun, dropout_rate):
        super(NonLinear, self).__init__()
        self.act_fun = factory.act_func_factory(act_fun)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        return self.act_fun(self.dropout(self.linear(inputs)))


class MultiNonLinearLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, act_fun='relu', dropout_rate=0.1):
        super(MultiNonLinearLayer, self).__init__()
        self.num_layers = num_layers
        self.act_fun = act_fun
        self.dropout_rate = dropout_rate

        self.check_params()
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(NonLinear(input_size, hidden_size, self.act_fun[i], self.dropout_rate[i]))
            input_size = hidden_size

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def check_params(self):
        if isinstance(self.act_fun, str):
            self.act_fun = [self.act_fun] * (self.num_layers - 1)
        else:
            assert len(self.ct_fun) == (self.num_layers - 1), \
                'act_fun param must be str or list of str with num_layers length'

        if isinstance(self.dropout_rate, float):
            self.dropout_rate = [self.dropout_rate] * (self.num_layers - 1)
        else:
            assert len(
                self.dropout_rate) == (self.num_layers - 1), \
                'dropout_rate param must be float or list of float with num_layers length'


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, 'hidden_size 必须为 num_heads 的整数倍'
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.query_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.key_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())
        self.value_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU())

    def forward(self, query, key, value, masks=None, inference=False):
        """
        multi-head attention layer
        :param query: [batch_size, seq_len, hidden_size]
        :param key: [batch_size, seq_len, hidden_size]
        :param value: [batch_size, seq_len, hidden_size]
        :param masks: [batch_size, seq_len]
        :param inference: bool, ner(bilstm-lan)的解码层需要设为True， 其他场景一律设为False
        :return:
        """
        # [batch_size, seq_len, hidden_size]  [batch_size, num_labels, hidden_size]
        query_proj = self.query_proj(query)
        key_proj = self.key_proj(key)
        value_proj = self.value_proj(value)

        # [batch_size * num_heads, seq_len, hidden_size / num_heads]
        query_proj = torch.cat(torch.chunk(query_proj, self.num_heads, dim=2), 0)
        key_proj = torch.cat(torch.chunk(key_proj, self.num_heads, dim=2), 0)
        value_proj = torch.cat(torch.chunk(value_proj, self.num_heads, dim=2), 0)

        # [batch_size * num_head, seq_len, num_labels]
        alpha = torch.bmm(query_proj, torch.transpose(key_proj, 1, 2)) / (key_proj.size(-1) ** 0.5)

        if not inference:
            alpha = F.softmax(alpha, dim=-1)

        if masks is not None:
            masks = torch.unsqueeze(masks, -1).expand(-1, -1, alpha.size(-1)).repeat(self.num_heads, 1, 1)
            alpha = alpha * masks

        if inference:
            assert self.num_heads == 1, 'MultiHeadAttention用于最后一层推理时num_heads 必须为1'
            return alpha

        hiddens = torch.bmm(alpha, value_proj)
        hiddens = torch.cat(torch.chunk(hiddens, self.num_heads, 0), -1)

        return hiddens + query


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











