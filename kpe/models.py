import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        assert hidden_size % 2 == 0, '双向LSTM的输出维度必须为偶数'
        self.lstm = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers, bidirectional=True, batch_first=True)

    def forward(self, inputs):
        outputs, hidden = self.lstm(inputs)
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, query_input_size, key_input_size, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.key = nn.Linear(key_input_size, hidden_size)
        self.query = nn.Linear(query_input_size, hidden_size)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, query, key, masks):
        if query.dim() == 2:
            query = torch.unsqueeze(query, 1).expand(-1, key.size(1), -1)

        query_proj = self.query(query)
        key_proj = self.key(key)

        score = self.attention(torch.tanh(query_proj + key_proj))  # [bs, ts, 1]
        score = torch.squeeze(score)
        score = torch.masked_fill(score, masks, float('-inf'))
        score = F.softmax(score, -1)  # [bs, ts]

        outputs = torch.bmm(torch.unsqueeze(score, 1), key).squeeze()  # [bs, input_size]
        return outputs, score  # [bs, input_size], [bs, ts]


class PointGeneratorNetWork(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_input_size):
        super(PointGeneratorNetWork, self).__init__()
        self.linear = nn.Linear(encoder_hidden_size + decoder_hidden_size + decoder_input_size, 1)

    def forward(self, encoder_output, decoder_hidden, decoder_input):
        inputs = torch.cat([encoder_output, decoder_hidden, decoder_input], -1)
        outputs = self.linear(inputs)
        return F.sigmoid(outputs)


class Coverage(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, hidden_size):
        super(Coverage, self).__init__()
        self.query = nn.Linear(decoder_hidden_size, hidden_size)
        self.key = nn.Linear(encoder_hidden_size, hidden_size)
        self.memery = nn.Linear(1, hidden_size)
        self.coverage = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_inputs, coverage_inputs, encoder_masks):
        """

        Args:
            encoder_outputs: [bs, ts, ehs]
            decoder_inputs: [bs, dhs]
            coverage_inputs: [bs, ts]

        Returns:

        """
        _, seq_len, _ = encoder_outputs.size()
        if decoder_inputs.dim() == 2:
            decoder_inputs = torch.unsqueeze(decoder_inputs, 1).expand(-1, seq_len, -1)

        if coverage_inputs.dim() == 2:
            coverage_inputs = torch.unsqueeze(coverage_inputs, -1)

        query = self.query(decoder_inputs)
        key = self.key(decoder_inputs)
        memery = self.memery(coverage_inputs)

        coverage_scores = self.coverage(F.tanh(query + key + memery)).squeeze()  # [bs, ts]
        coverage_scores = torch.masked_fill(coverage_scores, encoder_masks, float('-inf'))
        coverage_scores = F.softmax(coverage_scores, -1)
        coverage_outputs = torch.bmm(coverage_scores.unsqueeze(1), encoder_outputs)
        return coverage_outputs.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)


class Seq2SeqKPEModule(LightningModule):
    def __init__(self):
        super(Seq2SeqKPEModule, self).__init__()