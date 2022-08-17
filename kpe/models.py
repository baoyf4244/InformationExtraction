import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from tokenization import KPEChineseCharTokenizer
from pytorch_lightning import LightningModule

torch.autograd.set_detect_anomaly(True)


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
        return torch.sigmoid(outputs)


class Coverage(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, hidden_size):
        super(Coverage, self).__init__()
        self.query = nn.Linear(decoder_hidden_size, hidden_size)
        self.key = nn.Linear(encoder_hidden_size, hidden_size)
        self.memery = nn.Linear(1, hidden_size)
        self.coverage = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_inputs, encoder_masks, coverage_inputs):
        """

        Args:
            coverage_inputs: [bs, ts]
            encoder_outputs: [bs, ts, ehs]
            decoder_inputs: [bs, dhs]
            encoder_masks: [bs, ts]  [[0, 0, ... , 1, 1, 1], ..., [...]]

        Returns:

        """
        _, seq_len, _ = encoder_outputs.size()
        if decoder_inputs.dim() == 2:
            decoder_inputs = torch.unsqueeze(decoder_inputs, 1).expand(-1, seq_len, -1)

        query = self.query(decoder_inputs)
        key = self.key(encoder_outputs)
        memery = self.memery(coverage_inputs.unsqueeze(-1))

        coverage_scores = self.coverage(F.tanh(query + key + memery)).squeeze()  # [bs, ts]
        coverage_scores = torch.masked_fill(coverage_scores, encoder_masks, float('-inf'))
        coverage_scores = F.softmax(coverage_scores, -1)
        coverage_outputs = torch.bmm(coverage_scores.unsqueeze(1), encoder_outputs)  # [bs, 1, ehs]
        return coverage_outputs.squeeze(), coverage_scores


class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_input_size, decoder_hidden_size, attention_hidden_size,
                 coverage=True):
        super(Decoder, self).__init__()
        self.coverage = coverage
        self.lstm = nn.LSTMCell(decoder_input_size + encoder_hidden_size, decoder_hidden_size)
        if coverage:
            self.attention = Coverage(encoder_hidden_size, decoder_hidden_size, attention_hidden_size)
        else:
            self.attention = BahdanauAttention(decoder_hidden_size, encoder_hidden_size, attention_hidden_size)

    def forward(self, encoder_outputs, decoder_input, decoder_hidden, encoder_masks, coverage_inputs=None):
        """

        Args:
            encoder_outputs:
            decoder_input: y_i-1
            decoder_hidden: (h, c)
            encoder_masks:
            coverage_inputs: self.coverage为true时传参, [bs, ehs]
        Returns:

        """
        if self.coverage:
            attention_outputs, attention_scores = self.attention(encoder_outputs, decoder_hidden[0], encoder_masks, coverage_inputs)
        else:
            attention_outputs, attention_scores = self.attention(decoder_hidden[0], encoder_outputs, encoder_masks)
        decoder_hidden = self.lstm(torch.cat([attention_outputs, decoder_input], -1), decoder_hidden)
        return decoder_hidden, attention_outputs, attention_scores


class Seq2SeqKPEModule(LightningModule):
    def __init__(self, embedding_size, hidden_size, num_layers, decoder_max_steps, beam_size,
                 vocab_file, coverage=True):
        super(Seq2SeqKPEModule, self).__init__()
        self.coverage = coverage
        self.beam_size = beam_size
        self.tokenizer = KPEChineseCharTokenizer(vocab_file)
        self.decoder_max_steps = decoder_max_steps
        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(), embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, embedding_size, hidden_size, hidden_size, coverage)
        self.pointer = PointGeneratorNetWork(hidden_size, hidden_size, embedding_size)
        if coverage:
            self.register_buffer('coverage_memery', torch.zeros(1, 1))

        self.register_buffer('start_ids', torch.zeros(1, 1))
        self.register_buffer('vocab_extended', torch.zeros(1, 1))

        self.linear = nn.Sequential(nn.Linear(hidden_size + hidden_size, hidden_size),
                                    nn.Linear(hidden_size, self.tokenizer.get_vocab_size()))

    def decode(self, encoder_outputs, encoder_masks, encoder_vocab,
               oov_ids, decoder_input_ids, decoder_hidden, coverage_memery):
        """

        Args:
            decoder_input_ids:
            encoder_outputs:
            encoder_masks:
            encoder_vocab: 将 UNK 替换为oov_id后的input_ids
            oov_ids:
            decoder_hidden:
            coverage_memery:

        Returns:

        """
        encoder_masks = torch.logical_not(encoder_masks)
        decoder_input = self.embedding(decoder_input_ids)
        decoder_hidden, attention_output, attention_scores = self.decoder(encoder_outputs, decoder_input,
                                                                          decoder_hidden, encoder_masks, coverage_memery)
        if self.coverage:
            coverage_memery = coverage_memery + attention_scores
        p_vocab = self.linear(torch.cat([decoder_hidden[0], attention_output], -1))
        p_vocab = F.softmax(p_vocab, -1)
        p_vocab_extended = torch.cat([p_vocab, self.vocab_extended.repeat(oov_ids.size())], -1)
        p_gen = self.pointer(attention_output, decoder_hidden[0], decoder_input)
        p = p_gen * p_vocab_extended
        p = p.scatter_add(1, encoder_vocab, (1 - p_gen) * attention_scores)

        return p, decoder_hidden, coverage_memery

    def forward(self, input_ids, input_masks, input_vocab_ids, oov_ids, target_ids):
        inputs = self.embedding(input_ids)
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_hidden = (torch.cat(torch.chunk(encoder_hidden[0], 2, 0), -1).squeeze(),
                          torch.cat(torch.chunk(encoder_hidden[1], 2, 0), -1).squeeze())

        if self.coverage:
            coverage_memery = self.coverage_memery.repeat(input_ids.size())
        else:
            coverage_memery = None

        decoder_max_steps = target_ids.size(1)

        props = []
        for i in range(decoder_max_steps):
            decoder_input_ids = target_ids[:, i]
            p, decoder_hidden, coverage_memery = self.decode(encoder_outputs, input_masks, input_vocab_ids, oov_ids,
                                                             decoder_input_ids, decoder_hidden, coverage_memery)
            props.append(p.unsqueeze(1))
        return torch.cat(props, 1)

    def compute_step_stats(self, batch, stage):
        ids, targets, input_ids, input_masks, target_ids, target_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx = batch
        idx2oov = [{v: k for k, v in oov.items()} for oov in oov2idx]
        props = self(input_ids, input_masks, input_vocab_ids, oov_ids, target_ids)
        seqs = self.get_batch_seqs(props, self.tokenizer.get_vocab(), idx2oov)
        # loss = torch.gather(props, -1, target_ids.unsqueeze(-1)).squeeze()
        # loss = -torch.log(loss + 1e-8) * target_masks
        loss = F.cross_entropy(props.view(-1, props.size(-1)), target_ids.view(-1), reduction='none')
        loss = loss.view(-1, target_ids.size(1)) * target_masks
        loss = loss.sum() / loss.size(0)
        pred_num, gold_num, correct_num = self.get_f1_stats(seqs, targets)
        recall, precision, f1_score = self.get_f1_score(pred_num, gold_num, correct_num)

        logs = {
            stage + '_loss': loss,
            stage + '_pred_num': pred_num,
            stage + '_gold_num': gold_num,
            stage + '_correct_num': correct_num,
            stage + '_recall': recall,
            stage + '_precision': precision,
            stage + '_f1_score': f1_score
        }
        if stage == 'train':
            logs['loss'] = logs['train_loss']

        self.log_dict(logs, prog_bar=True)
        return logs

    def compute_epoch_states(self, outputs, stage='val'):
        pred_num = torch.Tensor([output[stage + '_pred_num'] for output in outputs]).sum()
        gold_num = torch.Tensor([output[stage + '_gold_num'] for output in outputs]).sum()
        correct_num = torch.Tensor([output[stage + '_correct_num'] for output in outputs]).sum()
        loss = torch.stack([output[stage + '_loss'] for output in outputs]).mean()

        recall, precision, f1 = self.get_f1_score(pred_num, gold_num, correct_num)

        logs = {
            stage + '_pred_num': pred_num,
            stage + '_gold_num': gold_num,
            stage + '_correct_num': correct_num,
            stage + '_loss': loss,
            stage + '_f1_score': f1,
            stage + '_recall': recall,
            stage + '_precision': precision
        }

        self.log_dict(logs)

    def training_step(self, batch, batch_idx):
        logs = self.compute_step_stats(batch, 'train')
        return logs

    def training_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        logs = self.compute_step_stats(batch, 'val')
        return logs

    def validation_epoch_end(self, outputs):
        self.compute_epoch_states(outputs, 'val')

    def get_seqs(self, input_ids, idx2word, idx2oov):
        seqs = []
        for input_id in input_ids:
            if input_id == self.tokenizer.get_end_id():
                break
            seqs.append(idx2word[input_id] if input_id < len(idx2word) else idx2oov[input_id])

        return seqs

    def get_batch_seqs(self, props, idx2word, idx2oovs):
        pred_ids = props.argmax(-1).detach().cpu().numpy().tolist()
        seqs = []
        for pred_id, idx2oov in zip(pred_ids, idx2oovs):
            seq = self.get_seqs(pred_id, idx2word, idx2oov)
            seqs.append(seq)
        return seqs

    @staticmethod
    def get_f1_stats(preds, targets):
        pred_num = 0
        gold_num = 0
        correct_num = 0
        for pred, target in zip(preds, targets):
            pred = ''.join(pred).split('|')
            pred_num += len(pred)
            gold_num += len(target)
            correct_num += len((set(pred).intersection(set(target))))
        return pred_num, gold_num, correct_num

    @staticmethod
    def get_f1_score(pred_num, gold_num, correct_num):
        recall = correct_num / (gold_num + 1e-8)
        precision = correct_num / (pred_num + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        return recall, precision, f1


