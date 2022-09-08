import torch
import torch.nn as nn
import torch.nn.functional as F
from module import IEModule
from kpe.vocab import KPESeq2SeqVocab
from layers import BahdanauAttention, Encoder


class PointGeneratorNetWork(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_input_size):
        super(PointGeneratorNetWork, self).__init__()
        self.linear = nn.Linear(encoder_hidden_size + decoder_hidden_size + decoder_input_size, 1)

    def forward(self, encoder_output, decoder_hidden, decoder_input):
        inputs = torch.cat([encoder_output, decoder_hidden, decoder_input], -1)
        outputs = self.linear(inputs)
        return torch.sigmoid(outputs)


class Decoder(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_input_size, decoder_hidden_size, attention_hidden_size,
                 is_coverage=True):
        super(Decoder, self).__init__()
        self.is_coverage = is_coverage
        self.lstm = nn.LSTMCell(decoder_input_size + encoder_hidden_size, decoder_hidden_size)
        self.attention = BahdanauAttention(decoder_hidden_size, encoder_hidden_size, attention_hidden_size, self.is_coverage)

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
        attention_outputs, attention_scores = self.attention(decoder_hidden[0], encoder_outputs, encoder_masks, coverage_inputs)
        decoder_hidden = self.lstm(torch.cat([attention_outputs, decoder_input], -1), decoder_hidden)
        return decoder_hidden, attention_outputs, attention_scores


class Seq2SeqKPEModule(IEModule):
    def __init__(
            self,
            embedding_size: int = 128,
            hidden_size: int = 128,
            num_layers: int = 1,
            decoder_max_steps: int = 20,
            decoder_sentence_num: int = 1,
            beam_size: int = 5,
            vocab_file: str = 'data/kpe/vocab.txt',
            do_lower: bool = False,
            is_coverage: bool = True,
            *args, **kwargs
    ):
        super(Seq2SeqKPEModule, self).__init__(*args, **kwargs)
        self.vocab = KPESeq2SeqVocab(vocab_file=vocab_file, do_lower=do_lower)

        self.is_coverage = is_coverage
        self.beam_size = beam_size
        self.decoder_max_steps = decoder_max_steps
        self.decoder_sentence_num = decoder_sentence_num
        self.embedding = nn.Embedding(self.vocab.get_vocab_size(), embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, embedding_size, hidden_size, hidden_size, is_coverage)
        self.pointer = PointGeneratorNetWork(hidden_size, hidden_size, embedding_size)
        self.linear = nn.Sequential(nn.Linear(hidden_size + hidden_size, hidden_size),
                                    nn.Linear(hidden_size, self.vocab.get_vocab_size()))

        if is_coverage:
            self.register_buffer('coverage_memery', torch.zeros(1, 1))

        self.register_buffer('start_ids', torch.LongTensor([self.vocab.get_start_id()]))
        self.register_buffer('vocab_extended', torch.zeros(1, 1))

    def forward(self, input_ids, input_masks, input_vocab_ids, oov_ids, target_ids=None):
        inputs = self.embedding(input_ids)
        encoder_outputs, encoder_hidden = self.encoder(inputs)  # [bs, ts, hs]
        decoder_hidden = (torch.cat(torch.chunk(encoder_hidden[0], 2, 0), -1).squeeze(0),
                          torch.cat(torch.chunk(encoder_hidden[1], 2, 0), -1).squeeze(0))  # [bs, hs]

        if target_ids is None:
            batch_seqs = self.beam_search_decode(encoder_outputs, decoder_hidden, input_masks, input_vocab_ids, oov_ids)
            return batch_seqs

        props = self.greedy_decode(encoder_outputs, decoder_hidden, input_masks, input_vocab_ids, oov_ids, target_ids)
        return props

    def greedy_decode(self, encoder_outputs, encoder_hidden, encoder_masks, encoder_vocab, oov_ids, target_ids=None):
        teacher_forcing = False if target_ids is None else True
        decoder_max_steps = self.decoder_max_steps if target_ids is None else target_ids.size(1)
        coverage_memery = self.coverage_memery.repeat(encoder_masks.size()) if self.is_coverage else None
        decoder_input_ids = self.start_ids.repeat(encoder_outputs.size(0))

        decoder_hidden = encoder_hidden
        probs = []

        for i in range(decoder_max_steps):
            p, decoder_hidden, coverage_memery = self.decode(decoder_hidden, decoder_input_ids,
                                                             encoder_masks, encoder_outputs, encoder_vocab,
                                                             oov_ids, coverage_memery)
            decoder_input_ids = target_ids[:, i] if teacher_forcing else p.argmax(-1)
            probs.append(p.unsqueeze(1))

        return torch.cat(probs, 1)

    @staticmethod
    def batch_expand(tensor, beam_size):
        batch_size = tensor.size(0)
        chunks = torch.chunk(tensor, batch_size)
        expand_sizes = (beam_size, ) + (-1, ) * (tensor.dim() - 1)
        chunks = [chunk.expand(expand_sizes) for chunk in chunks]
        return torch.cat(chunks, 0)

    def beam_search_decode(self, encoder_outputs, encoder_hidden, encoder_masks, encoder_vocab, oov_ids):
        batch_size = encoder_outputs.size(0)
        coverage_memery = self.coverage_memery.repeat(1, encoder_masks.size(1)) if self.is_coverage else None
        next_probs, decoder_hidden, coverage_memery = self.decode(encoder_hidden,
                                                                  self.start_ids.repeat(batch_size),
                                                                  encoder_masks, encoder_outputs,
                                                                  encoder_vocab, oov_ids,
                                                                  coverage_memery)

        vocab_size = next_probs.size(-1)
        probs, next_ids = torch.topk(next_probs, self.beam_size)  # [bs, bw]
        outputs = next_ids.reshape(-1, 1)

        encoder_outputs = self.batch_expand(encoder_outputs, self.beam_size)
        encoder_masks = self.batch_expand(encoder_masks, self.beam_size)
        encoder_vocab = self.batch_expand(encoder_vocab, self.beam_size)
        oov_ids = self.batch_expand(oov_ids, self.beam_size)

        decoder_hidden = (self.batch_expand(decoder_hidden[0], self.beam_size), self.batch_expand(decoder_hidden[1], self.beam_size))
        coverage_memery = self.batch_expand(coverage_memery, self.beam_size)

        for i in range(self.decoder_max_steps - 1):
            decoder_input_ids = torch.where(outputs[:, -1] >= self.vocab.get_vocab_size(), outputs[:, -1], torch.LongTensor([self.vocab.get_unk_id()]))
            next_probs, decoder_hidden, coverage_memery = self.decode(decoder_hidden,
                                                                      decoder_input_ids,
                                                                      encoder_masks, encoder_outputs,
                                                                      encoder_vocab, oov_ids,
                                                                      coverage_memery)
            # [bs * bw, 1] + [bs * bw, vs] => [bs, bw, vs]
            next_probs = next_probs.reshape(-1, self.beam_size, next_probs.size(-1))
            probs = probs.unsqueeze(-1) + next_probs
            probs = probs.flatten(start_dim=1)
            probs, idx = torch.topk(probs, self.beam_size)  # [bs, bw]
            next_ids = torch.remainder(idx, vocab_size).flatten().unsqueeze(-1)
            prev_indices = (idx / vocab_size).long()
            prev_indices = prev_indices + torch.arange(batch_size).unsqueeze(-1) * self.beam_size
            prev_ids = outputs[prev_indices].flatten(end_dim=-2)
            outputs = torch.cat([prev_ids, next_ids], 1)
            decoder_hidden = (decoder_hidden[0][prev_indices].flatten(end_dim=-2), decoder_hidden[1][prev_indices].flatten(end_dim=-2))
            coverage_memery = coverage_memery[prev_indices].flatten(end_dim=-2)

        probs, idx = torch.topk(probs, 1)
        prev_indices = (idx / vocab_size).long()
        prev_indices = prev_indices + torch.arange(batch_size).unsqueeze(-1) * self.beam_size

        return outputs[prev_indices].flatten(end_dim=-2)

    def decode(self, decoder_hidden, decoder_input_ids, encoder_masks, encoder_outputs, encoder_vocab, oov_ids, coverage_memery):
        """

        Args:
            decoder_hidden:
            decoder_input_ids:
            encoder_masks:
            encoder_outputs:
            encoder_vocab:
            oov_ids:
            coverage_memery:

        Returns:
            p: [bs, vs]
        """
        decoder_input = self.embedding(decoder_input_ids)
        decoder_hidden, attention_output, attention_scores = self.decoder(encoder_outputs, decoder_input,
                                                                          decoder_hidden, encoder_masks,
                                                                          coverage_memery)
        if self.is_coverage:
            assert coverage_memery is not None, '引入coverage时, coverage_memory 不能为空'
            coverage_memery = coverage_memery + attention_scores
        p_vocab = self.linear(torch.cat([decoder_hidden[0], attention_output], -1))
        p_vocab = F.softmax(p_vocab, -1)
        p_vocab_extended = torch.cat([p_vocab, self.vocab_extended.repeat(oov_ids.size())], -1)
        p_gen = self.pointer(attention_output, decoder_hidden[0], decoder_input)
        p = p_gen * p_vocab_extended
        p = p.scatter_add(1, encoder_vocab, (1 - p_gen) * attention_scores)
        return p, decoder_hidden, coverage_memery

    def get_training_outputs(self, batch):
        ids, input_ids, input_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx, targets, target_ids, target_masks = batch
        idx2oov = [{v: k for k, v in oov.items()} for oov in oov2idx]
        props = self(input_ids, input_masks, input_vocab_ids, oov_ids, target_ids)
        loss = torch.gather(props, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(loss + 1e-8) * target_masks
        loss = loss.sum() / loss.size(0)
        pred_ids = props.argmax(-1).detach().cpu().numpy().tolist()
        seqs = self.get_batch_seqs(pred_ids, self.vocab.get_vocab(), idx2oov)
        return seqs, targets, input_masks, loss

    def get_validation_outputs(self, batch):
        ids, input_ids, input_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx, targets, target_ids, target_masks = batch
        pred_ids = self(input_ids, input_masks, input_vocab_ids, oov_ids)
        idx2oov = [{v: k for k, v in oov.items()} for oov in oov2idx]
        seqs = self.get_batch_seqs(pred_ids.detach().cpu().numpy().tolist(), self.vocab.get_vocab(), idx2oov)
        return seqs, targets, input_masks, torch.FloatTensor(0)

    def get_predict_outputs(self, batch):
        ids, input_ids, input_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx = batch
        pred_ids = self(input_ids, input_masks, input_vocab_ids, oov_ids)
        idx2oov = [{v: k for k, v in oov.items()} for oov in oov2idx]
        seqs = self.get_batch_seqs(pred_ids.detach().cpu().numpy().tolist(), self.vocab.get_vocab(), idx2oov)
        return ids, seqs

    def get_seqs(self, input_ids, idx2word, idx2oov):
        seqs = []
        for input_id in input_ids:
            if input_id == self.vocab.get_end_id():
                break
            seqs.append(idx2word[input_id] if input_id < len(idx2word) else idx2oov[input_id])

        return ''.join(seqs).split(self.vocab.get_vertical_token())

    def get_batch_seqs(self, pred_ids, idx2word, idx2oovs):
        seqs = []
        for pred_id, idx2oov in zip(pred_ids, idx2oovs):
            seq = self.get_seqs(pred_id, idx2word, idx2oov)
            seqs.append(seq)
        return seqs

    def get_f1_stats(self, preds, targets, masks=None):
        pred_num = 0
        gold_num = 0
        correct_num = 0
        for pred, target in zip(preds, targets):
            pred_num += len(pred)
            gold_num += len(target)
            correct_num += len((set(pred).intersection(set(target))))
        return correct_num, pred_num - correct_num, gold_num - correct_num
