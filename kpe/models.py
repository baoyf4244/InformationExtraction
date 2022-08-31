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
            beam_size: int = 5,
            vocab_file: str = 'data/kpe/vocab.txt',
            do_lower: bool = False,
            is_coverage: bool = True
    ):
        super(Seq2SeqKPEModule, self).__init__()
        self.vocab = KPESeq2SeqVocab(vocab_file=vocab_file, do_lower=do_lower)

        self.is_coverage = is_coverage
        self.beam_size = beam_size
        self.decoder_max_steps = decoder_max_steps
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

    def forward(self, input_ids, input_masks, input_vocab_ids, oov_ids, target_ids):
        inputs = self.embedding(input_ids)
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoder_hidden = (torch.cat(torch.chunk(encoder_hidden[0], 2, 0), -1).squeeze(0),
                          torch.cat(torch.chunk(encoder_hidden[1], 2, 0), -1).squeeze(0))

        if self.is_coverage:
            coverage_memery = self.coverage_memery.repeat(input_ids.size())
        else:
            coverage_memery = None

        decoder_max_steps = target_ids.size(1)

        props = []
        for i in range(decoder_max_steps):
            if i == 0:
                decoder_input_ids = self.start_ids.repeat([target_ids.size(0)])
            else:
                decoder_input_ids = target_ids[:, i]
            p, decoder_hidden, coverage_memery = self.decode(encoder_outputs, input_masks, input_vocab_ids, oov_ids,
                                                             decoder_input_ids, decoder_hidden, coverage_memery)
            props.append(p.unsqueeze(1))
        return torch.cat(props, 1)

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
        decoder_input = self.embedding(decoder_input_ids)
        decoder_hidden, attention_output, attention_scores = self.decoder(encoder_outputs, decoder_input,
                                                                          decoder_hidden, encoder_masks, coverage_memery)
        if self.is_coverage:
            coverage_memery = coverage_memery + attention_scores
        p_vocab = self.linear(torch.cat([decoder_hidden[0], attention_output], -1))
        p_vocab = F.softmax(p_vocab, -1)
        p_vocab_extended = torch.cat([p_vocab, self.vocab_extended.repeat(oov_ids.size())], -1)
        p_gen = self.pointer(attention_output, decoder_hidden[0], decoder_input)
        p = p_gen * p_vocab_extended
        p = p.scatter_add(1, encoder_vocab, (1 - p_gen) * attention_scores)

        return p, decoder_hidden, coverage_memery

    def get_training_outputs(self, batch):
        ids, targets, input_ids, input_masks, target_ids, target_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx = batch
        idx2oov = [{v: k for k, v in oov.items()} for oov in oov2idx]
        props = self(input_ids, input_masks, input_vocab_ids, oov_ids, target_ids)
        loss = torch.gather(props, -1, target_ids.unsqueeze(-1)).squeeze()
        loss = -torch.log(loss + 1e-8) * target_masks
        # loss = F.cross_entropy(props.view(-1, props.size(-1)), target_ids.view(-1), reduction='none')
        # loss = loss.view(-1, target_ids.size(1)) * target_masks
        loss = loss.sum() / loss.size(0)
        seqs = self.get_batch_seqs(props, self.vocab.get_vocab(), idx2oov)
        return seqs, targets, input_masks, loss

    def get_seqs(self, input_ids, idx2word, idx2oov):
        seqs = []
        for input_id in input_ids:
            if input_id == self.vocab.get_end_id():
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

    def get_f1_stats(self, preds, targets, masks=None):
        pred_num = 0
        gold_num = 0
        correct_num = 0
        for pred, target in zip(preds, targets):
            pred = ''.join(pred).split(self.vocab.get_vertical_token())
            target = ''.join(target).split(self.vocab.get_vertical_token())
            pred_num += len(pred)
            gold_num += len(target)
            correct_num += len((set(pred).intersection(set(target))))
        return pred_num, gold_num, correct_num



