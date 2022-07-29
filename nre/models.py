import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from layers import Encoder, Decoder, BahdanauAttention
from tokenization import WhiteSpaceTokenizer


class Seq2SeqNREModule(LightningModule):
    def __init__(self, vocab_file, embedding_size, hidden_size, num_layers, decoder_max_steps):
        super(Seq2SeqNREModule, self).__init__()
        self.decoder_max_steps = decoder_max_steps
        self.tokenizer = WhiteSpaceTokenizer(vocab_file)
        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(), embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers)
        self.decoder = Decoder(embedding_size, hidden_size, self.tokenizer.get_vocab_size())
        self.register_buffer('h0', torch.zeros(1, hidden_size, dtype=torch.float))
        self.register_buffer('c0', torch.zeros(1, hidden_size, dtype=torch.float))
        self.register_buffer('start_ids', torch.LongTensor([[self.tokenizer.get_start_id()]]))

    def forward(self, input_ids, input_masks, input_vocab_maks, target_ids, inference=False):
        encoder_embeddings = self.embedding(input_ids)
        encoder_outputs = self.encoder(encoder_embeddings)

        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = (self.h0.expand(batch_size, -1), self.c0.expand(batch_size, -1))
        if inference:
            decoder_max_steps = self.decoder_max_steps
        else:
            decoder_max_steps = target_ids.size(1)

        outputs = []
        att_scores = []
        for i in range(decoder_max_steps):
            if i == 0:
                decoder_input_id = self.start_ids.repeat(batch_size, 1)
            elif inference:
                decoder_input_id = outputs[-1].squeeze().argmax(-1)
            else:
                decoder_input_id = target_ids[:, i]
            decoder_input = self.embedding(decoder_input_id)
            output, hidden, att_score = self.decoder(encoder_outputs, input_masks, decoder_input, hidden)
            if inference:
                output = torch.masked_fill(output, input_vocab_maks, float('-inf'))
            outputs.append(output.unsqueeze(1))
            att_scores.append(att_score.argmax(1, keepdim=True))

        return torch.cat(outputs, 1), torch.cat(att_scores, 1)  # [bs, ts, vs], [bs, ts]/代表src_inputs的位置

    def get_results(self, pred_ids, att_scores, tokens):
        preds = []
        for i, pred_id in enumerate(pred_ids):
            if pred_id.item() == self.tokenizer.get_unk_id():
                if att_scores[i] < len(tokens):
                    pred = tokens[att_scores[i]]
                else:
                    pred = self.tokenizer.get_pad_token()
            else:
                pred = self.tokenizer.convert_id_to_tokens(pred_id)

            if pred_id.item() == self.tokenizer.get_end_id():
                break

            preds.append(pred)
        return preds

    def get_batch_results(self, pred_ids, att_scores, tokens):
        preds = []
        for pred_id, att_score, token in zip(pred_ids, att_scores, tokens):
            preds.append(self.get_results(pred_id, att_score, token))
        return preds

    def get_f1_stats(self, preds, targets):
        pred_num = 0
        gold_num = 0
        correct_num = 0
        for pred, target in zip(preds, targets):
            pred_num += len(pred)
            gold_num += len(target)

            for p, t in zip(pred, target):
                if p == t:
                    correct_num += 1

        return pred_num, gold_num, correct_num

    def get_f1_score(self, pred_num, gold_num, correct_num):
        recall = correct_num / (1e-8 + gold_num)
        precision = correct_num / (1e-8 + pred_num)
        f1_score = 2 * recall * precision / (recall + precision + 1e-8)
        return recall, precision, f1_score

    def training_step(self, batch):
        tokens, targets, input_ids, input_masks, input_vocab_masks, target_ids = batch
        outputs, att_scores = self(input_ids, input_masks, input_vocab_masks, target_ids)
        pred_ids = outputs.argmax(-1)
        preds = self.get_batch_results(pred_ids, att_scores, tokens)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), target_ids.view(-1), reduction='none')
        loss = loss.sum() / target_ids.size(0)
        pred_num, gold_num, correct_num = self.get_f1_stats(preds, targets)
        recall, precision, f1_score = self.get_f1_score(pred_num, gold_num, correct_num)

        logs = {
            'loss': loss,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score
        }
        self.log_dict(logs, prog_bar=True, on_epoch=True)
        return logs


class PTRDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_labels):
        """

        Args:
            input_size: tgt input size
            hidden_size: src, tgt hidden size
            num_layers:
            num_labels:
        """
        super(PTRDecoder, self).__init__()
        self.lstm = nn.LSTMCell(input_size + hidden_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size, hidden_size)

        self.head_lstm = nn.LSTM(2 * hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.head_start_cls = nn.Linear(2 * hidden_size, 1)
        self.head_end_cls = nn.Linear(2 * hidden_size, 1)

        self.tail_lstm = nn.LSTM(4 * hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.tail_start_cls = nn.Linear(2 * hidden_size, 1)
        self.tail_end_cls = nn.Linear(2 * hidden_size, 1)

        self.relation_cls = nn.Linear(8 * hidden_size + input_size, num_labels)

    def forward(self, encoder_outputs, encoder_masks, decoder_input, decoder_hidden):
        att_output, att_scores = self.attention(decoder_input, encoder_outputs, encoder_masks)
        decoder_output, hidden = self.lstm(torch.cat([decoder_input, att_output], -1), decoder_hidden)

        head_inputs = torch.cat([encoder_outputs, decoder_output.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)],
                                -1)
        head_outputs, _ = self.head_lstm(head_inputs)  # [bs, ts, 2 * hs]
        head_start_logits = self.head_start_cls(head_outputs).squeeze()  # [bs, ts]
        head_start_logits = torch.masked_fill(head_start_logits, encoder_masks, float('-inf'))
        head_end_logits = self.head_end_cls(head_outputs).squeeze()
        head_end_logits = torch.masked_fill(head_end_logits, encoder_masks, float('-inf'))

        tail_inputs = torch.cat([head_inputs, head_outputs], -1)
        tail_outputs, _ = self.tail_lstm(tail_inputs)
        tail_start_logits = self.tail_start_cls(tail_outputs).squeeze()
        tail_start_logits = torch.masked_fill(tail_start_logits, encoder_masks, float('-inf'))
        tail_end_logits = self.tail_end_cls(tail_outputs).squeeze()
        tail_end_logits = torch.masked_fill(tail_end_logits, encoder_masks, float('-inf'))

        entity_outputs = torch.cat([torch.bmm(F.softmax(head_start_logits, -1).unsqueeze(1), head_outputs).squeeze(),
                                    torch.bmm(F.softmax(head_end_logits, -1).unsqueeze(1), head_outputs).squeeze(),
                                    torch.bmm(F.softmax(tail_start_logits, -1).unsqueeze(1), tail_outputs).squeeze(),
                                    torch.bmm(F.softmax(tail_end_logits, -1).unsqueeze(1), tail_outputs).squeeze()],
                                   -1)
        relation_inputs = torch.cat([entity_outputs, decoder_output], -1)

        relation_logits = self.relation_cls(relation_inputs)

        return head_start_logits, head_end_logits, tail_start_logits, tail_end_logits, relation_logits, (decoder_output, hidden), entity_outputs


class Seq2SeqPTRNREModule(LightningModule):
    def __init__(self, vocab_size, label_size, embedding_size, hidden_size, num_layers, max_decoder_step):
        super(Seq2SeqPTRNREModule, self).__init__()
        self.max_decode_step = max_decoder_step
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.label_embedding = nn.Embedding(label_size, embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers)
        self.decoder = PTRDecoder(hidden_size, hidden_size, num_layers, label_size)
        self.linear = nn.Linear(8 * hidden_size + embedding_size, hidden_size)  # 用于 decoder input 降维

        self.register_buffer('h0', torch.zeros(1, hidden_size, dtype=torch.float))
        self.register_buffer('c0', torch.zeros(1, hidden_size, dtype=torch.float))
        self.register_buffer('start_decoder_input', torch.zeros(1, 8 * hidden_size + embedding_size, dtype=torch.float))

    def forward(self, input_ids, input_masks, target_ids=None, inference=False):
        batch_size, encoder_step = input_ids.size()
        encoder_inputs = self.embedding(input_ids)
        encoder_outputs = self.encoder(encoder_inputs)

        if inference:
            decoder_step = self.max_decode_step
        else:
            decoder_step = target_ids.size(1)

        hidden = (self.h0.repeat(batch_size, 1), self.c0.repeat(batch_size, 1))
        decoder_input = self.start_decoder_input.repeat(batch_size, 1)
        head_start_logits, head_end_logits, tail_start_logits, tail_end_logits, relation_logits, entity_outputs = [], [], [], [], [], []
        for i in range(0, decoder_step):
            if i == 0:
                decoder_input = decoder_input
            elif inference:
                decoder_input = decoder_input + torch.cat([entity_outputs[-1], self.label_embedding(relation_logits[-1].argmax(-1))])
            else:
                decoder_input = decoder_input + torch.cat([entity_outputs[-1], self.label_embedding(target_ids[:, i])], -1)

            head_start_logit, head_end_logit, tail_start_logit, tail_end_logit, relation_logit, hidden, \
            entity_output = self.decoder(encoder_outputs, input_masks, self.linear(decoder_input), hidden)

            head_start_logits.append(head_start_logit)
            head_end_logits.append(head_end_logit)
            tail_start_logits.append(tail_start_logit)
            tail_end_logits.append(tail_end_logit)
            relation_logits.append(relation_logit)
            entity_outputs.append(entity_output)

        head_start_logits = torch.cat([logit.unsqueeze(1) for logit in head_start_logits], 1)
        head_end_logits = torch.cat([logit.unsqueeze(1) for logit in head_end_logits], 1)
        tail_start_logits = torch.cat([logit.unsqueeze(1) for logit in tail_start_logits], 1)
        tail_end_logits = torch.cat([logit.unsqueeze(1) for logit in tail_end_logits], 1)
        relation_logits = torch.cat([logit.unsqueeze(1) for logit in relation_logits], 1)

        return head_start_logits, head_end_logits, tail_start_logits, tail_end_logits, relation_logits

    @staticmethod
    def get_result(head_start_offsets, head_end_offsets, tail_start_offsets, tail_end_offsets, relation_ids):
        triples = set()
        for i in range(len(head_start_offsets)):
            triple = (head_start_offsets[i], head_end_offsets[i], tail_start_offsets[i],
                      tail_end_offsets[i], relation_ids[i])

            triples.add(triple)
        return triples

    def get_batch_results(self, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids, relation_ids):
        triples = []
        if head_start_ids.dim() == 3:
            head_start_ids = head_start_ids.argmax(-1)
            head_end_ids = head_end_ids.argmax(-1)
            tail_start_ids = tail_start_ids.argmax(-1)
            tail_end_ids = tail_end_ids.argmax(-1)
            relation_ids = relation_ids.argmax(-1)

        for i in range(len(head_start_ids)):
            triples.append(self.get_result(head_start_ids[i], head_end_ids[i], tail_start_ids[i], tail_end_ids[i], relation_ids[i]))

        return triples

    @staticmethod
    def get_f1_stats(preds, targets):
        pred_num = 0
        gold_num = 0
        correct_num = 0
        for pred, target in zip(preds, targets):
            pred_num += len(pred)
            gold_num += len(target)
            for p in pred:
                if p in target:
                    correct_num += 1
        return pred_num, gold_num, correct_num

    def get_f1_score(self, preds, targets):
        pred_num, gold_num, correct_num = self.get_f1_stats(preds, targets)
        recall = correct_num / (gold_num + 1e-8)
        precision = correct_num / (pred_num + 1e-8)
        f1_score = 2 * recall + precision / (recall + precision + 1e-8)
        return precision, recall, f1_score

    def training_step(self, batch):
        input_ids, input_masks, head_start_ids, head_end_ids, tail_start_ids, tail_end_ids, target_ids, target_masks = batch
        head_start_logits, head_end_logits, tail_start_logits, tail_end_logits, relation_logits = self(input_ids, input_masks, target_ids, False)
        head_start_loss = F.cross_entropy(head_start_logits.view(-1, head_start_logits.size(-1)), head_start_ids.view(-1), reduction='sum', ignore_index=-1)
        head_end_loss = F.cross_entropy(head_end_logits.view(-1, head_end_logits.size(-1)), head_end_ids.view(-1), reduction='sum', ignore_index=-1)
        tail_start_loss = F.cross_entropy(tail_start_logits.view(-1, tail_start_logits.size(-1)), tail_start_ids.view(-1), reduction='sum', ignore_index=-1)
        tail_end_loss = F.cross_entropy(tail_end_logits.view(-1, tail_end_logits.size(-1)), tail_end_ids.view(-1), reduction='sum', ignore_index=-1)
        relation_loss = F.cross_entropy(relation_logits.view(-1, relation_logits.size(-1)), target_ids.view(-1), reduction='sum', ignore_index=0)

        loss = (head_start_loss + head_end_loss + tail_start_loss + tail_end_loss + relation_loss) / input_ids.size(0)

        pred_triples = self.get_batch_results(head_start_logits, head_end_logits, tail_start_logits,
                                              tail_end_logits, relation_logits)
        target_triples = self.get_batch_results(head_start_ids, head_end_ids, tail_start_ids, tail_end_ids, target_ids)
        precision, recall, f1_score = self.get_f1_score(pred_triples, target_triples)

        logs = {
            'loss': loss,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score
        }
        self.log_dict(logs, prog_bar=True, on_epoch=True)
        return logs




