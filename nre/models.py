import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from layers import Encoder, Decoder
from tokenization import WhiteSpaceTokenizer


class Seq2SeqNREModule(LightningModule):
    def __init__(self, vocab_file, embedding_size, hidden_size, num_layers, decoder_max_steps):
        super(Seq2SeqNREModule, self).__init__()
        self.decoder_max_steps = decoder_max_steps
        self.tokenizer = WhiteSpaceTokenizer(vocab_file)
        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(), embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, hidden_size, self.tokenizer.get_vocab_size())
        self.register_buffer('h0', torch.zeros(1, hidden_size, dtype=torch.float))
        self.register_buffer('c0', torch.zeros(1, hidden_size, dtype=torch.float))
        self.register_buffer('start_ids', torch.LongTensor([[self.tokenizer.get_start_id()]]))

    def forward(self, input_ids, masks, input_vocab_maks, target_ids, inference=False):
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
            output, hidden, att_score = self.decoder(encoder_outputs, masks, decoder_input, hidden)
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
        tokens, targets, input_ids, masks, input_vocab_masks, target_ids = batch
        outputs, att_scores = self(input_ids, masks, input_vocab_masks, target_ids)
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








