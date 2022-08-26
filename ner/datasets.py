import os
import json
import torch
from typing import Optional
from collections import defaultdict
from module import IEDataSet, IEDataModule, Vocab, SpecialTokens
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, AutoTokenizer


class LabelVocab(Vocab):
    @staticmethod
    def get_non_entity_token():
        return SpecialTokens.NON_ENTITY.value

    def get_non_entity_token_id(self):
        return self.word2idx[SpecialTokens.NON_ENTITY.value]

    def convert_token_to_id(self, token):
        return self.word2idx[token]


class FlatNERDataSet(IEDataSet):
    def __init__(self, vocab, label_vocab: LabelVocab, *args, **kwargs):
        super(FlatNERDataSet, self).__init__(*args, **kwargs)
        self.vocab = vocab
        self.label_vocab = label_vocab

    def get_label_ids(self, labels, seq_len):
        label_ids = [self.label_vocab.get_non_entity_token_id()] * seq_len
        for label, indices in labels.items():
            for start, end in indices:
                if start >= seq_len:
                    continue
                label_ids[start] = self.label_vocab.convert_token_to_id('B-' + label)

                for i in range(start + 1, end + 1):
                    label_ids[i] = self.label_vocab.convert_token_to_id('I-' + label)

        return label_ids

    def get_data(self, line):
        data = {
            'id': line['id'],
            'tokens': line['text'].split()
        }

        data['masks'] = [1] * len(data['tokens'])
        data['input_ids'] = self.vocab.convert_tokens_to_ids(data['tokens'])
        data['target_ids'] = self.get_label_ids(line['labels'], len(data['tokens']))
        return data


class FlatNERDataModule(IEDataModule):
    def __init__(self, vocab_file, label_file, *args, **kwargs):
        super(FlatNERDataModule, self).__init__(*args, **kwargs)
        self.vocab = Vocab(vocab_file)
        self.label_vocab = LabelVocab(label_file)

    def get_dataset(self, data_file, is_predict=False):
        dataset = FlatNERDataSet(data_file, vocab=self.vocab, label_vocab=self.label_vocab,
                                 max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    def collocate_fn(self, batch):
        ids = self.get_data_by_name(batch, 'id')
        tokens = [self.get_data_by_name(batch, 'tokens')]
        masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'input_ids'), self.vocab.get_pad_id())
        target_ids = self.pad_batch(self.get_data_by_name(batch, 'target_ids'), self.label_vocab.get_pad_id())

        return ids, tokens, masks, input_ids, target_ids


class MRCNERDataSet(IEDataSet):
    def __init__(self, tokenizer: BertTokenizer, label_vocab, question_vocab, *args, **kwargs):
        super(MRCNERDataSet, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.question_vocab = question_vocab
        self.label_question_mapping = self.get_label_question_mapping()

    def get_label_question_mapping(self):
        label_question_mapping = dict(zip(self.label_vocab.get_vocab(), self.question_vocab.get_vocab()))
        return {key: self.tokenizer.tokenize(value) for key, value in label_question_mapping.items()}

    def get_data(self, line):
        dataset = []
        tokens = self.tokenizer.tokenize(line['text'])
        for label, question in self.label_question_mapping.items():
            token_type_ids = [0] * (len(question) + 2) + [1] * len(tokens)
            tokens = question + ['[SEP]'] + tokens
            tokens = ['[CLS]'] + tokens[self.max_len - 3] + ['[SEP]']
            start_labels = [0] * len(tokens)
            end_labels = [0] * len(tokens)
            span_labels = [[0 for _ in range(len(tokens))] for _ in range(len(tokens))]

            indices = line['labels'].get(label, [])
            for start, end in indices:
                start = start + len(question) + 2
                end = end + len(question) + 2
                if start < len(tokens) - 1:
                    start_labels[start] = 1

                    if end < len(tokens) - 1:
                        end_labels[end] = 1
                        span_labels[start][end] = 1
                    else:
                        end_labels[len(tokens) - 2] = 1
                        span_labels[start][len(tokens) - 2] = 1
            data = {
                'id': line['id'],
                'masks': [1] * len(tokens),
                'tokens': tokens,
                'input_ids': self.tokenizer.convert_tokens_to_ids(tokens),
                'token_type_ids': token_type_ids[: len(tokens)],
                'start_labels': start_labels,
                'end_labels': end_labels,
                'span_labels': span_labels
            }
            dataset.append(data)
        return dataset


class MRCNERDataModule(IEDataModule):
    def __init__(self, pretrained_model_name, label_file, question_file, *args, **kwargs):
        super(MRCNERDataModule, self).__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.label_vocab = LabelVocab(label_file)
        self.question_vocab = LabelVocab(question_file)

    def get_dataset(self, data_file, is_predict=False):
        dataset = MRCNERDataSet(self.tokenizer, self.label_vocab, self.question_vocab, data_file=data_file,
                                max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    @staticmethod
    def pad_batch_2d(batch, pad_token):
        max_len = max(len(b) for b in batch)
        max_len_2d = max(len(b[0]) for b in batch)
        padded_batch = torch.LongTensor([[pad_token for _ in  range(max_len_2d)] for _ in range(max_len)])
        for i in range(max_len):
            padded_batch[i, :len(batch[i])] = batch[i]
        return padded_batch

    def collocate_fn(self, batch):
        ids = self.get_data_by_name(batch, 'id')
        tokens = [self.get_data_by_name(batch, 'tokens')]
        masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'input_ids'), 0)
        token_type_ids = self.pad_batch(self.get_data_by_name(batch, 'token_type_ids'), 0)
        start_labels = self.pad_batch(self.get_data_by_name(batch, 'start_labels'), 0)
        end_labels = self.pad_batch(self.get_data_by_name(batch, 'end_labels'), 0)
        span_labels = self.pad_batch_2d(self.get_data_by_name(batch, 'span_labels'), 0)

        return ids, tokens, masks, input_ids, token_type_ids, start_labels, end_labels, span_labels


if __name__ == '__main__':
    filename = 'C:/Users/ML-YX01/code/InformationExtraction/data/test.txt'
    tag_filename = 'C:/Users/ML-YX01/code/InformationExtraction/data/tags.txt'
    vocab = Vocab('C:/Users/ML-YX01/code/InformationExtraction/data/vocab.txt')
    label_vocab = LabelVocab('C:/Users/ML-YX01/code/InformationExtraction/data/tags.txt')

    dataset = FlatNERDataSet(vocab, label_vocab, max_len=200, data_file=filename)
    dataset.make_dataset()
    print(dataset[0])

    # dataset = FlatNERDataSet().collocate_fn(dataset[:10])
