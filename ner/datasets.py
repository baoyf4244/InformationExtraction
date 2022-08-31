import torch
from transformers import BertTokenizer, AutoTokenizer

from ner.vocab import EntityLabelVocab
from module import IEDataSet, IEDataModule, Vocab


class FlatNERDataSet(IEDataSet):
    def __init__(self, vocab, label_vocab: EntityLabelVocab, *args, **kwargs):
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
        if not self.is_predict:
            data['target_ids'] = self.get_label_ids(line['labels'], len(data['tokens']))
        return data


class FlatNERDataModule(IEDataModule):
    def __init__(self, vocab_file, label_file, *args, **kwargs):
        super(FlatNERDataModule, self).__init__(*args, **kwargs)
        self.vocab = Vocab(vocab_file)
        self.label_vocab = EntityLabelVocab(label_file)

    def get_dataset(self, data_file, is_predict=False):
        dataset = FlatNERDataSet(data_file=data_file, vocab=self.vocab, label_vocab=self.label_vocab,
                                 max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    def collocate_fn(self, batch, is_predict=False):
        ids = self.get_data_by_name(batch, 'id')
        tokens = [self.get_data_by_name(batch, 'tokens')]
        masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'input_ids'), self.vocab.get_pad_id())
        if is_predict:
            return ids, tokens, masks, input_ids

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

    @staticmethod
    def get_labels(labels, entity_type, sent_len):
        start_labels = [0] * sent_len
        end_labels = [0] * sent_len
        span_labels = [[0 for _ in range(sent_len)] for _ in range(sent_len)]

        indices = labels.get(entity_type, [])
        for start, end in indices:
            start = start + len(question) + 2
            end = end + len(question) + 2
            if start < sent_len - 1:
                start_labels[start] = 1

                if end < sent_len - 1:
                    end_labels[end] = 1
                    span_labels[start][end] = 1
                else:
                    end_labels[sent_len - 2] = 1
                    span_labels[start][sent_len - 2] = 1
        return start_labels, end_labels, span_labels

    def get_data(self, line):
        dataset = []
        context_tokens = self.tokenizer.tokenize(line['text'])
        for label, question in self.label_question_mapping.items():
            token_type_ids = [0] * (len(question) + 2) + [1] * (len(context_tokens) + 1)
            tokens = question + ['[SEP]'] + context_tokens
            tokens = ['[CLS]'] + tokens[: self.max_len - 2] + ['[SEP]']

            data = {
                'id': line['id'],
                'entity_type': label,
                'masks': [1] * len(tokens),
                'tokens': tokens,
                'input_ids': self.tokenizer.convert_tokens_to_ids(tokens),
                'token_type_ids': token_type_ids[: len(tokens)]
            }

            if not self.is_predict:
                start_labels, end_labels, span_labels = self.get_labels(line['labels'], label, len(tokens))
                data['start_labels'] = start_labels
                data['end_labels'] = end_labels
                data['span_labels'] = span_labels
            dataset.append(data)
        return dataset


class MRCNERDataModule(IEDataModule):
    def __init__(self, pretrained_model_name, label_file, question_file, *args, **kwargs):
        super(MRCNERDataModule, self).__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.label_vocab = EntityLabelVocab(label_file)
        self.question_vocab = EntityLabelVocab(question_file)

    def get_dataset(self, data_file, is_predict=False):
        dataset = MRCNERDataSet(self.tokenizer, self.label_vocab, self.question_vocab, data_file=data_file,
                                max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    @staticmethod
    def pad_batch_2d(batch, pad_token):
        max_len = max(len(b) for b in batch)
        max_len_2d = max(len(b[0]) for b in batch)
        padded_batch = torch.fill(torch.zeros(len(batch), max_len, max_len_2d, dtype=torch.int64), pad_token)
        for i in range(len(batch)):
            padded_batch[i, :len(batch[i]), :len(batch[i][0])] = torch.LongTensor(batch[i])
        return padded_batch

    def collocate_fn(self, batch, is_predict=False):
        ids = self.get_data_by_name(batch, 'id')
        tokens = self.get_data_by_name(batch, 'tokens')
        entity_types = self.get_data_by_name(batch, 'entity_type')
        masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'input_ids'), 0)
        token_type_ids = self.pad_batch(self.get_data_by_name(batch, 'token_type_ids'), 0)

        if is_predict:
            return ids, tokens, entity_types, masks, input_ids, token_type_ids
        start_labels = self.pad_batch(self.get_data_by_name(batch, 'start_labels'), 0)
        end_labels = self.pad_batch(self.get_data_by_name(batch, 'end_labels'), 0)
        span_labels = self.pad_batch_2d(self.get_data_by_name(batch, 'span_labels'), 0)

        return ids, tokens, entity_types, masks, input_ids, token_type_ids, start_labels, end_labels, span_labels


if __name__ == '__main__':
    filename = '/data/ner/test.txt'
    tag_filename = 'C:/Users/ML-YX01/code/InformationExtraction/data/ner/labels.txt'
    question = 'C:/Users/ML-YX01/code/InformationExtraction/data/ner/questions.txt'
    vocab_file = '/data/ner/vocab.txt'
    # vocab = Vocab('C:/Users/ML-YX01/code/InformationExtraction/data/vocab.txt')
    # label_vocab = LabelVocab('C:/Users/ML-YX01/code/InformationExtraction/data/tags.txt')

    # dataset = FlatNERDataSet(vocab, label_vocab, max_len=200, data_file=filename)
    # dataset.make_dataset()

    data_module = MRCNERDataModule('bert-base-chinese', tag_filename, question, data_dir='data')
    dataset = data_module.get_dataset(filename, False)
    print(dataset[:10])
    pad = data_module.collocate_fn(dataset[:10])
    print(pad)

    # dataset = FlatNERDataSet().collocate_fn(dataset[:10])
