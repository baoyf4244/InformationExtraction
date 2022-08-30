import torch
from nre.vocab import WDVocab, PTRLabelVocab
from module import Vocab, LabelVocab, IEDataSet, IEDataModule


class WDNREDataSet(IEDataSet):
    def __init__(self, vocab: WDVocab, *args, **kwargs):
        super(WDNREDataSet, self).__init__(*args, **kwargs)
        self.data_file = data_file
        self.vocab = vocab
        self.label_vocab = self.vocab.label_vocab

    def get_vocab_masks(self, token_ids):
        vocab_masks = [1] * self.vocab.get_vocab_size()
        for token_id in token_ids:
            vocab_masks[token_id] = 0

        vocab_masks[self.vocab.get_end_id()] = 0
        vocab_masks[self.vocab.get_vertical_token_id()] = 0
        vocab_masks[self.vocab.get_semicolon_token_id()] = 0
        for relation in self.label_vocab.get_vocab():
            vocab_masks[self.vocab.convert_token_to_id(relation)] = 0

        return vocab_masks

    def get_targets(self, labels, tokens):
        targets = []
        for label in labels:
            head_start, head_end, tail_start, tail_end, relation = label.split()
            head = tokens[int(head_start): int(head_end) + 1]
            tail = tokens[int(tail_start): int(tail_end) + 1]
            target = head + [self.vocab.get_semicolon_token()] + tail + [self.vocab.get_special_tokens()] + [relation]
            targets.extend(target)
            targets.append(self.vocab.get_vertical_token())

        targets.append(self.vocab.get_end_token())
        return targets

    def get_data(self, line):
        tokens = self.vocab.tokenize(line['text'])
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        targets = self.get_targets(line['label'], tokens)
        target_ids = self.vocab.convert_tokens_to_ids(targets)

        data = {
            'tokens': tokens,
            'token_ids': token_ids,
            'masks': [1] * len(token_ids),
            'vocab_masks': self.get_vocab_masks(token_ids),
            'targets': targets,
            'target_ids': target_ids,
            'target_masks': [1] * len(token_ids)
        }
        return data


class WDNREDataModule(IEDataModule):
    def __init__(self, vocab_file, label_file, *args, **kwargs):
        super(WDNREDataModule, self).__init__(*args, **kwargs)
        self.label_vocab = LabelVocab(label_file)
        self.vocab = WDVocab(self.label_vocab, vocab_file=vocab_file)

    def get_dataset(self, data_file, is_predict=False):
        dataset = WDNREDataSet(self.vocab, data_file=data_file, max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    def collocate_fn(self, batch):
        tokens = self.get_data_by_name(batch, 'tokens')
        targets = self.get_data_by_name(batch, 'targets')
        vocab_masks = torch.LongTensor(self.get_data_by_name(batch, 'vocab_masks'))

        input_ids = self.pad_batch(self.get_data_by_name(batch, 'token_ids'), self.vocab.get_pad_id())
        input_masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        target_ids = self.pad_batch(self.get_data_by_name(batch, 'target_ids'), self.vocab.get_pad_id())
        target_masks = self.pad_batch(self.get_data_by_name(batch, 'target_masks'), 0)

        return tokens, targets, input_ids, input_masks, vocab_masks, target_ids, target_masks


class PTRNREDataSet(IEDataSet):
    def __init__(self, vocab: Vocab, label_vocab: PTRLabelVocab, *args, **kwargs):
        super(PTRNREDataSet, self).__init__(*args, **kwargs)
        self.data_file = data_file
        self.vocab = vocab
        self.label_vocab = label_vocab

    def get_targets(self, labels):
        head_start_offsets, head_end_offsets, tail_start_offsets, tail_end_offsets, relation_ids = [], [], [], [], []
        for label in labels:
            head_start, head_end, tail_start, tail_end, relation = label.split()
            head_start_offsets.append(int(head_start))
            head_end_offsets.append(int(head_end))
            tail_start_offsets.append(int(tail_start))
            tail_end_offsets.append(int(tail_end))
            relation_ids.append(self.label_vocab.convert_token_to_id(relation))

        return head_start_offsets, head_end_offsets, tail_start_offsets, tail_end_offsets, relation_ids

    def get_data(self, line):
        tokens = self.vocab.tokenize(line['text'])
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        head_start_offsets, head_end_offsets, tail_start_offsets, tail_end_offsets, relation_ids = self.get_targets(line['label'])

        data = {
            'tokens': tokens,
            'token_ids': token_ids,
            'masks': [1] * len(token_ids),
            'head_start_offsets': head_start_offsets + [-1],
            'head_end_offsets': head_end_offsets + [-1],
            'tail_start_offsets': tail_start_offsets + [-1],
            'tail_end_offsets': tail_end_offsets + [-1],
            'relation_ids': relation_ids + [self.label_vocab.get_end_id()],
            'target_masks': [1] * len(relation_ids)
        }
        return data


class PTRNREDataModule(IEDataModule):
    def __init__(
            self,
            vocab_file: str = 'data/nre/vocab.txt',
            label_file: str = 'data/nre/relations.txt',
            *args, **kwargs
    ):
        """
        Args:
            label_file:
            vocab_file:
        """
        super(PTRNREDataModule, self).__init__(*args, **kwargs)
        self.vocab = Vocab(vocab_file)
        self.label_vocab = PTRLabelVocab(label_file)
        self.save_hyperparameters()

    def get_dataset(self, data_file, is_predict=False):
        dataset = PTRNREDataSet(self.vocab, self.label_vocab, data_file=data_file, max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    def collocate_fn(self, batch):
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'token_ids'), self.vocab.get_pad_id())
        input_masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)

        head_start_offsets = self.pad_batch(self.get_data_by_name(batch, 'head_start_offsets'), 0)
        head_end_offsets = self.pad_batch(self.get_data_by_name(batch, 'head_end_offsets'), 0)
        tail_start_offsets = self.pad_batch(self.get_data_by_name(batch, 'tail_start_offsets'), 0)
        tail_end_offsets = self.pad_batch(self.get_data_by_name(batch, 'tail_end_offsets'), 0)
        relation_ids = self.pad_batch(self.get_data_by_name(batch, 'relation_ids'), 0)
        target_masks = self.pad_batch(self.get_data_by_name(batch, 'target_masks'), 0)

        return (input_ids, input_masks, head_start_offsets, head_end_offsets, tail_start_offsets,
                tail_end_offsets, relation_ids, target_masks)


if __name__ == '__main__':
    vocab_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/vocab.txt'
    data_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/train.txt'
    relations = 'C:/Users/ML-YX01/code/InformationExtraction/data/nre/relations.txt'

