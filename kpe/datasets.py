from kpe.vocab import KPESeq2SeqVocab
from module import IEDataSet, IEDataModule


class KPEDataSet(IEDataSet):
    def __init__(self, vocab: KPESeq2SeqVocab, *args, **kwargs):
        super(KPEDataSet, self).__init__(*args, **kwargs)
        self.vocab = vocab

    def get_oov(self, tokens, token_ids):
        input_vocab_ids = []
        oov2idx = {}
        for token, token_id in zip(tokens, token_ids):
            if token_id == self.vocab.get_unk_id():
                if token not in oov2idx:
                    oov2idx[token] = self.vocab.get_vocab_size() + len(oov2idx)
                input_vocab_ids.append(oov2idx[token])
            else:
                input_vocab_ids.append(token_id)
        return input_vocab_ids, oov2idx

    def get_target_ids(self, targets):
        target_ids = []
        for target in targets:
            tokens = self.vocab.tokenize(target)
            target_ids.extend(self.vocab.convert_tokens_to_ids(tokens))
            target_ids.append(self.vocab.get_vertical_id())
        target_ids.append(self.vocab.get_end_id())
        return target_ids

    def get_data(self, line):
        tokens = self.vocab.tokenize(line['text'])
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        masks = [1] * len(token_ids)
        input_vocab_ids, oov2idx = self.get_oov(tokens, token_ids)

        data = {
            'id': line['id'],
            'masks': masks,
            'input_ids': token_ids,
            'input_vocab_ids': input_vocab_ids,
            'oov2idx': oov2idx
        }

        if not self.is_predict:
            targets = line['keywords']
            target_ids = self.get_target_ids(targets)

            data['targets'] = targets
            data['target_ids'] = target_ids
            data['target_masks'] = [1] * len(target_ids)

        return data


class KPEDataModule(IEDataModule):
    def __init__(
            self,
            vocab_file: str = 'data/kpe/vocab.txt',
            min_freq: int = 10,
            do_lower: bool = False,
            *args, **kwargs
    ):
        """
        Args:
            vocab_file:
        """
        super(KPEDataModule, self).__init__(*args, **kwargs)
        self.vocab = KPESeq2SeqVocab(vocab_file=vocab_file, data_file=self.train_file, min_freq=min_freq, do_lower=do_lower)
        self.save_hyperparameters()

    def get_dataset(self, data_file, is_predict=False):
        dataset = KPEDataSet(self.vocab, data_file=data_file, max_len=self.max_len, is_predict=is_predict)
        dataset.make_dataset()
        return dataset

    def collocate_fn(self, batch, is_predict=False):
        ids = self.get_data_by_name(batch, 'id')

        oov2idx = self.get_data_by_name(batch, 'oov2idx')
        oov_ids = [list(oov.values()) for oov in oov2idx]
        oov_id_masks = [[1] * len(oov_id) for oov_id in oov_ids]
        oov_ids = self.pad_batch(oov_ids, self.vocab.get_pad_id())
        oov_id_masks = self.pad_batch(oov_id_masks, 0)

        input_masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'input_ids'), self.vocab.get_pad_id())
        input_vocab_ids = self.pad_batch(self.get_data_by_name(batch, 'input_vocab_ids'), self.vocab.get_pad_id())

        if is_predict:
            return ids, input_ids, input_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx

        targets = self.get_data_by_name(batch, 'targets')
        target_masks = self.pad_batch(self.get_data_by_name(batch, 'target_masks'), 0)
        target_ids = self.pad_batch(self.get_data_by_name(batch, 'target_ids'), self.vocab.get_pad_id())

        return ids, input_ids, input_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx, targets, target_ids, target_masks


if __name__ == '__main__':
    vocab_file = '../data/kpe/vocab.txt'
    data_file = '../data/kpe/train.txt'
    vocab = KPESeq2SeqVocab(vocab_file=vocab_file, data_file=data_file, min_freq=10, do_lower=False)

