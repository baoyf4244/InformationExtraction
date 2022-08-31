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

    def get_data(self, line):
        tokens = self.vocab.tokenize(line['text'])
        token_ids = self.vocab.convert_tokens_to_ids(tokens)
        masks = [1] * len(token_ids)
        input_vocab_ids, oov2idx = self.get_oov(tokens, token_ids)
        targets = self.vocab.tokenize(self.vocab.get_vertical_token().join(line['keywords']))
        target_ids = self.vocab.convert_tokens_to_ids(targets) + [self.vocab.get_end_id()]

        data = {
            'id': line['id'],
            'masks': masks,
            'targets': targets,
            'input_ids': token_ids,
            'target_ids': target_ids,
            'target_masks': [1] * len(target_ids),
            'input_vocab_ids': input_vocab_ids,
            'oov2idx': oov2idx
        }
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

    @staticmethod
    def pad(seqs):
        max_len = max(len(seq) for seq in seqs)
        padded_seqs = []
        for seq in seqs:
            padded_seqs.append(seq + [0] * (max_len - len(seq)))
        return padded_seqs

    def collocate_fn(self, batch):
        ids = self.get_data_by_name(batch, 'id')
        targets = self.get_data_by_name(batch, 'targets')

        oov2idx = self.get_data_by_name(batch, 'oov2idx')
        oov_ids = [list(oov.values()) for oov in oov2idx]
        oov_id_masks = [[1] * len(oov_id) for oov_id in oov_ids]
        oov_ids = self.pad_batch(oov_ids, self.vocab.get_pad_id())
        oov_id_masks = self.pad_batch(oov_id_masks, 0)

        input_masks = self.pad_batch(self.get_data_by_name(batch, 'masks'), 0)
        input_ids = self.pad_batch(self.get_data_by_name(batch, 'input_ids'), self.vocab.get_pad_id())
        target_masks = self.pad_batch(self.get_data_by_name(batch, 'target_masks'), 0)
        target_ids = self.pad_batch(self.get_data_by_name(batch, 'target_ids'), self.vocab.get_pad_id())
        input_vocab_ids = self.pad_batch(self.get_data_by_name(batch, 'input_vocab_ids'), self.vocab.get_pad_id())

        return ids, targets, input_ids, input_masks, target_ids, target_masks, input_vocab_ids, oov_ids, oov_id_masks, oov2idx


if __name__ == '__main__':
    vocab_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/kpe/vocab.txt'
    data_file = 'C:/Users/ML-YX01/code/InformationExtraction/data/kpe/dev.txt'

