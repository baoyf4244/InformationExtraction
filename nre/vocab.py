from module import Vocab, LabelVocab, SpecialTokens


class WDVocab(Vocab):
    def __init__(self, label_vocab: LabelVocab, *args, **kwargs):
        self.label_vocab = label_vocab
        super(WDVocab, self).__init__(*args, **kwargs)

    def get_start_id(self):
        return self.word2idx[SpecialTokens.SOS.value]

    @staticmethod
    def get_start_token():
        return SpecialTokens.SOS.value

    def get_end_id(self):
        return self.word2idx[SpecialTokens.EOS.value]

    @staticmethod
    def get_end_token():
        return SpecialTokens.EOS.value

    @staticmethod
    def get_vertical_token():
        return SpecialTokens.VERTICAL.value

    def get_vertical_token_id(self):
        return self.word2idx[SpecialTokens.VERTICAL.value]

    @staticmethod
    def get_semicolon_token():
        return SpecialTokens.SEMICOLON.value

    def get_semicolon_token_id(self):
        return self.word2idx[SpecialTokens.SEMICOLON.value]

    def get_special_tokens(self):
        return super(WDVocab, self).get_special_tokens() + [SpecialTokens.EOS.value, SpecialTokens.SOS.value, SpecialTokens.VERTICAL.value, SpecialTokens.SEMICOLON.value] + self.label_vocab.get_vocab()


class PTRLabelVocab(LabelVocab):
    @staticmethod
    def get_special_tokens():
        return [SpecialTokens.EOS.value]

    def get_end_id(self):
        return self.word2idx[SpecialTokens.EOS.value]

    @staticmethod
    def get_end_token():
        return SpecialTokens.EOS.value