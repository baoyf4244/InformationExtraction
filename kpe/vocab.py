from module import Seq2SeqVocab, SpecialTokens


class KPESeq2SeqVocab(Seq2SeqVocab):
    @staticmethod
    def get_special_tokens():
        return Seq2SeqVocab.get_special_tokens() + [SpecialTokens.VERTICAL.value]

    def get_vertical_id(self):
        return self.word2idx[SpecialTokens.VERTICAL.value]

    @staticmethod
    def get_vertical_token():
        return SpecialTokens.VERTICAL.value
