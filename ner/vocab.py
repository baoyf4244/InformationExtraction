from module import LabelVocab, SpecialTokens


class EntityLabelVocab(LabelVocab):
    @staticmethod
    def get_special_tokens():
        return [SpecialTokens.PAD.value, SpecialTokens.NON_ENTITY.value]

    @staticmethod
    def get_non_entity_token():
        return SpecialTokens.NON_ENTITY.value

    def get_non_entity_token_id(self):
        return self.word2idx[SpecialTokens.NON_ENTITY.value]

    def get_pad_id(self):
        return self.word2idx[SpecialTokens.PAD.value]

    @staticmethod
    def get_pad_token():
        return SpecialTokens.PAD.value
