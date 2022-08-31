import json

import torch
from ner.vocab import EntityLabelVocab
from module import Vocab, IEModule
from collections import defaultdict


class NERModule(IEModule):
    def __init__(self, vocab_file, label_file, *args, **kwargs):
        super(NERModule, self).__init__(*args, **kwargs)
        self.vocab = Vocab(vocab_file)
        self.label_vocab = EntityLabelVocab(label_file)
        self.vocab_size = self.vocab.get_vocab_size()
        self.num_labels = self.label_vocab.get_vocab_size()

    def get_f1_stats(self, preds, targets, masks=None):
        tp, fp, fn = 0, 0, 0
        pred_chunks = self.get_batch_ner_chunks(preds, masks)
        target_chunks = self.get_batch_ner_chunks(targets, masks)
        for pred_chunk, target_chunk in zip(pred_chunks, target_chunks):
            for label, indices in target_chunk.items():
                pred_indices = pred_chunk[label]
                tp += len(pred_indices.intersection(indices))

                for idx in pred_indices:
                    if idx not in indices:
                        fp += 1

                for idx in indices:
                    if idx not in pred_indices:
                        fn += 1
        return tp, fp, fn

    def get_batch_ner_chunks(self, tag_ids, masks):
        chunks = []
        for tag_id, mask in zip(tag_ids, masks):
            chunks.append(self.get_ner_chunks(tag_id, mask))
        return chunks

    def get_ner_chunks(self, tag_ids, masks):
        """
        根据标签ID返回标签名称对应的索引位置，标签采用BIO模式
        :param masks:
        :param tag_ids: 标签ID
        :return: dict，标签名称及相应的索引列表，索引列表为二元组，元组第一个元素为实体起始位置，第二个索引为实体结束位置+1（方便取值）
        """
        if isinstance(tag_ids, torch.Tensor):
            tag_ids = tag_ids.cpu().numpy().tolist()

        if isinstance(masks, torch.Tensor):
            seq_len = masks.sum()
        else:
            seq_len = sum(masks)

        for tag_id in tag_ids:
            if isinstance(tag_id, list):
                print(tag_ids)
        tags = [self.label_vocab.convert_id_to_token(tag_id) for tag_id in tag_ids]
        chunks = defaultdict(set)
        i = 0
        while i < seq_len:
            if tags[i].startswith('B-'):
                label = tags[i].split('-')[1]
                start = i
                i += 1
                while i < seq_len and tags[i].startswith('I-') and tags[i].split('-')[1] == label:
                    i += 1
                chunks[label].add((start, i))
            else:
                i += 1

        return chunks

    def get_training_outputs(self, batch):
        _, _, masks, input_ids, targets = batch
        preds, loss = self(input_ids, targets, masks)
        return preds, targets, masks, loss

    def get_predict_outputs(self, batch):
        ids, tokens, masks, input_ids = batch
        preds, loss = self(input_ids, masks=masks)
        return ids, tokens, preds, masks

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        ids, tokens, preds, masks = self.get_predict_outputs(batch)
        chunks = self.get_batch_ner_chunks(preds, masks)
        for idx, token, chunk in zip(ids, tokens, chunks):
            entities = []
            for entity_type,  offsets in chunk:
                for start, end in offsets:
                    entities.append({'type': entity_type, 'offset': [start, end - 1], 'span': token[start: end]})

            self.results.write(json.dumps({'id': idx, 'entities': entities}, ensure_ascii=False) + '\n')

