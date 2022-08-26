import torch
import transformers
from module import IEModule
from collections import defaultdict
from transformers import AutoModel, AutoConfig


class NERModule(IEModule):
    def get_f1_stats(self, preds, targets, masks=None):
        tp, fp, fn = 0, 0, 0
        pred_chunks = self.get_batch_ner_chunks(preds, masks)
        target_chunks = self.get_ner_chunks(targets, masks)
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
        tags = [self.idx2tag[tag_id] for tag_id in tag_ids]
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


class BertBasedModule(NERModule):
    def __init__(
            self,
            pretrained_model_name: str = 'bert-base-chinese',
            warmup_steps: int = 1000,
            num_total_steps: int = 270000
    ):
        super(BertBasedModule, self).__init__()
        self.warmup_steps = warmup_steps
        self.num_total_steps = num_total_steps
        self.config = self.get_config(pretrained_model_name)
        self.pretrained_model = self.get_pretrained_model(pretrained_model_name)

    @staticmethod
    def get_config(pretrained_model_name):
        return AutoConfig.from_pretrained(pretrained_model_name)

    @staticmethod
    def get_pretrained_model(pretrained_model_name):
        return AutoModel.from_pretrained(pretrained_model_name)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optim = torch.optim.AdamW(optimizer_grouped_parameters,
                                  2e-5,
                                  (0.9, 0.98),
                                  1e-8,
                                  0.01)

        lr = transformers.get_polynomial_decay_schedule_with_warmup(optim, self.warmup_steps,
                                                                    self.num_total_steps, lr_end=2e-5 / 5)
        return [optim], [lr]