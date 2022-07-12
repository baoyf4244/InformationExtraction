import os
import json
import torch
from typing import Optional
from collections import defaultdict

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from transformers import BertTokenizer
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        data_file: path to mrc-ner style json
        tokenizer: BertTokenizer
    """
    def __init__(self, data_file, tag_file, tokenizer: BertTokenizer, max_len=200, predict=False):
        self.data_file = data_file
        self.tag_file = tag_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.predict = predict
        self.tag2query = self.get_tag2query()
        self.dataset = self.make_dataset()

    def get_tag2query(self):
        with open(self.tag_file, encoding='utf-8') as f:
            tags = json.load(f)

        tag2query = {}
        for tag in tags.values():
            query_tokens = ['[CLS]'] + self.tokenizer.tokenize(tag['query']) + ['[SEP]']
            query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
            tag2query[tag['tag'].lower()] = query_ids
        return tag2query

    def make_dataset(self):
        dataset = []
        with open(self.data_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                segments = line.strip().split()
                context_tokens = []
                offsets = defaultdict(list)
                start_tags = []
                end_tags = []
                for segment in segments:
                    if self.predict:
                        word = segment
                        tag = 'o'
                    else:
                        word, tag = segment.split('/')
                    sub_tokens = self.tokenizer.tokenize(word)
                    context_tokens.extend(sub_tokens)
                    if tag in self.tag2query:
                        offsets[tag].append((len(start_tags), len(start_tags) + len(sub_tokens) - 1))
                        start_tags.extend([tag] + ['o'] * (len(sub_tokens) - 1))
                        end_tags.extend(['o'] * (len(sub_tokens) - 1) + [tag])
                    else:
                        start_tags.extend(['o'] * len(sub_tokens))
                        end_tags.extend(['o'] * len(sub_tokens))

                for tag, query in self.tag2query.items():
                    context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
                    context_token_ids = context_token_ids[: self.max_len - len(query) - 1]
                    context_token_ids.append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
                    token_ids = query + context_token_ids
                    token_type_ids = [0] * len(query) + [1] * len(context_token_ids)
                    attention_masks = [1] * len(token_ids)

                    start_labels = [0] * len(query) + [1 if context_tag == tag else 0 for context_tag in start_tags]
                    end_labels = [0] * len(query) + [1 if context_tag == tag else 0 for context_tag in end_tags]
                    start_labels = start_labels[: self.max_len - 1] + [0]
                    end_labels = end_labels[: self.max_len - 1] + [0]

                    assert len(start_labels) == len(end_labels) == len(token_ids) == len(attention_masks) ==len(token_type_ids)

                    span_labels = torch.zeros([len(start_labels), len(start_labels)], dtype=torch.int64)
                    for start, end in offsets[tag]:
                        start += len(query)
                        end += len(query)
                        if start >= len(start_labels) - 1 or end >= len(start_labels) - 1:
                            continue

                        assert start_labels[start] == 1
                        assert end_labels[end] == 1

                        span_labels[start, end] = 1

                    data = {
                        'input_ids': token_ids,
                        'attention_mask': attention_masks,
                        'token_type_ids': token_type_ids,
                        'start_labels': start_labels,
                        'end_labels': end_labels,
                        'span_labels': span_labels
                    }
                    dataset.append(data)

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            a dict with:
            input_ids: token ids of query + context, [seq_len]
            attention_mask:
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labels of NER in tokens, [seq_len]
            span_labels: span labels, [seq_len, seq_len]
        """
        return self.dataset[item]

    @staticmethod
    def collocate_fn(batch):
        max_seq_len = max([len(data['input_ids']) for data in batch])
        input_ids = [data['input_ids'] + [0] * (max_seq_len - len(data['input_ids'])) for data in batch]
        attention_mask = [data['attention_mask'] + [0] * (max_seq_len - len(data['attention_mask'])) for data in batch]
        token_type_ids = [data['token_type_ids'] + [0] * (max_seq_len - len(data['token_type_ids'])) for data in batch]
        start_labels = [data['start_labels'] + [0] * (max_seq_len - len(data['start_labels'])) for data in batch]
        end_labels = [data['end_labels'] + [0] * (max_seq_len - len(data['end_labels'])) for data in batch]
        span_labels = torch.zeros(len(batch), max_seq_len, max_seq_len, dtype=torch.int64)
        for i, data in enumerate(batch):
            length, width = data['span_labels'].size()
            span_labels[i, :length, :width] = data['span_labels']

        # return {
        #     'input_ids': torch.LongTensor(input_ids),
        #     'attention_mask': torch.LongTensor(attention_mask),
        #     'token_type_ids': torch.LongTensor(token_type_ids),
        #     'start_labels': torch.LongTensor(start_labels),
        #     'end_labels': torch.LongTensor(end_labels),
        #     'span_labels': span_labels
        # }

        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(token_type_ids), \
               torch.LongTensor(start_labels), torch.LongTensor(end_labels), span_labels


class MRCDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str = 'data',
                 max_len: int = 200,
                 batch_size: int = 16,
                 pretrained_model_name: str = 'bert-base-chinese'):
        """

        :param data_dir: 数据存放目录
        :param max_len: 数据保留的最大长度
        :param batch_size:
        :param pretrained_model_name:
        """
        super(MRCDataModule, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        train_file = os.path.join(self.data_dir, 'train.txt')
        val_file = os.path.join(self.data_dir, 'dev.txt')
        test_file = os.path.join(self.data_dir, 'test.txt')
        predict_file = os.path.join(self.data_dir, 'predict_file')
        tag_file = os.path.join(self.data_dir, 'questions.json')

        if stage == 'fit' or stage is None:
            self.train_dataset = MRCNERDataset(train_file, tag_file, self.tokenizer)
            if os.path.isfile(val_file):
                self.val_dataset = MRCNERDataset(val_file, tag_file, self.tokenizer)
            else:
                data_size = len(self.train_dataset)
                train_size = int(data_size * 0.8)
                val_size = data_size - train_size
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            if os.path.isfile(test_file):
                self.test_dataset = MRCNERDataset(test_file, tag_file, self.tokenizer)

        if stage == 'predict' or stage is None:
            self.predict_dataset = MRCNERDataset(predict_file, tag_file, self.tokenizer, True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, self.batch_size, collate_fn=MRCNERDataset.collocate_fn)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, self.batch_size, collate_fn=MRCNERDataset.collocate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size, collate_fn=MRCNERDataset.collocate_fn)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset, self.batch_size, collate_fn=MRCNERDataset.collocate_fn)


if __name__ == '__main__':
    filename = 'D:/code/NLP/InformationExtraction/data/test.txt'
    tag_filename = 'D:/code/NLP/InformationExtraction/data/questions.json'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = MRCNERDataset(filename, tag_filename, tokenizer)
    print(dataset[1])
