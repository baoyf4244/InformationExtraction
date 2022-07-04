import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from collections import defaultdict


class MRCNERDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        filename: path to mrc-ner style json
        tokenizer: BertTokenizer
    """
    def __init__(self, filename, tag2query_file, tokenizer: BertTokenizer, max_len=512):
        self.filename = filename
        self.tag2query_file = tag2query_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2query = self.get_tag2query()
        self.dataset = self.make_dataset()

    def get_tag2query(self):
        with open(self.tag2query_file, encoding='utf-8') as f:
            tags = json.load(f)

        tag2query = {}
        for tag in tags.values():
            query_tokens = ['[CLS]'] + self.tokenizer.tokenize(tag['query']) + ['[SEP]']
            query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
            tag2query[tag['tag'].lower()] = query_ids
        return tag2query

    def make_dataset(self):
        dataset = []
        with open(self.filename, encoding='utf-8') as f:
            for i, line in enumerate(f):
                segments = line.strip().split()
                context_tokens = []
                offsets = defaultdict(list)
                start_tags = []
                end_tags = []
                for segment in segments:
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


if __name__ == '__main__':
    filename = 'D:/code/NLP/InformationExtraction/data/test.txt'
    tag_filename = 'D:/code/NLP/InformationExtraction/data/questions.json'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = MRCNERDataset(filename, tag_filename, tokenizer)
    print(dataset[1])
