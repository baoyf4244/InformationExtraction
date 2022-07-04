import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ner.trainer import SpanNERTrainer
from ner.dataset import MRCNERDataset
from transformers import BertTokenizer

if __name__ == '__main__':
    model = SpanNERTrainer()
    train_filename = 'D:/code/NLP/InformationExtraction/data/train.txt'
    test_filename = 'D:/code/NLP/InformationExtraction/data/test.txt'
    tag_filename = 'D:/code/NLP/InformationExtraction/data/questions.json'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_dataset = MRCNERDataset(train_filename, tag_filename, tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=MRCNERDataset.collocate_fn, num_workers=0)

    test_dataset = MRCNERDataset(test_filename, tag_filename, tokenizer)
    test_dataloader = DataLoader(test_dataset, collate_fn=MRCNERDataset.collocate_fn, num_workers=0)

    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, test_dataloader)