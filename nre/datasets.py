from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class NREDataSet(Dataset):
    def __init__(self, sent_file, relations_file, tokenizer, char_tokenizer=None, tuples_file=None):
        super(NREDataSet, self).__init__()
        self.sent_file = sent_file
        self.tuples_file = tuples_file
        self.relations_file = relations_file
        self.relations = self.get_relations()

    def get_relations(self):
        relations = []
        with open(self.relations_file, encoding='utf-8') as f:
            for line in f:
                relations.append(line.strip())
        return relations

    def make_dataset(self):
        datasets = []
        with open(self.sent_file, encoding='utf-8') as f:
            for line in f:
                line
        return datasets

    def __getitem__(self, index) -> T_co:
        pass
