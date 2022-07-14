import torch.nn.functional as F
from ner.dataset import MRCNERDataset, FlatNERDataSet


def act_func_factory(name):
    act_funcs = {
        'relu': F.relu,
        'sigmod': F.sigmoid,
        'softmax': F.softmax,
        'gelu': F.gelu,
        'tanh': F.tanh
    }

    if name in act_funcs:
        return act_funcs[name]
    else:
        raise NotImplementedError('激活函数仅支持[relu, sigmod, softmax, gelu, tanh]')


def get_dataset(model_name):
    dataset = {
        'MRC': MRCNERDataset,
        'BILSTM-LAN': FlatNERDataSet
    }
    model_name = model_name.upper()
    if model_name in dataset:
        return dataset[model_name]
    else:
        raise NotImplementedError
