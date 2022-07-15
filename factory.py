import torch.nn.functional as F
from ner.dataset import MRCNERDataset, FlatNERDataSet
from ner.models import MRCNERModule, BiLSTMLanNERModule, BiLSTMCrfNERModule


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
        'BILSTM-LAN': FlatNERDataSet,
        'BILSTM-CRF': FlatNERDataSet
    }
    model_name = model_name.upper()
    if model_name in dataset:
        return dataset[model_name]
    else:
        raise NotImplementedError


def get_model(model_name):
    models = {
        'MRC': MRCNERModule,
        'BILSTM-LAN': BiLSTMLanNERModule,
        'BILSTM-CRF': BiLSTMCrfNERModule
    }

    model_name = model_name.upper()
    if model_name in models:
        return models[model_name]
    else:
        raise NotImplementedError

