import torch.nn.functional as F


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





