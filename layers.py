import torch
import torch.nn as nn
import factory


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, act_fun, dropout_rate):
        super(NonLinear, self).__init__()
        self.act_fun = factory.act_func_factory(act_fun)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        return self.act_fun(self.dropout(self.linear(inputs)))


class MultiNonLinearLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, act_fun='relu', dropout_rate=0.1):
        super(MultiNonLinearLayer, self).__init__()
        self.num_layers = num_layers
        self.act_fun = act_fun
        self.dropout_rate = dropout_rate

        self.check_params()
        self.layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(NonLinear(input_size, hidden_size, self.act_fun[i], self.dropout_rate[i]))
            input_size = hidden_size

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def check_params(self):
        if isinstance(self.act_fun, str):
            self.act_fun = [self.act_fun] * (self.num_layers - 1)
        else:
            assert len(self.ct_fun) == (self.num_layers - 1), 'act_fun param must be str or list of str with num_layers length'

        if isinstance(self.dropout_rate, float):
            self.dropout_rate = [self.dropout_rate] * (self.num_layers - 1)
        else:
            assert len(
                self.dropout_rate) == (self.num_layers - 1), 'dropout_rate param must be float or list of float with num_layers length'
