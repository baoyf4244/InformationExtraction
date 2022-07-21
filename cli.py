from ner.models import *
from ner.dataset import *
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser


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


class MyLightningCli(LightningCLI):
    def __init__(self, *args, **kwargs):
        super(MyLightningCli, self).__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments('model_name', 'data.model_name')


if __name__ == '__main__':
    parser = LightningArgumentParser()
    parser.add_argument('--model_name', default='bilstm-lan', type=str)

    cli = MyLightningCli(BiLSTMLanNERModule, NERDataModule)