import factory
import argparse
from ner.dataset import NERDataModule
from ner.models import MRCNERModule, BiLSTMLanNERModule
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='mrc', type=str)
    # args, _ = parser.parse_known_args()
    cli = LightningCLI(MRCNERModule, NERDataModule)
