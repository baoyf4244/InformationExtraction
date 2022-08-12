import torch
import argparse
# from ner.datasets import NERDataModule
# from ner.models import MRCNERModule, BiLSTMLanNERModule, BiLSTMCrfNERModule
from kpe.models import Seq2SeqKPEModule
from kpe.datasets import KPEDataModule
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='mrc', type=str)
    # args, _ = parser.parse_known_args()
    # cli = LightningCLI(MRCNERModule, NERDataModule)

    # cli = LightningCLI(BiLSTMCrfNERModule, NERDataModule)
    # cli = LightningCLI(BiLSTMLanNERModule, NERDataModule)
    cli = LightningCLI(Seq2SeqKPEModule, KPEDataModule)