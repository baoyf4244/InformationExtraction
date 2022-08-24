import torch
import argparse
# from ner.datasets import NERDataModule
# from ner.models import MRCNERModule, BiLSTMLanNERModule, BiLSTMCrfNERModule
from kpe.models import Seq2SeqKPEModule
from kpe.datasets import KPEDataModule
from nre.models import Seq2SeqPTRNREModule, Seq2SeqNREModule
from nre.datasets import NREDataModule
from pytorch_lightning.utilities.cli import LightningCLI


# torch.set_num_threads(2)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='mrc', type=str)
    # args, _ = parser.parse_known_args()
    # cli = LightningCLI(MRCNERModule, NERDataModule)

    # cli = LightningCLI(BiLSTMCrfNERModule, NERDataModule)
    # cli = LightningCLI(BiLSTMLanNERModule, NERDataModule)
    # cli = LightningCLI(Seq2SeqKPEModule, KPEDataModule)
    # cli = LightningCLI(Seq2SeqPTRNREModule, NREDataModule)
    cli = LightningCLI(Seq2SeqNREModule, NREDataModule)