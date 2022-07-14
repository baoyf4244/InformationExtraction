from ner.models import BiLSTMLanNERModule
from ner.dataset import NERDataModule
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == '__main__':
    cli = LightningCLI(BiLSTMLanNERModule, NERDataModule)
