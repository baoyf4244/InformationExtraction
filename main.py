from cli import MyLightningCLI
from ner.models import MRCNERModule
from ner.dataset import MRCDataModule
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == '__main__':
    cli = MyLightningCLI(MRCNERModule, MRCDataModule)
