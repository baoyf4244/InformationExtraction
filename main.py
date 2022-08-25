from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import LightningModule, LightningDataModule


if __name__ == '__main__':
    cli = LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)
