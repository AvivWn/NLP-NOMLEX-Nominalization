# from pytorch_lightning import LightningModule, LightningDataModule
# from pytorch_lightning.utilities.cli import LightningCLI
#
# from yet_another_verb.ml.models import TokenClassifier
#
#
# def main():
# 	# LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True, run=False)
# 	LightningCLI(TokenClassifier, LightningDataModule, run=True) #, subclass_mode_model=True, subclass_mode_data=True, run=False)
#
#
# if __name__ == "__main__":
# 	main()

import os.path
from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from yet_another_verb.ml.data_modules import TaggedTokensDataModule
from yet_another_verb.ml.data_modules.defaults import N_DATA_LOADING_WORKERS
from yet_another_verb.ml.models import TokenClassifier

import torch.multiprocessing
# torch.set_num_threads(N_DATA_LOADING_WORKERS)
torch.multiprocessing.set_sharing_strategy('file_system')


class WandbCustomTrainer(Trainer):
	@property
	def log_dir(self) -> Optional[str]:
		if isinstance(self.logger, WandbLogger):
			_ = self.logger.experiment  # force the experiment creation
			return os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)

		return super().log_dir


def main():
	LightningCLI(LightningModule, LightningDataModule, trainer_class=WandbCustomTrainer, subclass_mode_model=True, subclass_mode_data=True, run=True) # trainer_class=WandbCustomTrainer
# cli = LightningCLI(run=True) #, save_config_multifile=True) #, subclass_mode_model=True, subclass_mode_data=True, run=False)
# --config ../yet_another_verb/ml/train_configs/verbal_bio_config.yaml


if __name__ == "__main__":
	main()
