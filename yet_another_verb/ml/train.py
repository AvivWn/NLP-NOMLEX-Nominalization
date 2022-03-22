import argparse
import os.path
import os
from typing import Optional

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
import torch.multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')


class WandbCustomTrainer(Trainer):
	PARAMS_TO_LOG = ["gpus", "strategy", "val_check_interval", "accumulate_grad_batches"]

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		params = {name: kwargs[name] for name in self.PARAMS_TO_LOG}
		self.logger.log_hyperparams(argparse.Namespace(**params))

	@property
	def log_dir(self) -> Optional[str]:
		if not self.is_global_zero:
			return super().log_dir

		if isinstance(self.logger, WandbLogger):
			_ = self.logger.experiment  # force the experiment creation
			dirpath = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
			dirpath = self.training_type_plugin.broadcast(dirpath)
		else:
			dirpath = super().log_dir

		return dirpath


def main():
	LightningCLI(
		model_class=LightningModule,
		datamodule_class=LightningDataModule,
		trainer_class=WandbCustomTrainer,
		subclass_mode_model=True,
		subclass_mode_data=True,
		run=True
	)


if __name__ == "__main__":
	main()
