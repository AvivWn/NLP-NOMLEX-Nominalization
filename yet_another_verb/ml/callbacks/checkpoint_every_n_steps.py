from os.path import join

import pytorch_lightning as pl


class CheckpointEveryNSteps(pl.Callback):
	"""
	Save a checkpoint every N steps, instead of Lightning's default.
	"""

	def __init__(self, n_steps, prefix="checkpoint", use_modelcheckpoint_filename=False):
		"""
		Args:
			n_steps: how often to save in steps
			prefix: add a prefix to the name, only used if
				use_modelcheckpoint_filename=False
			use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
			default filename, don't use ours.
		"""
		self.n_steps = n_steps
		self.prefix = prefix
		self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

	def on_batch_end(self, trainer: pl.Trainer, _):
		""" Check if we should save a checkpoint in this step """
		epoch = trainer.current_epoch
		global_step = trainer.global_step

		if global_step % self.n_steps == 0 and trainer.checkpoint_callback.dirpath:
			if self.use_modelcheckpoint_filename:
				filename = trainer.checkpoint_callback.filename
			else:
				filename = f"{self.prefix}_epoch={epoch}_step={global_step}.ckpt"

			ckpt_path = join(trainer.checkpoint_callback.dirpath, filename)
			trainer.save_checkpoint(ckpt_path)