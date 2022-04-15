from typing import List

import torch
import torch.nn.functional as f
from torch import optim
from pytorch_lightning import LightningModule
from transformers import AutoModel
from pytorch_lightning.loggers import WandbLogger

from yet_another_verb.ml.data_modules.defaults import IGNORED_LABEL_ID
from yet_another_verb.ml.utils import labels_to_tagset


TRAIN_STEP, TEST_STEP = "train", "test"
LOSS = 'loss'


class TokenClassifier(LightningModule):
	def __init__(
			self, pretrained_model: str, labels: List[str],
			lr: float, weight_decay: float, dropout: float, label_smoothing: float,
			val_names: List[str], freeze_embeddings: bool = False, freeze_layers: str = ""
	):
		super().__init__()
		self.tagset = labels_to_tagset(labels)
		self.inversed_tagset = {v: k for k, v in self.tagset.items()}

		self.pretrained_model = AutoModel.from_pretrained(pretrained_model)
		hidden_dim = self.pretrained_model.embeddings.position_embeddings.weight.size(1)

		self.fc = torch.nn.Linear(hidden_dim, len(self.tagset))
		self.dropout = torch.nn.Dropout(p=dropout)
		self.loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORED_LABEL_ID, label_smoothing=label_smoothing)

		self.save_hyperparameters()
		self._freeze_demanded()

	def _freeze_demanded(self):
		if self.hparams.freeze_embeddings:
			for param in list(self.pretrained_model.embeddings.parameters()):
				param.requires_grad = False
			print("Froze Embedding Layer")

		# freeze_layers is a string "1,2,3" representing layer number
		if self.hparams.freeze_layers is not "":
			layer_indexes = [int(x) for x in self.hparams.freeze_layers.split(",")]
			for layer_idx in layer_indexes:
				for param in list(self.pretrained_model.encoder.layer[layer_idx].parameters()):
					param.requires_grad = False
				print("Froze Layer: ", layer_idx)

	def forward(self, token_ids, attention_mask):
		hidden_state = self.pretrained_model(token_ids, attention_mask)[0]
		fc_out = self.fc(self.dropout(hidden_state)).permute(0, 2, 1)
		return fc_out

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		return optimizer

	@staticmethod
	def _combine_metric_and_step(step_type, metric):
		return f"{step_type}/{metric}"

	def _define_logger_metric(self, step_type: str, metric: str, goal: str):
		if isinstance(self.logger, WandbLogger):
			name = self._combine_metric_and_step(step_type, metric)
			self.logger.experiment.define_metric(name=name, goal=goal, summary='best')

	def _decode_labels(self, labels_ids) -> list:
		return [self.inversed_tagset[l_id] for l_id in labels_ids]

	def _calculate_and_log_metrics(self, y_true, y_pred, step_type: str):
		pass

	def _in_batch_augmentation(self, batch, batch_idx):
		return batch

	def _handle_predict_and_loss(self, batch, step_type):
		*x, y = batch
		logits = self(*x)
		preds = torch.argmax(logits, dim=1)
		loss = self.loss(logits, y)
		self.log(self._combine_metric_and_step(step_type, LOSS), loss, add_dataloader_idx=False)
		return {LOSS: loss, "y": y, "preds": preds}

	def on_before_batch_transfer(self, batch, dataloader_idx):
		*x, y = batch
		token_ids, attention_mask = x

		# Filter out any unknown tags
		# unknown_idxs = [i for i in range(len(y)) if y[i] not in self.inversed_tagset]
		# unknown_tags = torch.cat([(y == self.tagset[tag]).nonzero(as_tuple=False) for tag in self.inversed_tagset])
		# y[unknown_idxs] = self.tagset['O']

		# truncate to longest sequence length in batch to save GPU RAM
		max_length = attention_mask.max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
		if max_length < attention_mask.shape[1]:
			token_ids = token_ids[:, :max_length]
			attention_mask = attention_mask[:, :max_length]

		y = y[:, :max_length]
		return token_ids, attention_mask, y

	def training_step(self, batch, batch_idx):
		batch = self._in_batch_augmentation(batch, batch_idx)
		logs = self._handle_predict_and_loss(batch, TRAIN_STEP)
		return logs

	def validation_step(self, batch, batch_idx, val_idx=0):
		logs = self._handle_predict_and_loss(batch, self.hparams.val_names[val_idx])
		return logs

	def _log_metrics_on_epoch_outputs(self, epoch_outputs, step_type):
		y, preds = [], []
		max_length = 0

		for step_output in epoch_outputs:
			y.append(step_output["y"])
			preds.append(step_output["preds"])
			max_length = max(max_length, y[-1].shape[1])

		for i in range(len(y)):
			padding_info = (1, max_length - y[i].shape[1] - 1)
			y[i] = f.pad(y[i], padding_info, value=IGNORED_LABEL_ID)
			preds[i] = f.pad(preds[i], padding_info, value=IGNORED_LABEL_ID)

		self._calculate_and_log_metrics(torch.cat(y), torch.cat(preds), step_type)

	def validation_epoch_end(self, outputs):
		for val_idx, val_outputs in enumerate(outputs):
			self._log_metrics_on_epoch_outputs(val_outputs, self.hparams.val_names[val_idx])

	# def test_step(self, batch, batch_idx):
	# 	logs = self._handle_predict_and_loss(batch, TEST_STEP)
	# 	return logs
	#
	# def test_epoch_end(self, outputs):
	# 	self._log_metrics_on_epoch_outputs(outputs, TEST_STEP)
