import os
from typing import List

import torch
import torch.nn.functional as f
from torch import optim
from pytorch_lightning import LightningModule
from transformers import AutoModel
from pytorch_lightning.loggers import WandbLogger
from seqeval.metrics.v1 import classification_report
from seqeval.scheme import IOB1

from yet_another_verb.file_handlers.labels_file_handler import LabelsFileHandler
from yet_another_verb.ml.data_files import LABELS_FILE
from yet_another_verb.ml.data_modules.defaults import IGNORED_LABEL_ID
from yet_another_verb.ml.utils import labels_to_tagset


TRAIN_STEP, TEST_STEP = "train", "test"
F1_SCORE, RECALL, PRECISION = 'f1-score', 'recall', 'precision'
METRICS = [F1_SCORE, RECALL, PRECISION]
LOSS = 'loss'


class TokenClassifier(LightningModule):
	def __init__(
			self, pretrained_model: str, data_dir: str,
			lr: float, weight_decay: float, dropout: float, label_smoothing: float,
			val_names: List[str]
	):
		super().__init__()
		labels_path = os.path.join(data_dir, LABELS_FILE)
		self.tagset = labels_to_tagset(LabelsFileHandler.load(labels_path))
		self.inversed_tagset = {v: k for k, v in self.tagset.items()}

		self.pretrained_model = AutoModel.from_pretrained(pretrained_model)
		hidden_dim = self.pretrained_model.embeddings.position_embeddings.weight.size(1)

		self.fc = torch.nn.Linear(hidden_dim, len(self.tagset))
		self.dropout = torch.nn.Dropout(p=dropout)
		self.loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORED_LABEL_ID, label_smoothing=label_smoothing)

		self.save_hyperparameters()

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

	def setup(self, stage=None):
		# self._define_logger_metric(TRAIN_STEP, LOSS, 'min')

		for step_type in [TRAIN_STEP, TEST_STEP] + self.hparams.val_names:
			self._define_logger_metric(step_type, LOSS, 'min')

			for metric in METRICS:
				self._define_logger_metric(step_type, metric, 'max')

	def _decode_labels(self, labels_ids) -> list:
		return [self.inversed_tagset[l_id] for l_id in labels_ids]

	def _get_classification_report(self, y_true, y_pred):
		if len(y_true) == 0:
			return {}

		y_true_labels, y_pred_labels = [], []
		for true_ids, pred_ids in zip(y_true, y_pred):
			relevant_ids = true_ids != IGNORED_LABEL_ID
			y_true_labels.append(self._decode_labels(true_ids[relevant_ids]))
			y_pred_labels.append(self._decode_labels(pred_ids[relevant_ids]))

		return classification_report(y_true_labels, y_pred_labels, output_dict=True, scheme=IOB1, zero_division=0)

	def _calculate_and_log_metrics(self, y_true, y_pred, step_type: str):
		cls_report = self._get_classification_report(y_true.cpu().numpy(), y_pred.cpu().numpy())

		for metric in METRICS:
			self.log(self._combine_metric_and_step(step_type, metric), float(cls_report["weighted avg"][metric]))

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

		# truncate to longest sequence length in batch to save GPU RAM
		max_length = attention_mask.max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
		if max_length < attention_mask.shape[1]:
			token_ids = token_ids[:, :max_length]
			attention_mask = attention_mask[:, :max_length]

		y = y[:, :max_length]
		return token_ids, attention_mask, y

	def training_step(self, batch, batch_idx):
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
