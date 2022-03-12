import os
from collections import defaultdict

import torch
import torch.nn.functional as f
from sklearn.metrics import classification_report
from torch import optim
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from transformers import AutoModel

# os.environ["TOKENIZERS_PARALLELISM"] = "true"
# torch.multiprocessing.set_sharing_strategy('file_system')
from yet_another_verb.ml.data_modules.defaults import IGNORED_LABEL_ID


class TokenClassifier(LightningModule):
	def __init__(self, pretrained_model: str, out_dim: int, lr: float, weight_decay: float, dropout: float):
		super().__init__()
		self.pretrained_model = AutoModel.from_pretrained(pretrained_model)
		hidden_dim = self.pretrained_model.embeddings.position_embeddings.weight.size(1)

		self.fc = torch.nn.Linear(hidden_dim, out_dim)
		self.dropout = torch.nn.Dropout(p=dropout)
		self.loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORED_LABEL_ID)

		self.save_hyperparameters()

	def forward(self, token_ids, attention_mask):
		# print("input", token_ids.shape, token_mask.shape)
		# truncate to longest sequence length in batch (usually much smaller than 512) to save GPU RAM
		# max_length = attention_mask.max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
		# if max_length < token_ids.shape[1]:
		# 	token_ids = token_ids[:, :max_length]
		# 	attention_mask = attention_mask[:, :max_length]

		# print("reshape", token_ids.shape, token_mask.shape)
		hidden_state = self.pretrained_model(token_ids, attention_mask)[0]
		# print("pretrained", hidden_state.shape)
		return self.fc(self.dropout(hidden_state)).permute(0, 2, 1)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		return optimizer
	#
	# @staticmethod
	# def _get_classification_report(y_true, y_hat, tagset):
	# 	y_true = y_true.cpu().numpy()
	#
	# 	if len(y_true) == 0:
	# 		default_report = {"weighted avg": defaultdict(float)}
	# 		for label in tagset:
	# 			default_report[label] = defaultdict(float)
	# 		return default_report
	#
	# 	y_pred = torch.argmax(y_hat, dim=1)
	# 	y_pred = y_pred.cpu().numpy()
	#
	# 	return classification_report(
	# 		y_true, y_pred, zero_division=0, output_dict=True,
	# 		labels=list(tagset.values()), target_names=list(tagset.keys()))
	#
	# def _calculate_metrics(self, y, y_hat):
	# 	cls_report = self._get_classification_report(y, y_hat, self.tagset)
	# 	metrics = {f"f1": cls_report["weighted avg"]["f1-score"]}
	#
	# 	for label in self.tagset:
	# 		metrics[f"{label}/f1"] = cls_report[label]["f1-score"]
	#
	# 	metrics = {k: torch.tensor([v]).to(y.device) for k, v in metrics.items()}
	# 	loss = f.cross_entropy(y_hat, y)
	# 	metrics["loss"] = loss.unsqueeze(0)
	#
	# 	# Save also the labels and predictions
	# 	# metrics["y"] = y
	# 	# metrics["y_hat"] = y_hat
	#
	# 	return metrics
	#
	# @staticmethod
	# def _mean_metrics(outputs, epoch_type):
	# 	metrics_outputs = defaultdict(list)
	#
	# 	for output in outputs:
	# 		for metric, value in output.items():
	# 			if metric != "log":
	# 				metrics_outputs[metric].append(value.mean())
	#
	# 	mean_metrics = dict([(metric, torch.tensor(values).mean()) for metric, values in metrics_outputs.items()])
	# 	for_logs = dict([(k + "/" + epoch_type, v) for k, v in mean_metrics.items()])
	# 	mean_metrics["log"] = for_logs
	#
	# 	return mean_metrics
	#
	# def _log_metrics(self, metrics, step_type):
	# 	for metric_type, value in metrics.items():
	# 		self.log(f"{metric_type}/{step_type}", value)

	def _get_loss_and_accuracy(self, batch):
		*x, y = batch
		logits = self(*x)
		preds = torch.argmax(logits, dim=1)
		loss = self.loss(logits, y)
		acc = accuracy(preds, y, ignore_index=IGNORED_LABEL_ID)
		return loss, acc

	def training_step(self, batch, batch_idx):
		loss, acc = self._get_loss_and_accuracy(batch)
		self.log('train/loss', loss)
		self.log('train/accuracy', acc)
		return {"train_loss": loss}

	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		loss, acc = self._get_loss_and_accuracy(batch)
		self.log('val/loss', loss)
		self.log('val/accuracy', acc)
		return {"val_loss": loss}

	# def validation_epoch_end(self, outputs):
	# 	for dataloader_idx in range(len(outputs)):
	# 		epoch_type = "control" if dataloader_idx == 0 else "val"
	#
	# 		y = []
	# 		y_hat = []
	# 		for output in outputs[dataloader_idx]:
	# 			y.append(output["y"])
	# 			y_hat.append(output["y_hat"])
	#
	# 		# Calcuate over everything (more accurate)
	# 		metrics = self._calculate_metrics(torch.cat(y), torch.cat(y_hat))
	# 		self._log_metrics(metrics, epoch_type)
	#
	# 		# Calculate by mean
	# 		# mean_metrics = self._mean_metrics(outputs[dataloader_idx], epoch_type)
	# 		# total_epoch_metrics["log"].update(mean_metrics["log"])
	#
	# def test_step(self, batch, batch_idx):
	# 	*x, y = batch
	# 	y_hat = self.forward(*x)
	# 	metrics = self._calculate_metrics(y, y_hat)
	# 	return metrics
	#
	# def test_epoch_end(self, outputs):
	# 	mean_metrics = self._mean_metrics(outputs, "test")
	# 	total_mean_metrics = {"log": mean_metrics["log"]}
	# 	return total_mean_metrics
