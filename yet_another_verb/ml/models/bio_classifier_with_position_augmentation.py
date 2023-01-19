from copy import deepcopy

import numpy as np
import torch
from seqeval.metrics.v1 import classification_report
from seqeval.scheme import IOB1

from yet_another_verb.ml.data_modules.defaults import IGNORED_LABEL_ID
from yet_another_verb.ml.models import TokenClassifier
from yet_another_verb.ml.models.token_classifier import TRAIN_STEP, TEST_STEP, LOSS

F1_SCORE, RECALL, PRECISION = 'f1-score', 'recall', 'precision'
METRICS = [F1_SCORE, RECALL, PRECISION]
BEGIN_PREFIX, INSIDE_PREFIX = 'B-', 'I-'


class BIOClassifierWithPositionAugmentation(TokenClassifier):
	def __init__(self, augmentation_factor: int = 1, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.save_hyperparameters()

		self.begin_tags = [tag for tag in self.tagset if tag.startswith(BEGIN_PREFIX)]
		self.inside_tags = [tag for tag in self.tagset if tag.startswith(INSIDE_PREFIX)]

	def setup(self, stage=None):
		for step_type in [TRAIN_STEP, TEST_STEP] + self.hparams.val_names:
			self._define_logger_metric(step_type, LOSS, 'min')

			for metric in METRICS:
				self._define_logger_metric(step_type, metric, 'max')

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

	@staticmethod
	def replace_and_shift(tensor, old_idx, new_idx, length):
		old_info = tensor[old_idx: old_idx + length].clone()

		if new_idx < old_idx:
			tensor[new_idx + length: old_idx + length] = tensor[new_idx: old_idx].clone()
			tensor[new_idx: new_idx + length] = old_info
		else:
			tensor[old_idx: new_idx] = tensor[old_idx + length: new_idx + length].clone()
			tensor[new_idx: new_idx + length] = old_info

	def _augment_batch(self, batch):
		augmented_batch = deepcopy(batch)

		for token_ids, attention_mask, y in zip(*augmented_batch):
			begin_indices = torch.cat([(y == self.tagset[b_tag]).nonzero(as_tuple=False) for b_tag in self.begin_tags])

			if len(begin_indices) == 0:
				continue

			begin_idx = begin_indices[torch.multinomial(torch.ones(len(begin_indices)), 1).item()]
			begin_tag = self.inversed_tagset[y[begin_idx].item()]
			inside_tag = INSIDE_PREFIX + begin_tag[len(BEGIN_PREFIX):]
			inside_tag_encoded = self.tagset[inside_tag]

			arg_len = 1
			while y[begin_idx + arg_len] == inside_tag_encoded:
				arg_len += 1

			sentence_length = attention_mask.nonzero(as_tuple=False)[-1].item()
			new_begin_idx = np.random.choice(range(1, sentence_length - arg_len))

			for values in [token_ids, attention_mask, y]:
				self.replace_and_shift(values, begin_idx, new_begin_idx, arg_len)

		return augmented_batch

	def _in_batch_augmentation(self, batch, batch_idx):
		*batch_x, batch_y = batch
		batch_token_ids, batch_attention_mask = batch_x
		batch = [batch_token_ids, batch_attention_mask, batch_y]

		batch_augmentations = [batch]
		for i in range(self.hparams.augmentation_factor):
			batch_augmentations.append(self._augment_batch(batch))

		return [torch.cat(all_values) for all_values in zip(*batch_augmentations)]
