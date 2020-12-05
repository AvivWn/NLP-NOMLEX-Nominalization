import os
import random
from os.path import join
from collections import defaultdict
from argparse import ArgumentParser, Namespace

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from torch import optim

from noun_as_verb.model_based.pretrained_wrapper import PretrainedWrapper
from noun_as_verb import config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.multiprocessing.set_sharing_strategy('file_system')
seed = 42
pl.seed_everything(seed)
rand = random.Random(seed)

class CheckpointEveryNSteps(pl.Callback):
	"""
	Save a checkpoint every N steps, instead of Lightning's default.
	"""

	def __init__(self, save_step_frequency, prefix="checkpoint", use_modelcheckpoint_filename=False):
		"""
		Args:
			save_step_frequency: how often to save in steps
			prefix: add a prefix to the name, only used if
				use_modelcheckpoint_filename=False
			use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
			default filename, don't use ours.
		"""
		self.save_step_frequency = save_step_frequency
		self.prefix = prefix
		self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

	def on_batch_end(self, trainer: pl.Trainer, _):
		""" Check if we should save a checkpoint in this step """
		epoch = trainer.current_epoch
		global_step = trainer.global_step

		if global_step % self.save_step_frequency == 0 and trainer.checkpoint_callback.dirpath:
			if self.use_modelcheckpoint_filename:
				filename = trainer.checkpoint_callback.filename
			else:
				filename = f"{self.prefix}_epoch={epoch}_step={global_step}.ckpt"

			ckpt_path = join(trainer.checkpoint_callback.dirpath, filename)
			trainer.save_checkpoint(ckpt_path)

class ClassifierNetwork(torch.nn.Module):
	"""" A neural network that classifies a "type" based of a given tensor. """

	def __init__(self, in_dim, out_dim, dropout_p):
		super(ClassifierNetwork, self).__init__()
		self.dropout = torch.nn.Dropout(p=dropout_p)
		# self.fc1 = torch.nn.Linear(in_dim, int(in_dim / 2))
		# self.batch_norm1 = torch.nn.BatchNorm1d(int(in_dim / 2))
		# self.fc2 = torch.nn.Linear(int(in_dim / 2), int(in_dim / 8))
		# self.batch_norm2 = torch.nn.BatchNorm1d(int(in_dim / 8))
		# self.fc3 = torch.nn.Linear(int(in_dim / 8), out_dim)

		#self.fc1 = torch.nn.Linear(in_dim, int(in_dim / 2))
		#self.batch_norm1 = torch.nn.BatchNorm1d(int(in_dim / 2))
		#self.fc2 = torch.nn.Linear(int(in_dim / 2), out_dim)

		self.fc = torch.nn.Linear(in_dim, out_dim)

	def forward(self, x):
		#fc1_out = self.dropout(F.relu(self.batch_norm1(self.fc1(x))))
		#fc2_out = self.dropout(F.relu(self.batch_norm2(self.fc2(fc1_out))))
		#return self.fc3(fc2_out)

		#fc1_out = self.dropout(F.relu(self.batch_norm1(self.fc1(x))))
		#return self.fc2(fc1_out)

		return self.fc(x)

class TypesModel(pl.LightningModule):
	"""
	The model generalizes NOMLEX existing predicates to not closed list of predicates
	"""

	pretrained_wrapper: PretrainedWrapper # Wraps a pretrained model

	def __init__(self, hyper_params):
		super(TypesModel, self).__init__()

		if type(hyper_params) == dict:
			hyper_params = Namespace(**hyper_params)

		# Hyper parameters
		self.save_hyperparameters(hyper_params)
		# print(f"Hyper Parameters: {hyper_params}")

		self.tagset = hyper_params.tagset
		self.labels = list(self.tagset.keys())

		# Pretrained model
		self.pretrained_wrapper = PretrainedWrapper(hyper_params)
		self.pretrained_enc = self.pretrained_wrapper.get_model()
		pre_dim = self.pretrained_wrapper.get_dim()
		pre_max_len = self.pretrained_wrapper.get_max_len()

		# Example input
		#ex_sent = ex_mask = torch.ones((2, pre_max_len)).long()
		#ex_idx = torch.randint(pre_max_len, (2,)).long()
		#ex_tagset_id = torch.tensor([0,1]).long()
		#self.example_input_array = ((ex_sent, ex_mask, ex_idx, ex_idx, ex_idx, ex_idx, ex_idx), ex_tagset_id)

		# Freeze or not all the weights, according to the specified parameters
		for param in self.pretrained_enc.parameters():
			param.requires_grad = False #self.hparams.finetune_pretrained

		for param in list(self.pretrained_enc.children())[-2].parameters():
			param.requires_grad = self.hparams.finetune_pretrained

		# The last encoder layer will be trained, unless the task are trained separently without a finetune
		for param in list(self.pretrained_enc.children())[-1].parameters():
			param.requires_grad = self.hparams.finetune_pretrained

		# Classification network
		self.cls_net = ClassifierNetwork(pre_dim, len(self.labels), self.hparams.dropout)

	def forward(self, x):
		token_ids, token_mask, start_arg_idx, end_arg_idx, start_pred_idx, end_pred_idx, suitable_verb_idx = x
		batch_size = token_ids.shape[0]
		batch_range = torch.arange(batch_size)

		# truncate to longest sequence length in batch (usually much smaller than 512) to save GPU RAM
		max_length = token_mask.max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
		if max_length < token_ids.shape[1]:
			token_ids = token_ids[:, :max_length]
			token_mask = token_mask[:, :max_length]

		# for i in range(batch_size):
		# 	print(self.pretrained_wrapper.tokenizer.decode(list(token_ids[i].cpu())[:max_length]))

		enc_out = self.pretrained_enc(token_ids, token_mask)[0]

		# With or without context
		if self.hparams.context:
			start_pred_emb = enc_out[batch_range, start_pred_idx]
			end_pred_emb = enc_out[batch_range, end_pred_idx]
			start_arg_emb = enc_out[batch_range, start_arg_idx]
			end_arg_emb = enc_out[batch_range, end_arg_idx]
			#verb_emb = enc_out[batch_range, suitable_verb_idx]

			net_in = (start_arg_emb + end_arg_emb + start_pred_emb + end_pred_emb) / 4
		else:
			net_in = enc_out[batch_range, 0]

		net_out = self.cls_net(net_in)
		return net_out

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)

		# Hyper Parameters
		parser.add_argument('--lr', type=float, default=2e-5)
		parser.add_argument('--weight_decay', type=float, default=2e-3)
		parser.add_argument('--dropout', type=float, default=0.8)
		parser.add_argument('--finetune_pretrained', action='store_true')
		parser.add_argument('--context', action='store_true') # Using sentence context

		# Version Info
		parser.add_argument('--data_version', type=str, default='base')
		parser.add_argument('--experiment_version', type=str, default='debug')
		parser.add_argument('--tagset', type=str, default='syntactic')
		return parser



	def _get_path(self, file_name, suffix, specific_dataset=True):
		path = config.DATASETS_PATH
		path = join(path, self.hparams.data_version)

		if not specific_dataset:
			path = os.path.dirname(path)

		return join(path, f"{file_name}.{suffix}")

	def encode(self, tokens, arg_start_idx, arg_end_idx, predicate_idx, verb, tagset_type, all_sizes=False):
		features = self.pretrained_wrapper.encode(tokens, arg_start_idx, arg_end_idx,
												  predicate_idx, verb, tagset_type, self.hparams.context, all_sizes)

		if not features:
			return None

		# Exapnd the features to have an extra 0-dim, if they aren't already
		if len(features[0].shape) == 1:
			features = [t.unsqueeze(0) for t in features]

		return features

	def _pandas_to_pytorch(self, dataset: pd.DataFrame, dataset_type, encoded_examples:dict):
		dataset_tuples = []
		max_len = 0
		del_count = 0

		for row_info in tqdm(dataset.itertuples(), f"Processing {dataset_type} dataset", leave=True):
			(i, sentence, predicate_idx, predicate, is_verb, verb,
			arg_head_idx, arg_start_idx, arg_end_idx, arg,
			label, tagset_type) = tuple(row_info)

			assert "#" not in verb, "verb includes count!"
			assert "#" not in predicate, "predicate includes count!"

			if self.hparams.context:
				encoded_key = (sentence, predicate_idx, is_verb, verb,
							   arg_head_idx, arg_start_idx, arg_end_idx,
							   label, tagset_type)
			else:
				encoded_key = (verb, arg, label)

			# Generate the features for the current example
			features = encoded_examples.get(encoded_key, None)
			if not features:
				words = sentence.split()
				features = self.encode(words, arg_start_idx, arg_end_idx,
									   predicate_idx, verb, tagset_type)

				if not features:
					del_count += 1
					continue

				encoded_examples[encoded_key] = features

			tokens_length = features[1].max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
			max_len = max(max_len, tokens_length)

			label_id = self.tagset[label]
			dataset_tuples.append((*features, torch.tensor([label_id])))

		print(f"DEL={del_count}, MAX={max_len}")
		return TensorDataset(*(map(torch.cat, zip(*dataset_tuples))))

	def _prepare_dataset(self, dataset_type, encoded_examples=None):
		# Load the existing tensor dataset
		dataset_path = self._get_path(dataset_type, "pth")
		if config.LOAD_DATASET and os.path.exists(dataset_path):
			return torch.load(dataset_path)

		# Load the already encoded examples only if we haven't done it yet
		if encoded_examples == {}:
			encoded_examples.update(self._load_encoded_examples())

		# Load the csv dataset, and transform it into tensor dataset
		dataset_df = pd.read_csv(self._get_path(dataset_type, "csv"), sep="\t", header=None, keep_default_na=False)
		processed_dataset = self._pandas_to_pytorch(dataset_df, dataset_type, encoded_examples)
		torch.save(processed_dataset, dataset_path)
		return processed_dataset

	def _load_encoded_examples(self):
		keys_path = self._get_path("encoded_keys", "csv", specific_dataset=False)
		features_path = self._get_path("encoded_features", "pth", specific_dataset=False)
		if not os.path.exists(keys_path) or not os.path.exists(features_path):
			return {}

		# Load the pre-encoded features and their appropriate keys
		encoded_keys_df = pd.read_csv(keys_path, sep="\t", header=None, keep_default_na=False)
		encoded_keys = list(encoded_keys_df.itertuples(name=None, index=False))
		encoded_features = list(torch.load(features_path))
		return dict(zip(encoded_keys, encoded_features))

	def _save_encoded_examples(self, encoded_examples):
		keys_path = self._get_path("encoded_keys", "csv", specific_dataset=False)
		features_path = self._get_path("encoded_features", "pth", specific_dataset=False)
		if os.path.exists(keys_path) and os.path.exists(features_path):
			return

		# Save keys of all the examples
		encoded_keys = pd.DataFrame(encoded_examples.keys())
		encoded_keys.to_csv(keys_path, sep="\t", header=False, index=False)

		# Save the encoded features
		encoded_features = TensorDataset(*(map(torch.cat, zip(*encoded_examples.values()))))
		torch.save(encoded_features, features_path)

	def prepare_data(self):
		encoded_examples = {}
		self._prepare_dataset("train", encoded_examples)
		self._prepare_dataset("sanity", encoded_examples)
		self._prepare_dataset("val", encoded_examples)
		self._prepare_dataset("test", encoded_examples)
		self._save_encoded_examples(encoded_examples)

	def setup(self, stage):
		if stage == "fit" or stage is None:
			self.train_dataset = torch.load(self._get_path("train", "pth"))
			self.sanity_dataset = torch.load(self._get_path("sanity", "pth"))
			self.val_dataset = torch.load(self._get_path("val", "pth"))

		if stage == "test" or stage is None:
			self.test_dataset = torch.load(self._get_path("test", "pth"))



	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=64, num_workers=32)

	def val_dataloader(self):
		return [DataLoader(self.sanity_dataset, batch_size=256, num_workers=4),
				DataLoader(self.val_dataset, batch_size=256, num_workers=4)]

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=256, num_workers=4)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		return optimizer



	@staticmethod
	def _get_classification_report(y_true, y_hat, tagset):
		y_true = y_true.cpu().numpy()

		if len(y_true) == 0:
			default_report = {"weighted avg": defaultdict(float)}
			for label in tagset:
				default_report[label] = defaultdict(float)
			return default_report

		y_pred = torch.argmax(y_hat, dim=1)
		y_pred = y_pred.cpu().numpy()

		return classification_report(y_true, y_pred, zero_division=0, output_dict=True,
									 labels=list(tagset.values()), target_names=list(tagset.keys()))

	def _calculate_metrics(self, y, y_hat, step_type):
		cls_report = self._get_classification_report(y, y_hat, self.tagset)
		metrics = {f"f1": cls_report["weighted avg"]["f1-score"]}

		for label in set(self.labels):
			metrics[f"{label}/f1"] = cls_report[label]["f1-score"]

		metrics = {k: torch.tensor([v]).to(y.device) for k, v in metrics.items()}
		loss = F.cross_entropy(y_hat, y)
		metrics["loss"] = loss.unsqueeze(0)

		for_logs = dict([(k + "/" + step_type, v) for k, v in metrics.items()])
		metrics["log"] = for_logs

		# Save also the labels and predictions
		metrics["y"] = y
		metrics["y_hat"] = y_hat

		return metrics

	@staticmethod
	def _mean_metrics(outputs, epoch_type):
		metrics_outputs = defaultdict(list)

		for output in outputs:
			for metric, value in output.items():
				if metric != "log":
					metrics_outputs[metric].append(value.mean())

		mean_metrics = dict([(metric, torch.tensor(values).mean()) for metric, values in metrics_outputs.items()])
		for_logs = dict([(k + "/" + epoch_type, v) for k, v in mean_metrics.items()])
		mean_metrics["log"] = for_logs

		return mean_metrics

	def on_fit_start(self):
		self.logger.log_hyperparams(params=self.hparams)

	def training_step(self, batch, batch_idx):
		*x, tagset_idx, y = batch
		y_hat = self.forward(x)
		metrics = self._calculate_metrics(y, y_hat, "train")
		return metrics

	def validation_step(self, batch, batch_idx, dataloader_idx):
		*x, tagset_idx, y = batch
		y_hat = self.forward(x)
		step_type = "sanity" if dataloader_idx == 0 else "val"
		metrics = self._calculate_metrics(y, y_hat, step_type)
		return metrics

	def validation_epoch_end(self, outputs):
		total_epoch_metrics = {"log": {}}

		for dataloader_idx in range(len(outputs)):
			epoch_type = "sanity" if dataloader_idx == 0 else "val"

			y = []
			y_hat = []
			for output in outputs[dataloader_idx]:
				y.append(output["y"])
				y_hat.append(output["y_hat"])

			# Calcuate over everything (more accurate)
			metrics = self._calculate_metrics(torch.cat(y), torch.cat(y_hat), epoch_type)
			total_epoch_metrics["log"].update(metrics["log"])

			# Calculate by mean
			# mean_metrics = self._mean_metrics(outputs[dataloader_idx], epoch_type)
			# total_epoch_metrics["log"].update(mean_metrics["log"])

		return total_epoch_metrics

	def test_step(self, batch, batch_idx):
		*x, tagset_idx, y = batch
		y_hat = self.forward(x)
		metrics = self._calculate_metrics(y, y_hat, "test")
		return metrics

	def test_epoch_end(self, outputs):
		mean_metrics = self._mean_metrics(outputs, "test")
		total_mean_metrics ={"log": mean_metrics["log"]}
		return total_mean_metrics