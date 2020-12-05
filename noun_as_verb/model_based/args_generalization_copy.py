import os
import math
import random
from os.path import join
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from itertools import chain
from copy import deepcopy

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Sampler, DistributedSampler
from sklearn.metrics import classification_report
from torch import optim

from noun_as_verb.model_based.pretrained_wrapper import PretrainedWrapper
from noun_as_verb.constants.dataset_constants import ARGS_TAG, NOUNS_TAG
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

class DoubleTaskDataset(TensorDataset):
	""" Custom dataset that contains examples of two tasks, and "knows" to seperate them by indexes. """

	def __init__(self, *tensors, split_idxs_to_tagsets):
		super(DoubleTaskDataset, self).__init__(*tensors)
		assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
		self.tensors = tensors
		tagset_idxs = tensors[-2]
		self.args_idxs, self.nouns_idxs = split_idxs_to_tagsets(tagset_idxs)
		self.args_idxs = self.args_idxs.numpy()
		self.nouns_idxs = self.nouns_idxs.numpy()

	def __getitem__(self, index):
		return tuple(tensor[index] for tensor in self.tensors)

	def __len__(self):
		return self.tensors[0].size(0)

	def get_args_idxs(self):
		return self.args_idxs

	def get_nouns_idxs(self):
		return self.nouns_idxs

class DoubleTaskSampler(Sampler):
	"""
		Custom sampler that assures that every batch will be balanced with regard the two tasks.
	    The order of sampled indices is important to achieve balance.
	"""

	def __init__(self, data_source:DoubleTaskDataset):
		super(DoubleTaskSampler, self).__init__(data_source)
		self.data_source = data_source

	def __iter__(self):
		x = self.data_source.get_args_idxs()
		args_idxs = list(self.data_source.get_args_idxs())
		nouns_idxs = list(self.data_source.get_nouns_idxs())

		rand.shuffle(args_idxs)
		rand.shuffle(nouns_idxs)

		# Balance the number of examples of the two tasks
		if len(args_idxs) > len(nouns_idxs):
			nouns_idxs += rand.choices(nouns_idxs, k=len(args_idxs)-len(nouns_idxs))
			# nouns_idxs += rand.sample(nouns_idxs, k=len(args_idxs)-len(nouns_idxs))
		else:
			args_idxs += rand.choices(args_idxs, k=len(nouns_idxs)-len(args_idxs))
			# args_idxs += rand.sample(args_idxs, k=len(nouns_idxs)-len(args_idxs))
		assert len(args_idxs) == len(nouns_idxs)

		# Combine the two tasks in a way that assures that every batch we be balanced
		# Each batch will be something like- ARGS, NOUNS, ARGS, NOUNS, ...
		combined_idxs = chain(*zip(args_idxs, nouns_idxs))
		return iter(combined_idxs)

	def __len__(self):
		n_args = len(self.data_source.get_args_idxs())
		n_nouns = len(self.data_source.get_nouns_idxs())

		return max(n_args, n_nouns) * 2

class DoubleTaskDistributedSampler(DistributedSampler):
	"""
		Custom distributed sampler that wraps the custom standard sampler.
		It uses the indices sampled by the given sampler, and partition them
		in a way that do not change the order of indices that the given sampler sampled.
	"""

	def __init__(self, sampler:DoubleTaskSampler, num_replicas, rank):
		super(DoubleTaskDistributedSampler, self).__init__(sampler.data_source, num_replicas=num_replicas, rank=rank, shuffle=False)
		self.sampler = sampler
		self.num_replicas = num_replicas
		self.rank = rank
		self.num_samples = int(math.ceil(len(self.sampler) * 1.0 / self.num_replicas))
		self.total_size = self.num_samples * self.num_replicas

	def __iter__(self):
		indices = list(self.sampler.__iter__())

		# add extra samples to make it evenly divisible
		n_extra_indices = self.total_size-len(indices)
		indices += indices[(len(indices)%2):n_extra_indices+len(indices)%2]
		assert len(indices)%self.num_replicas == 0

		start_idx = self.num_samples * self.rank
		end_idx = self.num_samples * (self.rank+1)
		sub_indices = indices[start_idx:end_idx]
		assert len(sub_indices) == self.num_samples

		return iter(sub_indices)

	def __len__(self):
		return self.num_samples

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

class ArgsGeneralizaitonModel(pl.LightningModule):
	"""
	The model generalizes NOMLEX existing predicates to not closed list of predicates
	"""

	pretrained_wrapper: PretrainedWrapper # Wraps a pretrained model

	def __init__(self, args_tagset, nouns_tagset, hyper_params:Namespace):
		super(ArgsGeneralizaitonModel, self).__init__()

		# Hyper parameters
		self.save_hyperparameters(hyper_params)
		# print(f"Hyper Parameters: {hyper_params}")

		self.args_tagset = args_tagset
		self.args_labels = list(self.args_tagset.keys())
		self.n_args = len(self.args_tagset) # index 0

		self.nouns_tagset = nouns_tagset
		self.nouns_labels = list(self.nouns_tagset.keys())
		self.n_nouns = len(self.nouns_tagset) # index 1

		self.max_size = max(self.n_args, self.n_nouns)

		# Pretrained model
		self.pretrained_wrapper = PretrainedWrapper(hyper_params)
		self.pretrained_net = self.pretrained_wrapper.get_model()
		pre_dim = self.pretrained_wrapper.get_dim()
		pre_max_len = self.pretrained_wrapper.get_max_len()

		# Example input
		#ex_sent = ex_mask = torch.ones((2, pre_max_len)).long()
		#ex_idx = torch.randint(pre_max_len, (2,)).long()
		#ex_tagset_id = torch.tensor([0,1]).long()
		#self.example_input_array = ((ex_sent, ex_mask, ex_idx, ex_idx, ex_idx, ex_idx, ex_idx), ex_tagset_id)

		# Freeze or not all the weights, according to the specified parameters
		for param in self.pretrained_net.parameters():
			param.requires_grad = False #self.hparams.finetune_pretrained

		for param in list(self.pretrained_net.children())[-2].parameters():
			param.requires_grad = self.hparams.finetune_pretrained

		# The last encoder layer will be trained, unless the task are trained separently without a finetune
		for param in list(self.pretrained_net.children())[-1].parameters():
			param.requires_grad = self.hparams.finetune_pretrained or not self.hparams.separate_tasks

		# When training each task separately with finetune, use two pretrained networks
		if self.hparams.separate_tasks and self.hparams.finetune_pretrained:
			self.pretrained_net_args = self.pretrained_net
			self.pretrained_net_nouns = deepcopy(self.pretrained_net)

		# Arguments and Nouns networks
		self.args_net = ClassifierNetwork(pre_dim, self.n_args, self.hparams.dropout)
		self.nouns_net = ClassifierNetwork(pre_dim, self.n_nouns, 0.9)#self.hparams.dropout)

	def forward(self, x, tagset_id):
		token_ids, token_mask, start_arg_idx, end_arg_idx, start_pred_idx, end_pred_idx, suitable_verb_idx = x
		batch_size = token_ids.shape[0]
		batch_range = torch.arange(batch_size)

		# Get the indexes of each tagset whithin batch
		args_idxs, nouns_idxs = self.split_idxs_to_tagsets(tagset_id)
		assert (len(args_idxs)==len(nouns_idxs) and batch_size%2==0) or \
			   (abs(len(args_idxs)-len(nouns_idxs))==1 and batch_size%2==1) or not self.training

		# truncate to longest sequence length in batch (usually much smaller than 512) to save GPU RAM
		max_length = token_mask.max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
		if max_length < token_ids.shape[1]:
			token_ids = token_ids[:, :max_length]
			token_mask = token_mask[:, :max_length]

		# for i in range(batch_size):
		# 	print(self.pretrained_wrapper.tokenizer.decode(list(token_ids[i].cpu())[:max_length]))

		enc_out = self.pretrained_net(token_ids, token_mask)[0]
		start_pred_emb = enc_out[batch_range, start_pred_idx]
		end_pred_emb = enc_out[batch_range, end_pred_idx]
		start_arg_emb = enc_out[batch_range, start_arg_idx]
		end_arg_emb = enc_out[batch_range, end_arg_idx]
		#verb_emb = enc_out[args_range, suitable_verb_idx]

		net_in = (start_arg_emb + end_arg_emb + start_pred_emb + end_pred_emb) / 4
		net_out = self.args_net(net_in)

		# if self.pretrained_encoder_args is not self.pretrained_encoder_nouns:
		# if self.hparams.separate_tasks and self.hparams.finetune_pretrained:
		# 	enc_out_args = self.pretrained_net_args(token_ids[args_idxs], token_mask[args_idxs])[0]
		# 	enc_out_nouns = self.pretrained_net_nouns(token_ids[nouns_idxs], token_mask[nouns_idxs])[0]
		# else:
		# 	enc_out = self.pretrained_net(token_ids, token_mask)[0]
		# 	enc_out_args = enc_out[args_idxs]
		# 	enc_out_nouns = enc_out[nouns_idxs]
		#
		# assert len(enc_out_args) >= 1 and len(enc_out_nouns) >= 1, (len(enc_out_args), len(enc_out_nouns))
		#
		# # Relevant vectors in the encoder output
		# args_range = torch.arange(len(args_idxs))
		# start_pred_emb = enc_out_args[args_range, start_pred_idx[args_idxs]]
		# end_pred_emb = enc_out_args[args_range, end_pred_idx[args_idxs]]
		# start_arg_emb = enc_out_args[args_range, start_arg_idx[args_idxs]]
		# end_arg_emb = enc_out_args[args_range, end_arg_idx[args_idxs]]
		# verb_emb = enc_out_args[args_range, suitable_verb_idx[args_idxs]]
		#
		# # Arguments tagset network
		# #args_in = torch.cat([start_arg_emb, end_arg_emb, verb_emb, start_pred_emb, end_pred_emb], dim=1)
		# args_in = (enc_out_args[args_range, 0] + start_arg_emb + end_arg_emb + verb_emb + start_pred_emb + end_pred_emb) / 6
		# args_out = self.args_net(args_in)
		#
		# # Nouns tagset network
		# nouns_range = torch.arange(len(nouns_idxs))
		# start_pred_emb = enc_out_nouns[nouns_range, start_pred_idx[nouns_idxs]]
		# end_pred_emb = enc_out_nouns[nouns_range, end_pred_idx[nouns_idxs]]
		# #nouns_in = torch.cat([start_pred_emb, end_pred_emb], dim=1)
		# nouns_in = (enc_out_nouns[nouns_range, 0] + start_pred_emb + end_pred_emb) / 3
		# nouns_out = self.nouns_net(nouns_in)

		# Pad output for the entire batch
		# output = torch.zeros((batch_size, self.max_size), device=batch_device)

		# args_pad = -np.inf*torch.ones((len(args_indexes), self.max_size - self.n_args), device=batch_device)
		# output[args_indexes] = torch.cat([args_out, args_pad], dim=1)
		#
		# nouns_pad = -np.inf*torch.ones((len(nouns_indexes), self.max_size - self.n_nouns), device=batch_device)
		# output[nouns_indexes] = torch.cat([nouns_out, nouns_pad], dim=1)

		#return args_out, nouns_out

		return net_out[args_idxs], net_out[nouns_idxs]

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)

		# Hyper Parameters
		parser.add_argument('--lr', type=float, default=2e-5)
		parser.add_argument('--weight_decay', type=float, default=2e-3)
		parser.add_argument('--dropout', type=float, default=0.8)
		parser.add_argument('--args_loss_weight', type=float, default=0.5) # How much to prefer ARGS over NOUNS
		parser.add_argument('--finetune_pretrained', action='store_true')
		parser.add_argument('--separate_tasks', action='store_true')
		parser.add_argument('--context', action='store_true') # Using sentence context

		# Version Info
		parser.add_argument('--data_version', type=str, default='base')
		parser.add_argument('--experiment_version', type=str, default='debug')
		parser.add_argument('--tagset', type=str, default='syntactic')
		return parser



	@staticmethod
	def split_idxs_to_tagsets(tagset_id):
		args_indexes = (tagset_id == 0).nonzero(as_tuple=False).view(-1)  # ARGS = 0
		nouns_indexes = tagset_id.nonzero(as_tuple=False).view(-1)  # NOUNS = 1
		return args_indexes, nouns_indexes

	def _get_path(self, file_name, suffix, specific_dataset=True):
		path = config.DATASETS_PATH
		path = join(path, self.hparams.data_version)

		if not specific_dataset:
			path = os.path.dirname(path)

		return join(path, f"{file_name}.{suffix}")

	def encode(self, tokens, arg_start_idx, arg_end_idx, predicate_idx, verb, tagset_type):
		return self.pretrained_wrapper.encode(tokens, arg_start_idx, arg_end_idx,
											  predicate_idx, verb, tagset_type)

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

			encoded_key = (sentence, predicate_idx, is_verb, verb,
						   arg_head_idx, arg_start_idx, arg_end_idx,
						   label, tagset_type)

			# Generate the features for the current example
			features = encoded_examples.get(encoded_key, None)
			if not features:
				tokens = sentence.split()
				features = self.encode(tokens, arg_start_idx, arg_end_idx,
									   predicate_idx, verb, tagset_type)

				if not features:
					del_count += 1
					continue

			# Exapnd the features to have an extra 0-dim, if they aren't already
			if len(features[-1].shape) == 0:
				features = [t.unsqueeze(0) for t in features]
				encoded_examples[encoded_key] = features

			print(features[1])
			print(features[1].max(0))
			print(features[1].max(0)[0])
			print(features[1].max(0)[0].nonzero(as_tuple=False))
			exit()
			tokens_length = features[1].max(0)[0].nonzero(as_tuple=False)[-1].item() + 1
			max_len = max(max_len, tokens_length)

			# label_id = self.args_tagset[label] if tagset_type == ARGS_TAG else self.nouns_tagset[label]
			label_id = self.args_tagset[label]
			dataset_tuples.append((*features, torch.tensor([label_id])))

		print(f"DEL={del_count}, MAX={max_len}")
		return DoubleTaskDataset(*(map(torch.cat, zip(*dataset_tuples))), split_idxs_to_tagsets=self.split_idxs_to_tagsets)

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

	def _get_sampler(self, dataset):
		sampler = DoubleTaskSampler(dataset)
		if self.trainer.distributed_backend == "ddp":
			sampler = DoubleTaskDistributedSampler(sampler, self.trainer.num_gpus, self.trainer.local_rank)

		return sampler


	def train_dataloader(self):
		train_sampler = self._get_sampler(self.train_dataset)
		return DataLoader(self.train_dataset, batch_size=64, num_workers=32, sampler=train_sampler)

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

	def _weight_loss(self, args_loss, nouns_loss):
		return self.hparams.args_loss_weight * args_loss + \
			   (1 - self.hparams.args_loss_weight) * nouns_loss

	def _calculate_metrics(self, y, y_hat, tagset_id, step_type):
		args_indexes, nouns_indexes = self.split_idxs_to_tagsets(tagset_id)
		args_out, nouns_out = y_hat

		args_y = y[args_indexes]
		args_report = self._get_classification_report(args_y, args_out, self.args_tagset)

		nouns_y = y[nouns_indexes]
		nouns_report = self._get_classification_report(nouns_y, nouns_out, self.nouns_tagset)

		metrics = {f"{NOUNS_TAG}/f1": nouns_report["weighted avg"]["f1-score"],
				   f"{ARGS_TAG}/f1": args_report["weighted avg"]["f1-score"]}

		for label in set(self.args_labels + self.nouns_labels):
			if label in args_report:
				metrics[f"{ARGS_TAG}_{label}/f1"] = args_report[label]["f1-score"]

			if label in nouns_report:
				metrics[f"{NOUNS_TAG}_{label}/f1"] = nouns_report[label]["f1-score"]

		metrics = {k: torch.tensor([v]).to(y.device) for k, v in metrics.items()}
		args_loss = F.cross_entropy(args_out, y[args_indexes])
		nouns_loss = F.cross_entropy(nouns_out, y[nouns_indexes])
		loss = self._weight_loss(args_loss, nouns_loss)
		metrics["loss"] = loss.unsqueeze(0)
		metrics[f"{ARGS_TAG}-loss"] = args_loss.unsqueeze(0)
		metrics[f"{NOUNS_TAG}-loss"] = nouns_loss.unsqueeze(0)

		for_logs = dict([(k + "/" + step_type, v) for k, v in metrics.items()])
		metrics["log"] = for_logs

		# Save also the labels and predictions
		metrics["y"] = y
		metrics["y_args_hat"] = args_out
		metrics["y_nouns_hat"] = nouns_out
		metrics["tagset"] = tagset_id

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
		*x, tagset_id, y = batch
		y_hat = self.forward(x, tagset_id)
		metrics = self._calculate_metrics(y, y_hat, tagset_id, "train")
		return metrics

	def validation_step(self, batch, batch_idx, dataloader_idx):
		*x, tagset_id, y = batch
		y_hat = self.forward(x, tagset_id)
		step_type = "sanity" if dataloader_idx == 0 else "val"
		metrics = self._calculate_metrics(y, y_hat, tagset_id, step_type)
		return metrics

	def validation_epoch_end(self, outputs):
		total_epoch_metrics = {"log": {}}

		for dataloader_idx in range(len(outputs)):
			epoch_type = "sanity" if dataloader_idx == 0 else "val"

			y = []
			y_args_hat = []
			y_nouns_hat = []
			tagset_id = []
			for output in outputs[dataloader_idx]:
				y.append(output["y"])
				y_args_hat.append(output["y_args_hat"])
				y_nouns_hat.append(output["y_nouns_hat"])
				tagset_id.append(output["tagset"])

			# Calcuate over everything (more accurate)
			metrics = self._calculate_metrics(torch.cat(y), (torch.cat(y_args_hat), torch.cat(y_nouns_hat)), torch.cat(tagset_id), epoch_type)
			total_epoch_metrics["log"].update(metrics["log"])

			# Calculate by mean
			# mean_metrics = self._mean_metrics(outputs[dataloader_idx], epoch_type)
			# total_epoch_metrics["log"].update(mean_metrics["log"])

		return total_epoch_metrics

	def test_step(self, batch, batch_idx):
		*x, tagset_id, y = batch
		y_hat = self.forward(x, tagset_id)
		metrics = self._calculate_metrics(y, y_hat, tagset_id, "test")
		return metrics

	def test_epoch_end(self, outputs):
		mean_metrics = self._mean_metrics(outputs, "test")
		total_mean_metrics ={"log": mean_metrics["log"]}
		return total_mean_metrics