import os
import numpy as np
from collections import defaultdict
from random import shuffle

from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.metrics.classification import f1_score, accuracy, precision, recall
from sklearn.metrics import classification_report
from torch import optim

from arguments_extractor.rule_based.lexicon import Lexicon
from arguments_extractor import config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
pl.seed_everything(42)

class CheckpointEveryNSteps(pl.Callback):
	"""
	Save a checkpoint every N steps, instead of Lightning's default that checkpoints
	based on validation loss.
	"""

	def __init__(
		self,
		save_step_frequency,
		prefix="checkpoint",
		use_modelcheckpoint_filename=False,
	):
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
		""" Check if we should save a checkpoint after every train batch """
		epoch = trainer.current_epoch
		global_step = trainer.global_step
		if global_step % self.save_step_frequency == 0:
			if self.use_modelcheckpoint_filename:
				filename = trainer.checkpoint_callback.filename
			else:
				filename = f"{self.prefix}_epoch={epoch}_step={global_step}.ckpt"
			ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
			trainer.save_checkpoint(ckpt_path)

class TypesModel(pl.LightningModule):
	def __init__(self, args_dataset, nouns_dataset, pretrained_wrapper, lr, weight_decay):
		super(TypesModel, self).__init__()

		self.args_dataset = args_dataset
		self.n_args = len(self.args_dataset) # index 0

		self.nouns_dataset = nouns_dataset
		self.n_nouns = len(self.nouns_dataset) # index 1

		self.pretrained_wrapper = pretrained_wrapper
		self.pretrained_encoder = pretrained_wrapper.pretrained_model
		bert_dim = pretrained_wrapper.dim

		# Arguments network
		self.dropout = torch.nn.Dropout(p=0.8)
		self.fc1 = torch.nn.Linear(bert_dim * 4, int(bert_dim / 2))
		self.batch_norm1 = torch.nn.BatchNorm1d(int(bert_dim / 2))
		self.fc2 = torch.nn.Linear(int(bert_dim / 2), int(bert_dim / 8))
		self.batch_norm2 = torch.nn.BatchNorm1d(int(bert_dim / 8))
		self.fc3 = torch.nn.Linear(int(bert_dim / 8), self.n_args)

		# Nouns network
		self.fc4 = torch.nn.Linear(bert_dim, int(bert_dim / 2))
		self.batch_norm4 = torch.nn.BatchNorm1d(int(bert_dim / 2))
		self.fc5 = torch.nn.Linear(int(bert_dim / 2), int(bert_dim / 8))
		self.batch_norm5 = torch.nn.BatchNorm1d(int(bert_dim / 8))
		self.fc6 = torch.nn.Linear(int(bert_dim / 8), self.n_nouns)

		self.max_size = max(self.n_args, self.n_nouns)

		# Hyper parameters
		self.lr = lr
		self.weight_decay = weight_decay
		self.save_hyperparameters("lr", "weight_decay", "args_dataset", "nouns_dataset")

	def forward(self, x, dataset_id):
		token_ids, token_mask, start_argument_index, end_argument_index, predicate_index, suitable_verb_index = x
		batch_size = token_ids.shape[0]
		batch_device = token_ids.device

		# truncate to longest sequence length in batch (usually much smaller than 512) to save GPU RAM
		max_length = (token_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item()
		if max_length < token_ids.shape[1]:
			token_ids = token_ids[:, :max_length]
			token_mask = token_mask[:, :max_length]

		bert_last_layer = self.pretrained_encoder(token_ids, token_mask)[0]

		# Get the indexes of each dataset whithin batch
		batch_range = torch.arange(batch_size)
		args_idexes = (dataset_id == 0).nonzero(as_tuple=False).view(-1)	# ARGS = 0
		nouns_idexes = dataset_id.nonzero(as_tuple=False).view(-1) 			# NOUNS = 1

		# Relevant embedding in bert
		predicate_emb = bert_last_layer[batch_range, predicate_index]
		start_argument_emb = bert_last_layer[args_idexes, start_argument_index[args_idexes]]
		end_argument_emb = bert_last_layer[args_idexes, end_argument_index[args_idexes]]
		verb_emb = bert_last_layer[args_idexes, suitable_verb_index[args_idexes]]

		# Arguments dataset network
		args_in = torch.cat([start_argument_emb, end_argument_emb, verb_emb, predicate_emb[args_idexes]], dim=1)
		fc1_out = self.dropout(F.relu(self.batch_norm1(self.fc1(args_in))))
		fc2_out = self.dropout(F.relu(self.batch_norm2(self.fc2(fc1_out))))
		args_out = self.fc3(fc2_out)

		# Nouns dataset network
		nouns_in = predicate_emb[nouns_idexes]
		fc4_out = self.dropout(F.relu(self.batch_norm4(self.fc4(nouns_in))))
		fc5_out = self.dropout(F.relu(self.batch_norm5(self.fc5(fc4_out))))
		nouns_out = self.fc6(fc5_out)

		# Pad output for the entire batch
		output = torch.zeros((batch_size, self.max_size), device=batch_device)

		args_pad = -np.inf*torch.ones((len(args_idexes), self.max_size - self.n_args), device=batch_device)
		output[args_idexes] = torch.cat([args_out, args_pad], dim=1)

		nouns_pad = -np.inf*torch.ones((len(nouns_idexes), self.max_size - self.n_nouns), device=batch_device)
		output[nouns_idexes] = torch.cat([nouns_out, nouns_pad], dim=1)

		return output

	@staticmethod
	def get_dataset_path(dataset_type):
		return config.DATASETS_PATH + f"/dataset_{dataset_type}_full.pth"

	def split_df(self, df, ):
		pass

	@staticmethod
	def get_verb_type(verb_labels):
		if "IND-OBJ" in verb_labels:
			return "ditransitive"
		elif "OBJECT" in verb_labels:
			return "transitive"
		else:
			return "intransitive"

	# def _split_raw_data(self):
		# print(1)
		# nom_lexicon = Lexicon(config.LEXICON_FILE_NAME)
		#
		# args_df = pd.read_csv(config.ARG_DATASET_PATH, sep="\t", header=None, names=["sentence", "argument_start_index", "argument_end_index", "predicate_index", "suitable_verb", "label"])
		# args_df = args_df.loc[(args_df["argument_start_index"] != args_df["argument_end_index"]) | (args_df["predicate_index"] != args_df["argument_end_index"])]
		# args_df["dataset_name"] = "ARGS"
		#
		# print(2)
		#
		# nouns_df = pd.read_csv(config.NOUN_DATASET_PATH, sep="\t", header=None, names=["sentence", "predicate_index", "suitable_verb", "label"])
		# nouns_df["argument_start_index"] = -1
		# nouns_df["argument_end_index"] = -1
		# nouns_df["dataset_name"] = "NOUNS"
		#
		# print(3)
		#
		# df = pd.concat([args_df, nouns_df])
		# print(4)
		#
		# # Split the data by the types of possible nominalizations for each verb
		# nom_types_per_verb = nom_lexicon.get_types_per_verb()
		# df["nom_types"] = df.apply(lambda row: " ".join(list(nom_types_per_verb.get(row["suitable_verb"], ["NONE"]))), axis=1)
		# print(df.head(10))
		#
		# labels_for_verb = defaultdict(list)
		# for i, row in tqdm(df.iterrows()):
		# 	if row["argument_start_index"] != -1:
		# 		labels_for_verb[row["suitable_verb"]].append(row["label"])
		#
		# df["verb_type"] = df.apply(lambda row: self.get_verb_type(labels_for_verb[row["suitable_verb"]]), axis=1)
		# df.to_csv(config.DATASETS_PATH + "/total_dataset.csv", sep="\t", header=None, index=False)
		#
		# verb_counts = df.groupby(["suitable_verb"]).size().reset_index(name="count")
		# total_n_sents = len(df)
		#
		# verbs = list(set(df["suitable_verb"].tolist()))
		# print(len(verbs), verbs[:20])
		# shuffle(verbs)
		#
		# test_verbs = []
		# test_sents_count = 0
		# for verb in verbs:
		# 	test_verbs.append(verb)
		# 	test_sents_count += verb_counts[verb]
		#
		# 	if test_sents_count >= 0.2 * total_n_sents:
		# 		break
		#
		# print()
		#
		# # Aggregate together too rare groups
		# types_df = df.groupby(["types"]).size().reset_index(name="count")
		# not_common = types_df.loc[types_df["count"] == 1]["types"].tolist()
		# df.loc[df["types"].isin(not_common), "types"] = "UNK"
		# # a = df.groupby(["types"]).size().reset_index(name="count")
		# # print(type(a))
		# #
		# # for line in a:
		# # 	c = df.loc[df["types"] == line["types"]].groupby(["suitable_verb"]).size()
		# # 	print(c)
		# # 	# a[i] = df.loc["types" == a[i, "types"]].groupby(["verb"]).size()
		# #
		# # print(a)
		#
		# train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=SEED, stratify=df[["types", "suitable_verb"]])
		# train_df, val_df = train_test_split(df, test_size=0.25, shuffle=True, random_state=SEED, stratify=df[["types", "suitable_verb"]])
		#
		# return train_df, val_df, test_df


	def _pandas_to_pytorch(self, dataset: pd.DataFrame, dataset_type):
		dataset_tuples = []

		for row_info in tqdm(dataset.itertuples(), f"Processing {dataset_type} dataset", leave=True):
			(i, sentence, arg_start_idx, arg_end_idx,
			 predicate_index, suitable_verb, label,
			 tagset_type, nom_types, verb_type, types) = tuple(row_info)

			suitable_verb = suitable_verb.split("#")[0]
			tokens = sentence.split()
			features = self.pretrained_wrapper.encode(tokens, arg_start_idx, arg_end_idx,
													  predicate_index, suitable_verb, tagset_type)

			label_id = self.args_dataset[label] if tagset_type == "ARGS" else self.nouns_dataset[label]
			dataset_tuples.append((*features, torch.tensor([label_id])))

		return TensorDataset(*tuple(map(torch.cat, zip(*dataset_tuples))))

	def _process_dataset(self, dataset:pd.DataFrame, dataset_type):
		dataset_path = self.get_dataset_path(dataset_type)
		processed_dataset = self._pandas_to_pytorch(dataset, dataset_type)
		torch.save(processed_dataset, dataset_path)
		return processed_dataset

	def _load_dataset(self, dataset_type):
		dataset_path = self.get_dataset_path(dataset_type)
		if config.LOAD_DATASET and os.path.exists(dataset_path):
			return torch.load(dataset_path)

		return None

	def prepare_data(self):
		self.train_dataset = self._load_dataset("train")
		self.val_dataset = self._load_dataset("val")
		self.test_dataset = self._load_dataset("test")

		if not (self.train_dataset and self.val_dataset and self.test_dataset):
			train_df = pd.read_csv(config.DATASETS_PATH + "/dataset_train_full.csv", sep="\t", header=None, keep_default_na=False)
			val_df = pd.read_csv(config.DATASETS_PATH + "/dataset_val_full.csv", sep="\t", header=None, keep_default_na=False)
			test_df = pd.read_csv(config.DATASETS_PATH + "/dataset_test_full.csv", sep="\t", header=None, keep_default_na=False)
			# train_df, val_df, test_df = self._split_raw_data()
			self.train_dataset = self._process_dataset(train_df, "train")
			self.val_dataset = self._process_dataset(val_df, "val")
			self.test_dataset = self._process_dataset(test_df, "test")

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=128, num_workers=32, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=2048, num_workers=32)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=2048, num_workers=32)

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=2*(10**-5), weight_decay=2e-3)
		return optimizer



	@staticmethod
	def _calculate_metrics(y, y_hat, dataset_id, step_type):
		args_idexes = (dataset_id == 0).nonzero(as_tuple=False).view(-1)
		args_y, args_y_hat = y[args_idexes], torch.argmax(y_hat[args_idexes], dim=1)
		args_f1 = f1_score(args_y_hat, args_y).unsqueeze(0)

		nouns_idexes = dataset_id.nonzero(as_tuple=False).view(-1)
		nouns_y, nouns_y_hat = y[nouns_idexes], torch.argmax(y_hat[nouns_idexes], dim=1)
		nouns_f1 = f1_score(nouns_y_hat, nouns_y).unsqueeze(0)

		loss = F.cross_entropy(y_hat, y).unsqueeze(0)

		# @TODO: ignore f1 for batches with only one dataset

		metrics = {"loss": loss,
				   "nouns_f1": nouns_f1, "args_f1": args_f1}

		for_logs = dict([(step_type + "_" + k, v) for k, v in metrics.items()])
		metrics["log"] = for_logs
		# metrics["y"] = y
		# metrics["y_hat"] = y_hat
		return metrics

	@staticmethod
	def _mean_metrics(outputs, epoch_type):
		metrics_outputs = defaultdict(list)

		for output in outputs:
			for metric, value in output.items():
				if metric not in ["log", "nouns_y", "nouns_y_hat"]:
					metrics_outputs["mean_" + metric].append(value.mean())

		mean_metrics = dict([(metric, torch.tensor(values).mean()) for metric, values in metrics_outputs.items()])
		for_logs = dict([(epoch_type + "_" + k, v) for k, v in mean_metrics.items()])
		mean_metrics["log"] = for_logs

		return mean_metrics

	def on_fit_start(self):
		self.logger.log_hyperparams(params=self.hparams)

	def training_step(self, batch, batch_idx):
		*x, dataset_id, y = batch
		y_hat = self.forward(x, dataset_id)
		metrics = self._calculate_metrics(y, y_hat, dataset_id, "train")
		return metrics

	def validation_step(self, batch, batch_idx):
		*x, dataset_id, y = batch
		y_hat = self.forward(x, dataset_id)
		metrics = self._calculate_metrics(y, y_hat, dataset_id, "val")

		# DEBUGING nouns dataset
		nouns_idexes = dataset_id.nonzero(as_tuple=False).view(-1)
		metrics["nouns_y"] = y[nouns_idexes]
		metrics["nouns_y_hat"] = y_hat[nouns_idexes]
		return metrics

	def validation_epoch_end(self, outputs):
		y = torch.cat([output["nouns_y"] for output in outputs]).cpu().numpy()
		y_hat = np.argmax(torch.cat([output["nouns_y_hat"] for output in outputs]).cpu().numpy(), axis=1)
		print(classification_report(y, y_hat))

		return self._mean_metrics(outputs, "val")

	def test_step(self, batch, batch_idx):
		*x, dataset_id, y = batch
		y_hat = self.forward(x, dataset_id)
		metrics = self._calculate_metrics(y, y_hat, dataset_id, "test")
		return metrics

	def test_epoch_end(self, outputs):
		return self._mean_metrics(outputs, "test")