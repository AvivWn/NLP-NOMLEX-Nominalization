import abc
import os
from os.path import join
from typing import Dict, Optional

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from toolz.dicttoolz import itemmap
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from yet_another_verb.file_handlers import TXTFileHandler
from yet_another_verb.file_handlers.csv_file_handler import CSVFileHandler
from yet_another_verb.file_handlers.file_extensions import TORCH_EXTENSION, CSV_EXTENSION
from yet_another_verb.file_handlers.tensor_dataset_file_handler import TensorDatasetFileHandler

TRAIN, VALIDATION, TEST = "train", "validaiton", "test"
DATASET = "dataset"
LABELS_FILE = "labels.txt"


class PretrainedDataModule(LightningDataModule, abc.ABC):
	train_dataset: TensorDataset
	val_dataset: TensorDataset
	testset_dataset: TensorDataset

	def __init__(self, pretrained_model: str, data_dir: str, batch_size: int, n_loading_workers: int):
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case="uncased" in pretrained_model)
		self.save_hyperparameters()

		# self.tagset = self._obtain_tagset()

	def _get_dataset_path(self, file_name: str, extension: str):
		return join(self.hparams.data_dir, f"{file_name}.{extension}")

	@abc.abstractmethod
	def _obtain_unique_labels(self, df: pd.DataFrame) -> list:
		raise NotImplementedError()

	def _obtain_tagset(self) -> Dict[str, int]:
		labels_path = join(self.hparams.data_dir, LABELS_FILE)
		if os.path.exists(labels_path):
			labels = TXTFileHandler().load(labels_path)
			labels = [label.strip() for label in labels]
		else:
			df = CSVFileHandler.load(self._get_dataset_path(DATASET, CSV_EXTENSION))
			labels = self._obtain_unique_labels(df)
			TXTFileHandler().save(labels_path, [label + "\n" for label in labels])

		return itemmap(reversed, dict(enumerate(sorted(labels))))

	def _split_raw_dataset(self):
		train_path = self._get_dataset_path(TRAIN, CSV_EXTENSION)
		validation_path = self._get_dataset_path(VALIDATION, CSV_EXTENSION)

		if not os.path.exists(train_path) or not os.path.exists(validation_path):
			df = CSVFileHandler.load(self._get_dataset_path(DATASET, CSV_EXTENSION))
			train_df, validation_df = train_test_split(df, test_size=0.2)
			CSVFileHandler.save(train_path, train_df)
			CSVFileHandler.save(validation_path, validation_df)

	def _load_prepared_dataset(self, file_name: str) -> Optional[TensorDataset]:
		tensor_dataset_path = self._get_dataset_path(file_name, TORCH_EXTENSION)
		if os.path.exists(tensor_dataset_path):
			return TensorDatasetFileHandler.load(tensor_dataset_path)

	def _save_prepared_dataset(self, file_name: str, dataset: TensorDataset):
		TensorDatasetFileHandler.save(self._get_dataset_path(file_name, TORCH_EXTENSION), dataset)

	@abc.abstractmethod
	def _encode_data(self, df: pd.DataFrame) -> list:
		raise NotImplementedError()

	def _setup_dataset(self, file_name: str) -> TensorDataset:
		prepared_dataset = self._load_prepared_dataset(file_name)
		if prepared_dataset is not None:
			return prepared_dataset

		df = CSVFileHandler.load(self._get_dataset_path(file_name, CSV_EXTENSION))
		encoded_dataset = self._encode_data(df)
		encoded_dataset = [[torch.tensor([f]) for f in features] for features in zip(*encoded_dataset)]
		encoded_dataset = TensorDataset(*(map(torch.cat, zip(*encoded_dataset))))
		self._save_prepared_dataset(file_name, encoded_dataset)
		return encoded_dataset

	def setup(self, stage: Optional[str] = None):
		self._split_raw_dataset()
		self.val_dataset = self._setup_dataset(VALIDATION)
		self.train_dataset = self._setup_dataset(TRAIN)
		# self.testset_dataset = self._prepare_dataset(TEST)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_loading_workers)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_loading_workers)

	# def test_dataloader(self):
	# 	return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.n_loading_workers)
