import abc
import os
import shutil
from os.path import join
from typing import Optional, Union, List

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from yet_another_verb.file_handlers.csv_file_handler import CSVFileHandler
from yet_another_verb.file_handlers.file_extensions import TORCH_EXTENSION, CSV_EXTENSION
from yet_another_verb.file_handlers.tensor_dataset_file_handler import TensorDatasetFileHandler
from yet_another_verb.ml.data_files import TRAIN, VAL, CONTROL_VAL, TEST, DATASET
from yet_another_verb.ml.data_modules.defaults import N_DATA_LOADING_WORKERS
from yet_another_verb.ml.utils import labels_to_tagset


class PretrainedDataModule(LightningDataModule, abc.ABC):
	train_dataset: TensorDataset
	control_val_dataset: TensorDataset  # validation from train
	val_dataset: TensorDataset
	testset_dataset: TensorDataset

	def __init__(
			self, pretrained_model: str, labels: List[str],
			data_dir: str, train_dir: str, val_dir: str, cache_dir: str,
			batch_size: int, n_loading_workers: Optional[int] = N_DATA_LOADING_WORKERS,
			val_size: Optional[Union[int, float]] = None, use_cache=True
	):
		super().__init__()
		self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, do_lower_case="uncased" in pretrained_model)
		self.tagset = labels_to_tagset(labels)
		self.save_hyperparameters()

	def _get_dataset_path(self, dataset_name: str):
		return join(self.hparams.data_dir, dataset_name, f"{DATASET}.{CSV_EXTENSION}")

	def _get_cache_path(self, dataset_name: str, file_extension: str):
		return join(self.hparams.data_dir, 'train-datasets', self.hparams.cache_dir, f"{dataset_name}.{file_extension}")

	def _split_raw_dataset(self, raw_dataset_name, train_filename, validation_filename, val_size: Union[int, float]):
		train_path = self._get_cache_path(train_filename, CSV_EXTENSION)
		validation_path = self._get_cache_path(validation_filename, CSV_EXTENSION)

		if not os.path.exists(train_path) or not os.path.exists(validation_path):
			df = CSVFileHandler.load(self._get_dataset_path(raw_dataset_name))
			train_df, validation_df = train_test_split(df, test_size=val_size)
			CSVFileHandler.save(train_path, train_df)
			CSVFileHandler.save(validation_path, validation_df)

	def _load_prepared_dataset(self, dataset_name: str) -> Optional[TensorDataset]:
		tensor_dataset_path = self._get_cache_path(dataset_name, TORCH_EXTENSION)
		if os.path.exists(tensor_dataset_path):
			return TensorDatasetFileHandler.load(tensor_dataset_path)

	def _save_prepared_dataset(self, dataset_name: str, dataset: TensorDataset):
		tensor_dataset_path = self._get_cache_path(dataset_name, TORCH_EXTENSION)
		TensorDatasetFileHandler.save(tensor_dataset_path, dataset)

	@abc.abstractmethod
	def _encode_data(self, df: pd.DataFrame) -> list:
		raise NotImplementedError()

	def _copy_dataset(self, dataset_name: str, dataset_type: str):
		dataset_path = self._get_dataset_path(dataset_name)
		cache_pdataset_path = self._get_cache_path(dataset_type, CSV_EXTENSION)
		shutil.copy(dataset_path, cache_pdataset_path)

	def _setup_dataset(self, dataset_name: str) -> TensorDataset:
		prepared_dataset = self._load_prepared_dataset(dataset_name)
		if prepared_dataset is not None and self.hparams.use_cache:
			return prepared_dataset

		df = CSVFileHandler.load(self._get_cache_path(dataset_name, CSV_EXTENSION))
		encoded_dataset = self._encode_data(df)
		encoded_dataset = [[torch.tensor([f]) for f in features] for features in zip(*encoded_dataset)]
		encoded_dataset = TensorDataset(*(map(torch.cat, zip(*encoded_dataset))))
		self._save_prepared_dataset(dataset_name, encoded_dataset)
		return encoded_dataset

	def setup(self, stage: Optional[str] = None):
		val_size = self.hparams.val_size
		if val_size is None:
			val_size = len(CSVFileHandler.load(self._get_dataset_path(self.hparams.val_dir)))

		# store datasets in cache
		self._split_raw_dataset(self.hparams.train_dir, TRAIN, CONTROL_VAL, val_size)
		self._copy_dataset(self.hparams.val_dir, VAL)

		# encode datasets
		self.val_dataset = self._setup_dataset(VAL)
		self.train_dataset = self._setup_dataset(TRAIN)
		self.control_val_dataset = self._setup_dataset(CONTROL_VAL)
		# self.testset_dataset = self._prepare_dataset(TEST)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_loading_workers)

	def val_dataloader(self):
		return [
			DataLoader(self.control_val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_loading_workers),
			DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_loading_workers)
		]

	# def test_dataloader(self):
	# 	return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.n_loading_workers)
