import abc
from os.path import isfile
from typing import Sized


class DatasetCreator(abc.ABC):
	dataset_size: int

	def __init__(self, dataset_size=None):
		self.dataset_size = dataset_size  # None means without size limit

	def has_reached_size(self, dataset_so_far: Sized):
		if self.dataset_size is None:
			return False

		return len(dataset_so_far) >= self.dataset_size

	def is_dataset_exist(self, out_dataset_path) -> bool:
		return isfile(out_dataset_path)

	@abc.abstractmethod
	def create_dataset(self, out_dataset_path):
		...
