import abc
from os.path import isfile
from typing import Sized, Union


class DatasetCreator(abc.ABC):
	dataset_size: int

	def __init__(self, dataset_size=None):
		self.dataset_size = dataset_size  # None means without size limit

	def has_reached_size(self, dataset_so_far: Union[Sized, int]):
		if self.dataset_size is None:
			return False

		if isinstance(dataset_so_far, int):
			size_so_far = dataset_so_far
		else:
			size_so_far = len(dataset_so_far)

		return size_so_far >= self.dataset_size

	def is_dataset_exist(self, out_dataset_path) -> bool:
		return isfile(out_dataset_path)

	def append_dataset(self, out_dataset_path):
		raise NotImplementedError()

	@abc.abstractmethod
	def create_dataset(self, out_dataset_path):
		...
