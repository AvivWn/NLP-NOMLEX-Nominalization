import abc
from os.path import dirname
from os import makedirs


class FileHandler(abc.ABC):
	def __init__(self):
		pass

	@staticmethod
	@abc.abstractmethod
	def load(file_path):
		pass

	@staticmethod
	@abc.abstractmethod
	def save(file_path, data):
		pass

	@staticmethod
	def _make_relevant_dirs(file_path):
		dir_name = dirname(file_path)

		if dir_name != '':
			makedirs(dir_name, exist_ok=True)
