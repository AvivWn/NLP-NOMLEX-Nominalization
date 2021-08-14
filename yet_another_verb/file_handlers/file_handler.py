import abc


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
