import torch
from torch.utils.data import TensorDataset

from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler


class TensorDatasetFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> TensorDataset:
		return torch.load(file_path)

	@staticmethod
	def save(file_path: str, data: TensorDataset):
		FileHandler._make_relevant_dirs(file_path)
		torch.save(data, file_path)
