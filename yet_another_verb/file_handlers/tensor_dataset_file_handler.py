import torch
from torch.utils.data import TensorDataset

from yet_another_verb.file_handlers.file_handler import FileHandler


class TensorDatasetFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> TensorDataset:
		return torch.load(file_path)

	@staticmethod
	def save(file_path: str, data: TensorDataset):
		torch.save(data, file_path)
