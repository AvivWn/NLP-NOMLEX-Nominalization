from typing import List

from yet_another_verb.data_handling import TXTFileHandler
from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler


class LabelsFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> List[str]:
		labels = TXTFileHandler().load(file_path)
		return [label.strip() for label in labels]

	@staticmethod
	def save(file_path: str, labels: List[str]):
		FileHandler._make_relevant_dirs(file_path)
		labels = [label + "\n" for label in labels]
		TXTFileHandler().save(file_path, labels)
