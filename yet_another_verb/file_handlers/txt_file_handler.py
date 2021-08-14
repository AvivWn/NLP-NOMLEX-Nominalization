from os import makedirs
from typing import List

from yet_another_verb.file_handlers.file_handler import FileHandler


class TXTFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> str:
		return open(file_path, "r").read()

	@staticmethod
	def save(file_path: str, data: List[str]):
		makedirs(file_path, exist_ok=True)

		with open(file_path, "w") as target_file:
			target_file.writelines(data)
