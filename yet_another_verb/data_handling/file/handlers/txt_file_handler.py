from typing import List, Union

from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler


class TXTFileHandler(FileHandler):
	def __init__(self, as_lines=True):
		super().__init__()
		self.as_lines = as_lines

	def load(self, file_path: str) -> Union[str, List[str]]:
		with open(file_path, "r") as file:
			if self.as_lines:
				return file.readlines()

			return file.read()

	def save(self, file_path: str, data: List[str]):
		FileHandler._make_relevant_dirs(file_path)
		with open(file_path, "w") as target_file:
			target_file.writelines(data)
