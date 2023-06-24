from typing import List, Union

from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler


class TXTFileHandler(FileHandler):
	def __init__(self, as_lines=True, handle_multi_line=False):
		super().__init__()
		self.as_lines = as_lines
		self.handle_multi_line = handle_multi_line

	def load(self, file_path: str) -> Union[str, List[str]]:
		with open(file_path, "r") as file:
			if self.as_lines:
				lines = file.readlines()
				return [line.strip() for line in lines] if self.handle_multi_line else lines
			else:
				text = file.read()
				return text.strip() if self.handle_multi_line else text

	def save(self, file_path: str, data: List[str]):
		FileHandler._make_relevant_dirs(file_path)

		if self.handle_multi_line:
			data = [line.strip() + "\n" for line in data]

		with open(file_path, "w") as target_file:
			target_file.writelines(data)
