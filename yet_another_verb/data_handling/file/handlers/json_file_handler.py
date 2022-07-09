import json

from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler


class JsonFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> dict:
		with open(file_path, "r") as json_file:
			return json.load(json_file)

	@staticmethod
	def save(file_path: str, data: dict):
		FileHandler._make_relevant_dirs(file_path)
		with open(file_path, "w") as json_file:
			json.dump(data, json_file)
