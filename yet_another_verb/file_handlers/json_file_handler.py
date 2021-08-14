import json
from os import makedirs

from yet_another_verb.file_handlers.file_handler import FileHandler


class JsonFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> dict:
		with open(file_path, "r") as json_file:
			return json.load(json_file)

	@staticmethod
	def save(file_path: str, data: dict):
		makedirs(file_path, exist_ok=True)
		with open(file_path, "w") as json_file:
			json.dump(data, json_file)
