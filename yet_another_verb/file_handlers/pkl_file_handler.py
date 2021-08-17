import pickle
from os import makedirs
from os.path import dirname

from yet_another_verb.file_handlers.file_handler import FileHandler


class PKLFildHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> object:
		return pickle.load(open(file_path, "rb"))

	@staticmethod
	def save(file_path: str, data: object):
		makedirs(dirname(file_path), exist_ok=True)
		with open(file_path, "wb") as target_file:
			pickle.dump(data, target_file)
