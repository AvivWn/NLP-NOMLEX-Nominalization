import pickle

from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler
from yet_another_verb._bw_alias import *


class PKLFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> object:
		return pickle.load(open(file_path, "rb"))

	@staticmethod
	def save(file_path: str, data: object):
		FileHandler._make_relevant_dirs(file_path)
		with open(file_path, "wb") as target_file:
			pickle.dump(data, target_file)
