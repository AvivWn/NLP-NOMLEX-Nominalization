import pickle

from yet_another_verb.file_handlers.file_handler import FileHandler


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
