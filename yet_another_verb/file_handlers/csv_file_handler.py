from os import makedirs

import pandas as pd

from yet_another_verb.file_handlers.file_handler import FileHandler


class CSVFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path):
		return pd.read_csv(file_path, sep="\t", header=None, keep_default_na=False)

	@staticmethod
	def save(file_path, data):
		makedirs(file_path, exist_ok=True)
		df = pd.DataFrame(data)
		df.to_csv(file_path, sep="\t", index=False, header=False)
