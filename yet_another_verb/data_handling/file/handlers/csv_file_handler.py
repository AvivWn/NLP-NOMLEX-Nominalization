import pandas as pd

from yet_another_verb.data_handling.file.handlers.file_handler import FileHandler


class CSVFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> pd.DataFrame:
		return pd.read_csv(file_path, sep="\t", keep_default_na=False)

	@staticmethod
	def save(file_path: str, data: pd.DataFrame):
		FileHandler._make_relevant_dirs(file_path)
		data.to_csv(file_path, sep="\t", index=False)
