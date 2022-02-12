from yet_another_verb.file_handlers.file_handler import FileHandler


class BinaryFileHandler(FileHandler):
	def __init__(self):
		super().__init__()

	@staticmethod
	def load(file_path: str) -> bytes:
		with open(file_path, "rb") as input_file:
			return input_file.read()

	@staticmethod
	def save(file_path: str, data: bytes):
		FileHandler._make_relevant_dirs(file_path)
		with open(file_path, 'wb') as output_file:
			output_file.write(data)
