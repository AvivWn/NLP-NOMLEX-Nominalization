import os

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.file_handlers import BinaryFileHandler


class ParsedBinFileHandler(BinaryFileHandler):
	def __init__(self, parser: DependencyParser):
		super().__init__()
		self.parser = parser

	def extend_file_name(self, file_path: str) -> str:
		return f"{file_path}-{self.parser.id}"

	def load(self, file_path: str) -> ParsedBin:
		file_path = self.extend_file_name(file_path)
		parsed_bin = self.parser.generate_parsed_bin()

		if os.path.exists(file_path):
			data = super().load(file_path)
			parsed_bin.from_bytes(data)

		return parsed_bin

	def save(self, file_path: str, parsed_bin: ParsedBin):
		file_path = self.extend_file_name(file_path)
		data = parsed_bin.to_bytes()
		super().save(file_path, data)
