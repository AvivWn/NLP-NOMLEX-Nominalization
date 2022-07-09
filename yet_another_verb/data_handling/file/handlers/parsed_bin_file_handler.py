import os

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.data_handling import BinaryFileHandler
from yet_another_verb.dependency_parsing.parser_names import engine_by_parser


class ParsedBinFileHandler(BinaryFileHandler):
	def __init__(self, parser: DependencyParser, extend_with_parser_id=True):
		super().__init__()
		self.parser = parser
		self.extend_with_parser_id = extend_with_parser_id

	def extend_file_name(self, file_path: str) -> str:
		return f"{file_path}-{engine_by_parser[type(self.parser)]}-{self.parser.name}"

	def load(self, file_path: str) -> ParsedBin:
		if self.extend_with_parser_id:
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
