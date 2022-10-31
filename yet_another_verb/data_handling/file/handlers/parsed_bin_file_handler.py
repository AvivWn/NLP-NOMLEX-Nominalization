import os

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.data_handling import BinaryFileHandler
from yet_another_verb.dependency_parsing.parser_names import extended_path_by_parser, get_parser_by_extension


class ParsedBinFileHandler(BinaryFileHandler):
	def __init__(self, parser: DependencyParser = None):
		super().__init__()
		self.parser = parser

	def extend_file_name(self, file_path: str) -> str:
		return extended_path_by_parser(file_path, self.parser)

	def load(self, file_path: str) -> ParsedBin:
		if self.parser is not None:
			parser = self.parser
			self.extend_file_name(file_path)
		else:
			parser = get_parser_by_extension(file_path)

		parsed_bin = parser.generate_parsed_bin()

		if os.path.exists(file_path):
			data = super().load(file_path)
			parsed_bin.from_bytes(data)

		return parsed_bin

	def save(self, file_path: str, parsed_bin: ParsedBin):
		assert self.parser is not None
		file_path = self.extend_file_name(file_path)
		data = parsed_bin.to_bytes()
		super().save(file_path, data)
