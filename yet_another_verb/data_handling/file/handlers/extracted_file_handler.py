from typing import Union

from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction, \
	MultiWordExtraction, MultiWordExtractions
from yet_another_verb.data_handling import BinaryFileHandler
from yet_another_verb.data_handling.bytes.extracted_bytes_handler import ExtractedBytesHandler
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser

ExtractionObject = Union[Extraction, Extractions, MultiWordExtraction, MultiWordExtractions]


class ExtractedFileHandler(BinaryFileHandler):
	"""
		Smae as ExtractedBytesHandler, but pickles into a file
	"""

	def __init__(self, dependency_parser: DependencyParser = None, keep_compressed: bool = False):
		super().__init__()
		self.extracted_bytes_handler = ExtractedBytesHandler(dependency_parser, keep_compressed)

	def load(self, file_path: str) -> ExtractionObject:
		bytes_data = super().load(file_path)
		return self.extracted_bytes_handler.loads(bytes_data)

	def save(self, file_path: str, extraction_obj: ExtractionObject):
		super().save(file_path, self.extracted_bytes_handler.saves(extraction_obj))
