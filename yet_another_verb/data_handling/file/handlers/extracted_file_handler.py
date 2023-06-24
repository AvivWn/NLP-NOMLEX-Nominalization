from os.path import join, splitext, exists
from typing import Union

import torch
from torch.utils.data import TensorDataset

from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction, \
	MultiWordExtraction, MultiWordExtractions
from yet_another_verb.data_handling import BinaryFileHandler, ParsedBinFileHandler, TensorDatasetFileHandler
from yet_another_verb.data_handling.bytes.extracted_bytes_handler import ExtractedBytesHandler
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser

ExtractionObject = Union[Extraction, Extractions, MultiWordExtraction, MultiWordExtractions]

EXTRACTED_FILE = "extractions.extracted"
PARSED_FILE = "parsings.parsed"
ENCODED_FILE = "encodings.pt"


class ExtractedFileHandler(BinaryFileHandler):
	"""
		Smae as ExtractedBytesHandler, but pickles into a file
	"""

	def __init__(
			self, dependency_parser: DependencyParser = None,
			keep_compressed: bool = False,  # keep parsings and encodings compressed after load
			compress_internally: bool = False,  # save binary of parsings and encodings in the extraction binary
			compress_parsing: bool = True,
			compress_encoding: bool = True
	):
		super().__init__()
		self.parsed_bin_file_handler = ParsedBinFileHandler(dependency_parser)
		self.dependency_parser = dependency_parser
		self.extracted_bytes_handler = ExtractedBytesHandler(
			self.dependency_parser, keep_compressed=keep_compressed, compress_internally=compress_internally,
			compress_parsing=compress_parsing, compress_encoding=compress_encoding
		)
		self.compress_internally = compress_internally
		self.compress_encoding = compress_encoding
		self.compress_parsing = compress_parsing

	def _get_extracted_path(self, file_path: str) -> str:
		return join(splitext(file_path)[0], EXTRACTED_FILE) if not self.compress_internally else file_path

	@staticmethod
	def _get_parsed_path(file_path: str) -> str:
		return join(splitext(file_path)[0], PARSED_FILE)

	@staticmethod
	def _get_encoded_path(file_path: str) -> str:
		return join(splitext(file_path)[0], ENCODED_FILE)

	def load(self, file_path: str) -> ExtractionObject:
		if not self.compress_internally:
			if self.compress_parsing:
				parsed_bin = self.parsed_bin_file_handler.load(self._get_parsed_path(file_path))
				parsings = list(parsed_bin.get_parsed_texts())
				self.extracted_bytes_handler.set_indexed_parsings(parsings)

			if self.compress_encoding:
				encodings = []
				encoded_path = self._get_encoded_path(file_path)
				if exists(encoded_path):
					encodings_dataset = TensorDatasetFileHandler.load(encoded_path)
					encodings = list(torch.unbind(encodings_dataset.tensors[0]))

				self.extracted_bytes_handler.set_indexed_encodings(encodings)

		bytes_extraction = super().load(self._get_extracted_path(file_path))
		return self.extracted_bytes_handler.loads(bytes_extraction)

	def _save_parsings(self, file_path: str):
		if self.compress_parsing:
			parsings = self.extracted_bytes_handler.get_indexed_parsings()
			parsed_bin = self.dependency_parser.generate_parsed_bin()
			parsed_bin.add_multiple(parsings)
			self.parsed_bin_file_handler.save(self._get_parsed_path(file_path), parsed_bin)

	def _save_encodings(self, file_path: str):
		if self.compress_encoding:
			encodings = self.extracted_bytes_handler.get_indexed_encodings()

			if len(encodings) > 0:
				encodings_dataset = TensorDataset(torch.stack(encodings))
				TensorDatasetFileHandler.save(self._get_encoded_path(file_path), encodings_dataset)

	def save(self, file_path: str, extraction_obj: ExtractionObject):
		self.extracted_bytes_handler.reset_indices()
		bytes_extraction = self.extracted_bytes_handler.saves(extraction_obj)
		super().save(self._get_extracted_path(file_path), bytes_extraction)

		if not self.compress_internally:
			self._save_parsings(file_path)
			self._save_encodings(file_path)
