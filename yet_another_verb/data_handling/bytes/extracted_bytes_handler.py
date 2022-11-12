from itertools import chain
from typing import Union, Dict

import torch
from tqdm import tqdm

from yet_another_verb.arguments_extractor.extraction.words import Words
from yet_another_verb.data_handling import TorchBytesHandler
from yet_another_verb.data_handling.bytes.compressed.compressed_encoding import CompressedEncoding
from yet_another_verb.data_handling.bytes.pkl_bytes_handler import PKLBytesHandler
from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction, \
	MultiWordExtraction, MultiWordExtractions
from yet_another_verb.dependency_parsing import engine_by_parser
from yet_another_verb.data_handling.bytes.compressed.compressed_parsed_text import CompressedParsedText
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.sentence_encoding.encoding import Encoding

ExtractionObject = Union[Extraction, Extractions, MultiWordExtraction, MultiWordExtractions]


class ExtractedBytesHandler(PKLBytesHandler):
	"""
		Pickles a simple extracted object, but saves the parsed data inside efficiently
		This object is able to save the extractions, while compressing the parsed data
	"""

	def __init__(self, dependency_parser: DependencyParser = None, keep_compressed: bool = False):
		super().__init__()
		self.dependency_parser = dependency_parser
		self.keep_compressed = keep_compressed

	@staticmethod
	def _get_extractions(extraction_obj: ExtractionObject) -> Extractions:
		extractions = [extraction_obj]
		if isinstance(extraction_obj, list):
			extractions = []
			extractions += extraction_obj

		for ext in extractions:
			if isinstance(ext, MultiWordExtraction):
				extractions += chain(*ext.extractions_per_idx.values())

		return extractions

	def _get_compressed_words(self, words: Words) -> CompressedParsedText:
		if isinstance(words, ParsedText):
			assert self.dependency_parser is not None
			words = CompressedParsedText(
				bytes_data=words.to_bytes(),
				parsing_egnine=engine_by_parser[type(self.dependency_parser)],
				parser_name=self.dependency_parser.name)
		elif isinstance(words, CompressedParsedText):
			words = CompressedParsedText(
				bytes_data=words.bytes_data,
				parsing_egnine=words.parsing_egnine,
				parser_name=words.parser_name)
			words.parsed_text = None

		return words

	@staticmethod
	def _get_compressed_encoding(encoding: Encoding) -> CompressedEncoding:
		if isinstance(encoding, torch.Tensor):
			encoding = CompressedEncoding(bytes_data=TorchBytesHandler.saves(encoding))
		elif isinstance(encoding, CompressedEncoding):
			encoding = CompressedEncoding(bytes_data=encoding.bytes_data)
			encoding.encoding = None

		return encoding

	def _get_decompressed_words(self, words: Words, parser_by_id: Dict[str, DependencyParser]) -> ParsedText:
		if not isinstance(words, CompressedParsedText):
			return words

		parser_id = words.parser_name
		parser = parser_by_id.get(parser_id)
		if parser is None:
			parser = DependencyParserFactory(words.parsing_egnine, parser_name=words.parser_name)()
			parser_by_id[parser_id] = parser

		parsed_text = parser.from_bytes(words.bytes_data)
		words.parsed_text = parsed_text
		return words if self.keep_compressed else parsed_text

	def _get_decompressed_encoding(self, encoding: Encoding) -> torch.Tensor:
		if not isinstance(encoding, CompressedEncoding):
			return encoding

		decompressed_encoding = TorchBytesHandler.loads(encoding.bytes_data)
		encoding.encoding = decompressed_encoding

		return encoding if self.keep_compressed else decompressed_encoding

	def loads(self, bytes_data: bytes) -> ExtractionObject:
		extraction_obj = super().loads(bytes_data)
		extractions = ExtractedBytesHandler._get_extractions(extraction_obj)

		parser_by_id = {}
		for ext in tqdm(extractions):
			if isinstance(ext.words, CompressedParsedText):
				ext.words = self._get_decompressed_words(ext.words, parser_by_id)

			for arg in ext.all_args:
				arg.encoding = self._get_decompressed_encoding(arg.encoding)

		return extraction_obj

	def saves(self, extraction_obj: ExtractionObject) -> bytes:
		extractions = ExtractedBytesHandler._get_extractions(extraction_obj)

		original_words = []
		for ext in tqdm(extractions):
			original_words.append(ext.words)
			ext.words = self._get_compressed_words(ext.words)

			for arg in ext.all_args:
				arg.encoding = self._get_compressed_encoding(arg.encoding)

		bytes_data = super().saves(extraction_obj)

		for words, ext in zip(original_words, extractions):
			ext.words = words

		return bytes_data
