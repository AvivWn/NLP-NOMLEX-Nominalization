from itertools import chain
from typing import Union

from yet_another_verb.data_handling.bytes.pkl_bytes_handler import PKLBytesHandler
from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction, \
	MultiWordExtraction, MultiWordExtractions
from yet_another_verb.dependency_parsing import engine_by_parser
from yet_another_verb.dependency_parsing.compressed_parsed_text import CompressedParsedText
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory

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

	def loads(self, bytes_data: bytes) -> ExtractionObject:
		extraction_obj = super().loads(bytes_data)
		extractions = ExtractedBytesHandler._get_extractions(extraction_obj)

		parser_by_id = {}
		for ext in extractions:
			if isinstance(ext.words, CompressedParsedText):
				parser_id = ext.words.parser_name
				parser = parser_by_id.get(parser_id)

				if parser is None:
					parser = DependencyParserFactory(ext.words.parsing_egnine, parser_name=ext.words.parser_name)()
					parser_by_id[parser_id] = parser

				parsed_text = parser.from_bytes(ext.words.bytes_data)
				ext.words.parsed_text = parsed_text

				if not self.keep_compressed:
					ext.words = parsed_text

		return extraction_obj

	def saves(self, extraction_obj: ExtractionObject) -> bytes:
		extractions = ExtractedBytesHandler._get_extractions(extraction_obj)

		original_words = []
		for ext in extractions:
			original_words.append(ext.words)
			if isinstance(ext.words, ParsedText):
				ext.words = CompressedParsedText(
					bytes_data=ext.words.to_bytes(),
					parsing_egnine=engine_by_parser[type(self.dependency_parser)],
					parser_name=self.dependency_parser.name)
			elif isinstance(ext.words, CompressedParsedText):
				ext.words.parsed_text = None

		bytes_data = super().saves(extraction_obj)

		for words, ext in zip(original_words, extractions):
			ext.words = words

		return bytes_data
