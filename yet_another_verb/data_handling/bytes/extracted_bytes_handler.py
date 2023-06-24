from itertools import chain
from typing import Union, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from yet_another_verb.arguments_extractor.extraction.words import Words, TokenizedText
from yet_another_verb.data_handling import TorchBytesHandler
from yet_another_verb.data_handling.bytes.compressed.compressed_encoding import CompressedEncoding
from yet_another_verb.data_handling.bytes.pkl_bytes_handler import PKLBytesHandler
from yet_another_verb.arguments_extractor.extraction import Extractions, Extraction, \
	MultiWordExtraction, MultiWordExtractions
from yet_another_verb.data_handling.bytes.compressed.compressed_parsed_text import CompressedParsedText
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.sentence_encoding.encoding import Encoding
from yet_another_verb.configuration.parsing_config import ENGINE_BY_PARSER

ExtractionObject = Union[Extraction, Extractions, MultiWordExtraction, MultiWordExtractions]
StoredWords = Union[int, Words]
StoredEncoding = Union[int, Encoding]


class ExtractedBytesHandler(PKLBytesHandler):
	"""
		Pickles a simple extracted object, but saves the parsed data inside efficiently
		This object is able to save the extractions, while compressing the parsed data
	"""

	def __init__(
			self, dependency_parser: DependencyParser = None,
			keep_compressed: bool = False,  # keep parsings and encodings compressed after load
			compress_internally: bool = False,  # save binary of parsings and encodings in the extraction binary
			compress_parsing: bool = True,
			compress_encoding: bool = True,
			encodings: Optional[List[torch.Tensor]] = None,
			parsings: Optional[List[ParsedText]] = None,
	):
		super().__init__()
		self.dependency_parser = dependency_parser
		self.keep_compressed = keep_compressed
		self.compress_internally = compress_internally
		self.compress_parsing = compress_parsing
		self.compress_encoding = compress_encoding
		self._encodings = encodings if encodings is not None else []
		self._parsings = parsings if parsings is not None else []

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

	@staticmethod
	def _get_tokenized_text(words: Words) -> List[str]:
		if isinstance(words, ParsedText):
			return words.tokenized_text.split()
		elif isinstance(words, CompressedParsedText):
			return words.parsed_text.tokenized_text.split()
		elif all(isinstance(word, str) for word in words):
			return words

		raise NotImplementedError()

	def _get_compressed_words(self, words: Words) -> StoredWords:
		if not self.compress_parsing:
			return words

		if isinstance(words, ParsedText):
			if not self.compress_internally:
				self._parsings.append(words)
				return len(self._parsings) - 1

			assert self.dependency_parser is not None
			words = CompressedParsedText(
				bytes_data=words.to_bytes(),
				parsing_egnine=ENGINE_BY_PARSER[type(self.dependency_parser)],
				parser_name=self.dependency_parser.name)
		elif isinstance(words, CompressedParsedText):
			if not self.compress_internally:
				self._parsings.append(words.parsed_text)
				return len(self._parsings) - 1

			words = CompressedParsedText(
				bytes_data=words.bytes_data,
				parsing_egnine=words.parsing_egnine,
				parser_name=words.parser_name)
			words.parsed_text = None
		elif all(isinstance(word, str) for word in words):
			pass
		else:
			raise NotImplementedError()

		return words

	def _get_compressed_encoding(self, encoding: Optional[Encoding]) -> Optional[StoredEncoding]:
		if not self.compress_encoding:
			return encoding

		if isinstance(encoding, np.ndarray):
			encoding = torch.tensor(encoding)

		if isinstance(encoding, torch.Tensor):
			encoding = encoding.clone()

			if not self.compress_internally:
				self._encodings.append(encoding)
				return len(self._encodings) - 1

			encoding = CompressedEncoding(bytes_data=TorchBytesHandler.saves(encoding))
		elif isinstance(encoding, CompressedEncoding):
			if not self.compress_internally:
				self._encodings.append(encoding.encoding)
				return len(self._encodings) - 1

			encoding = CompressedEncoding(bytes_data=encoding.bytes_data)
			encoding.encoding = None
		elif encoding is None:
			return encoding
		else:
			raise NotImplementedError()

		return encoding

	def _get_decompressed_words(
			self, words: StoredWords, parser_by_id: Dict[str, DependencyParser]) -> Words:
		if not self.compress_parsing:
			return words.text_words if isinstance(words, TokenizedText) else words

		if isinstance(words, TokenizedText):
			return self._get_decompressed_words(words.words, parser_by_id)

		if isinstance(words, int):
			return words if self.keep_compressed else self._parsings[words]

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

	def _get_decompressed_encoding(self, encoding: StoredEncoding) -> Encoding:
		if not self.compress_encoding:
			return encoding

		if isinstance(encoding, int):
			return encoding if self.keep_compressed else self._encodings[encoding]

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
			ext.words = self._get_decompressed_words(ext.words, parser_by_id)

			for arg in ext.all_args:
				arg.encoding = self._get_decompressed_encoding(arg.encoding)

		return extraction_obj

	def saves(self, extraction_obj: ExtractionObject) -> bytes:
		extractions = ExtractedBytesHandler._get_extractions(extraction_obj)

		original_words = []
		original_encodings, arguments = [], []
		for ext in tqdm(extractions):
			original_words.append(ext.words)

			if not isinstance(ext.words, TokenizedText):
				ext.words = TokenizedText(self._get_tokenized_text(ext.words), self._get_compressed_words(ext.words))

			for arg in ext.all_args:
				arguments.append(arg)
				original_encodings.append(arg.encoding)
				arg.encoding = self._get_compressed_encoding(arg.encoding)

		bytes_data = super().saves(extraction_obj)

		for words, ext in zip(original_words, extractions):
			ext.words = words

		for encoding, arg in zip(original_encodings, arguments):
			arg.encoding = encoding

		return bytes_data

	def get_indexed_parsings(self) -> List[ParsedText]:
		return self._parsings

	def get_indexed_encodings(self) -> List[torch.Tensor]:
		return self._encodings

	def set_indexed_parsings(self, parsings: List[ParsedText]):
		self._parsings = parsings

	def set_indexed_encodings(self, encodings: List[torch.Tensor]):
		self._encodings = encodings

	def reset_indices(self):
		self._parsings = []
		self._encodings = []
