from typing import Dict, List
from os.path import exists

import unittest
from unittest import TestCase
from parameterized import parameterized

from yet_another_verb.arguments_extractor.extraction import MultiWordExtraction
from yet_another_verb.arguments_extractor.extractors.nomlex_args_extractor import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extraction.representation import ParsedStrRepresentation
from yet_another_verb.dependency_parsing.parser_names import extended_path_by_parser
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.data_handling import ParsedBinFileHandler, PKLFileHandler, ExtractedBytesHandler
from yet_another_verb.utils.debug_utils import timeit
from extracted_examples import EXTRACTIONS_BY_SENTENCE
from config import PARSED_EXAMPLES_PATH, BINARY_EXTRACTION_PATH
from yet_another_verb.utils.print_utils import print_multi_word_extraction

ExtractionRepr = Dict[str, str]
ExtractionsRepr = List[ExtractionRepr]


class TestNomlexExtractor(TestCase):
	args_extractor: NomlexArgsExtractor
	parsed_file_handler: ParsedBinFileHandler
	parsed_bin: ParsedBin
	parsing_by_text: dict
	binary_extraction_path: str
	binary_extraction_by_text: Dict[str, bytes]

	@classmethod
	def setUpClass(cls) -> None:
		cls.args_extractor = NomlexArgsExtractor()

		dependency_parser = DependencyParserFactory()()
		cls.parsed_file_handler = ParsedBinFileHandler(dependency_parser)
		cls.parsed_bin = cls.parsed_file_handler.load(PARSED_EXAMPLES_PATH)
		cls.parsing_by_text = cls.parsed_bin.get_parsing_by_text()

		cls.extracted_file_handler = ExtractedBytesHandler(dependency_parser)
		cls.binary_extraction_path = extended_path_by_parser(BINARY_EXTRACTION_PATH, dependency_parser)
		if exists(cls.binary_extraction_path):
			cls.binary_extraction_by_text = PKLFileHandler.load(cls.binary_extraction_path)
		else:
			cls.binary_extraction_by_text = {}

	@classmethod
	def tearDownClass(cls) -> None:
		cls.parsed_file_handler.save(PARSED_EXAMPLES_PATH, cls.parsed_bin)

		if not exists(cls.binary_extraction_path):
			PKLFileHandler.save(cls.binary_extraction_path, cls.binary_extraction_by_text)

	@staticmethod
	def _compare_extractions(extractions: ExtractionsRepr, expected_extractions: ExtractionsRepr):
		for e in extractions:
			assert e in expected_extractions

		for e in expected_extractions:
			assert e in extractions

	def _compare_binary(self, text: str, multi_word_extraction: MultiWordExtraction):
		expected_bytes_repr = self.binary_extraction_by_text.get(text)
		bytes_repr = self.extracted_file_handler.saves(multi_word_extraction)

		if expected_bytes_repr is None:
			self.binary_extraction_by_text[text] = bytes_repr

		assert bytes_repr == expected_bytes_repr

	@parameterized.expand(list(EXTRACTIONS_BY_SENTENCE.items()))
	def test_extraction(self, text: str, expected_extractions: Dict[str, ExtractionsRepr]):
		multi_word_extraction = timeit(self.args_extractor.extract_multiword)(self.parsing_by_text.get(text, text))
		parsed_text = multi_word_extraction.words

		if text not in self.parsing_by_text:
			self.parsing_by_text[text] = parsed_text
			self.parsed_bin.add(parsed_text)

		extractions_per_word = ParsedStrRepresentation().represent_by_word(multi_word_extraction)
		print_multi_word_extraction(extractions_per_word)

		assert len(extractions_per_word) == len(expected_extractions)

		for word, extractions in extractions_per_word.items():
			word = word.split(".")[0]
			assert word in expected_extractions
			self._compare_extractions(extractions, expected_extractions[word])

		self._compare_binary(text, multi_word_extraction)


if __name__ == "__main__":
	unittest.main()
