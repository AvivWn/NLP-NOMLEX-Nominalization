from typing import Dict

import unittest
from unittest import TestCase
from parameterized import parameterized

from yet_another_verb.arguments_extractor.extractors.nomlex_args_extractor import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extraction.representation.parsed_str_representation import \
	ParsedStrRepresentation
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.file_handlers import ParsedBinFileHandler
from yet_another_verb.utils.debug_utils import timeit
from extracted_examples import EXTRACTIONS_BY_SENTENCE
from config import PARSED_EXAMPLES_PATH
from yet_another_verb.utils.print_utils import print_multi_word_extraction


class TestNomlexExtractor(TestCase):
	args_extractor: NomlexArgsExtractor
	parsed_file_handler: ParsedBinFileHandler
	parsed_bin: ParsedBin
	parsing_by_text: dict

	@classmethod
	def setUpClass(cls) -> None:
		cls.args_extractor = NomlexArgsExtractor()

		dependency_parser = DependencyParserFactory()()
		cls.parsed_file_handler = ParsedBinFileHandler(dependency_parser)
		cls.parsed_bin = cls.parsed_file_handler.load(PARSED_EXAMPLES_PATH)
		cls.parsing_by_text = cls.parsed_bin.get_parsing_by_text()

	@classmethod
	def tearDownClass(cls) -> None:
		cls.parsed_file_handler.save(PARSED_EXAMPLES_PATH, cls.parsed_bin)

	@staticmethod
	def _compare_extractions(extractions, expected_extractions):
		for e in extractions:
			assert e in expected_extractions

		for e in expected_extractions:
			assert e in extractions

	@parameterized.expand(list(EXTRACTIONS_BY_SENTENCE.items()))
	def test_extraction(self, text: str, expected_extractions: Dict[str, Dict[str, str]]):
		multi_word_extraction = timeit(self.args_extractor.extract_multiword)(self.parsing_by_text.get(text, text))
		parsed_text = multi_word_extraction.words

		if text not in self.parsing_by_text:
			self.parsing_by_text[text] = parsed_text
			self.parsed_bin.add(parsed_text)

		extractions_per_word = ParsedStrRepresentation().represent_dict(multi_word_extraction)
		print_multi_word_extraction(extractions_per_word)

		assert len(extractions_per_word) == len(expected_extractions)

		for word, extractions in extractions_per_word.items():
			word = word.split(".")[0]
			assert word in expected_extractions
			self._compare_extractions(extractions, expected_extractions[word])


if __name__ == "__main__":
	unittest.main()
