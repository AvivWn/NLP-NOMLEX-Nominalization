from typing import Dict
import os

import unittest
from unittest import TestCase
from parameterized import parameterized

from yet_another_verb.arguments_extractor.extractors.nomlex_args_extractor import NomlexArgsExtractor
from yet_another_verb.arguments_extractor.extraction.representation.parsed_str_representation import \
	ParsedStrRepresentation
from yet_another_verb.dependency_parsing.dependency_parser.parsed_bin import ParsedBin
from yet_another_verb.file_handlers import BinaryFileHandler
from yet_another_verb.utils.debug_utils import timeit
from yet_another_verb.configuration.parsing_config import PARSING_CONFIG
from extracted_examples import EXTRACTIONS_BY_SENTENCE
from config import PARSED_EXAMPLES_PATH
from yet_another_verb.utils.print_utils import print_extraction


class TestNomlexExtractor(TestCase):
	args_extractor: NomlexArgsExtractor
	parsed_bin: ParsedBin
	parsing_by_text: dict

	@classmethod
	def setUpClass(cls) -> None:
		cls.args_extractor = NomlexArgsExtractor()

		cls.parsed_bin = PARSING_CONFIG.DEFAULT_PARSED_BIN_MAKER()
		if os.path.exists(PARSED_EXAMPLES_PATH):
			cls.parsed_bin = cls.parsed_bin.from_bytes(BinaryFileHandler.load(PARSED_EXAMPLES_PATH))

		cls.parsing_by_text = cls.parsed_bin.get_parsing_by_text(cls.args_extractor.dependency_parser)

	@classmethod
	def tearDownClass(cls) -> None:
		BinaryFileHandler.save(PARSED_EXAMPLES_PATH, cls.parsed_bin.to_bytes())

	def _parse_text(self, text: str):
		if text in self.parsing_by_text:
			return self.parsing_by_text[text]

		parsed_text = self.args_extractor.preprocess(text)
		self.parsed_bin.add(parsed_text)
		return parsed_text

	@staticmethod
	def _compare_extractions(extractions, expected_extractions):
		for e in extractions:
			assert e in expected_extractions

		for e in expected_extractions:
			assert e in extractions

	@parameterized.expand(list(EXTRACTIONS_BY_SENTENCE.items()))
	def test_extraction(self, text: str, expected_extractions: Dict[str, Dict[str, str]]):
		parsed_text = self._parse_text(text)
		extractions_per_idx = timeit(self.args_extractor.extract_multiword)(parsed_text)
		extractions_per_word = ParsedStrRepresentation(parsed_text).represent_dict(extractions_per_idx)
		print_extraction(extractions_per_word)

		for word, extractions in extractions_per_word.items():
			word = word.split(".")[0]
			assert word in expected_extractions
			self._compare_extractions(extractions, expected_extractions[word])


if __name__ == "__main__":
	unittest.main()