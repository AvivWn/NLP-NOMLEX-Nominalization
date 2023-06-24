from typing import Dict, List
from os.path import exists

import unittest
from unittest import TestCase
from parameterized import parameterized

from yet_another_verb import ArgsExtractor
from yet_another_verb.arguments_extractor.extraction import MultiWordExtraction
from yet_another_verb.arguments_extractor.extraction.representation import ParsedStrRepresentation
from yet_another_verb.configuration.parsing_config import PARSING_CONFIG
from yet_another_verb.dependency_parsing.parser_names import extended_path_by_parser
from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory
from yet_another_verb.data_handling import PKLFileHandler, ExtractedBytesHandler
from yet_another_verb.factories.extractor_factory import ExtractorFactory
from yet_another_verb.nomlex.nomlex_version import NomlexVersion
from yet_another_verb.utils.debug_utils import timeit
from extracted_examples import NOMLEX_EXTRACTIONS_BY_SENTENCE, VERB_PATTERNS_EXTRACTIONS_BY_SENTENCE
from config import BINARY_EXTRACTION_PATH
from yet_another_verb.utils.print_utils import print_multi_word_extraction

ExtractionRepr = Dict[str, str]
ExtractionsRepr = List[ExtractionRepr]


class TestExtractor(TestCase):
	__test__ = False
	args_extractor: ArgsExtractor
	extracted_bytes_handler: ExtractedBytesHandler
	binary_extraction_path: str
	binary_extraction_by_text: Dict[str, bytes]

	@classmethod
	def set_up_class(cls, extractor_name: str, parser_params: dict, extractor_params: dict) -> None:
		cls.args_extractor = ExtractorFactory(**extractor_params)()
		dependency_parser = DependencyParserFactory(**parser_params)()

		cls.extracted_bytes_handler = ExtractedBytesHandler(dependency_parser, compress_internally=True)
		cls.binary_extraction_path = extended_path_by_parser(
			BINARY_EXTRACTION_PATH.format(extractor=extractor_name), dependency_parser)
		if exists(cls.binary_extraction_path):
			cls.binary_extraction_by_text = PKLFileHandler.load(cls.binary_extraction_path)
		else:
			cls.binary_extraction_by_text = {}

	@classmethod
	def tearDownClass(cls) -> None:
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
		bytes_repr = self.extracted_bytes_handler.saves(multi_word_extraction)

		if expected_bytes_repr is None:
			self.binary_extraction_by_text[text] = bytes_repr
		else:
			assert bytes_repr == expected_bytes_repr

	def _test_extraction(self, text: str, expected_extractions: Dict[str, ExtractionsRepr]):
		multi_word_extraction = timeit(self.args_extractor.extract_multiword)(text)

		extractions_per_word = ParsedStrRepresentation().represent_by_word(multi_word_extraction)
		print_multi_word_extraction(extractions_per_word)

		assert len(extractions_per_word) == len(expected_extractions)

		for word, extractions in extractions_per_word.items():
			word = word.split(".")[0]
			assert word in expected_extractions
			self._compare_extractions(extractions, expected_extractions[word])

		self._compare_binary(text, multi_word_extraction)


class TestNomlexExtractor(TestExtractor):
	__test__ = True

	@classmethod
	def setUpClass(cls):
		extraction_mode = "nomlex"
		parser_params = {"parser_name": PARSING_CONFIG.PARSER_NAME}

		cls.set_up_class(
			extractor_name=extraction_mode,
			parser_params=parser_params,
			extractor_params={"extraction_mode": extraction_mode, "nomlex_version": NomlexVersion.V2, **parser_params}
		)

	@parameterized.expand(list(NOMLEX_EXTRACTIONS_BY_SENTENCE.items()))
	def test_extraction(self, text: str, expected_extractions: Dict[str, ExtractionsRepr]):
		super()._test_extraction(text, expected_extractions)


class TestVerbPatternsExtractor(TestExtractor):
	__test__ = True

	@classmethod
	def setUpClass(cls):
		extraction_mode = "verb-patterns"
		parser_params = {"parser_name": PARSING_CONFIG.PARSER_NAME}

		cls.set_up_class(
			extractor_name=extraction_mode,
			parser_params=parser_params,
			extractor_params={"extraction_mode": extraction_mode, **parser_params}
		)

	@parameterized.expand(list(VERB_PATTERNS_EXTRACTIONS_BY_SENTENCE.items()))
	def test_extraction(self, text: str, expected_extractions: Dict[str, ExtractionsRepr]):
		super()._test_extraction(text, expected_extractions)


if __name__ == "__main__":
	unittest.main()
