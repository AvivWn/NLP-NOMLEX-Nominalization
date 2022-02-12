from argparse import ArgumentParser
from typing import Optional

from yet_another_verb.factories.factory import Factory
from yet_another_verb.configuration import PARSING_CONFIG
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.spacy.spacy_parser import SpacyParser


class DependencyParserFactory(Factory):
	def __init__(self, dependency_parsing_engine: str = PARSING_CONFIG.PARSING_ENGINE, **kwargs):
		self.dependency_parsing_engine = dependency_parsing_engine
		self.params = kwargs

	def __call__(self) -> DependencyParser:
		if self.dependency_parsing_engine == "spacy":
			return SpacyParser(**self.params)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser = DependencyParserFactory._expand_optional_parser(arg_parser)
		arg_parser.add_argument(
			'--dependency-parsing-engine', '-e', default=PARSING_CONFIG.PARSING_ENGINE,
			help="The engine which will parse the sentences"
		)
		arg_parser.add_argument(
			'--parser-name', '-p', default=PARSING_CONFIG.PARSER_NAME,
			help="The name of the parsing model"
		)
		return arg_parser
