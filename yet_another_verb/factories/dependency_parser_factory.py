from argparse import ArgumentParser
from typing import Optional

from yet_another_verb.factories.factory import Factory
from yet_another_verb.configuration import PARSING_CONFIG
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.dependency_parsing.parser_names import parser_by_engine


class DependencyParserFactory(Factory):
	def __init__(self, parsing_engine: str = PARSING_CONFIG.PARSING_ENGINE, **kwargs):
		self.dependency_parsing_engine = parsing_engine
		self.params = kwargs

	def __call__(self) -> DependencyParser:
		return parser_by_engine[self.dependency_parsing_engine](**self.params)

	@staticmethod
	def expand_parser(arg_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
		arg_parser.add_argument(
			'--parsing-engine', '-e', default=PARSING_CONFIG.PARSING_ENGINE,
			help="The engine which will parse the sentences"
		)
		arg_parser.add_argument(
			'--parser-name', '-p', default=PARSING_CONFIG.PARSER_NAME,
			help="The name of the parsing model"
		)
		return arg_parser
