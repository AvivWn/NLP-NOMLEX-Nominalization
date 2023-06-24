import pathlib

from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser
from yet_another_verb.configuration.parsing_config import ENGINE_BY_PARSER


def extended_path_by_parser(file_path: str, parser: DependencyParser) -> str:
	# engine and parser should assume not to include '-' character
	return f"{file_path}-{ENGINE_BY_PARSER[type(parser)]}-{parser.name}"


def get_parser_by_extension(file_path: str) -> DependencyParser:
	from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory

	suffix = pathlib.Path(file_path).suffix
	_, engine, parser_name = suffix.split('-')
	return DependencyParserFactory(parsing_engine=engine, parser_name=parser_name)()
