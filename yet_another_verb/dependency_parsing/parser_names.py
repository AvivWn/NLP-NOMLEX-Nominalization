import pathlib

from yet_another_verb.dependency_parsing.spacy.spacy_parser import SpacyParser
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser

SPACY_ENGINE = "spacy"


parser_by_engine = {
	SPACY_ENGINE: SpacyParser
}
engine_by_parser = {v: k for k, v in parser_by_engine.items()}


def extended_path_by_parser(file_path: str, parser: DependencyParser) -> str:
	# engine and parser should assume not to include '-' character
	return f"{file_path}-{engine_by_parser[type(parser)]}-{parser.name}"


def get_parser_by_extension(file_path: str) -> DependencyParser:
	from yet_another_verb.factories.dependency_parser_factory import DependencyParserFactory

	suffix = pathlib.Path(file_path).suffix
	_, engine, parser_name = suffix.split('-')
	return DependencyParserFactory(parsing_engine=engine, parser_name=parser_name)()
