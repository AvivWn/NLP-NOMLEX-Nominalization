from yet_another_verb.dependency_parsing.spacy.spacy_parser import SpacyParser
from yet_another_verb.dependency_parsing.dependency_parser.dependency_parser import DependencyParser

SPACY_ENGINE = "spacy"


parser_by_engine = {
	SPACY_ENGINE: SpacyParser
}
engine_by_parser = {v: k for k, v in parser_by_engine.items()}


def extended_path_by_parser(file_path: str, parser: DependencyParser) -> str:
	return f"{file_path}-{engine_by_parser[type(parser)]}-{parser.name}"
