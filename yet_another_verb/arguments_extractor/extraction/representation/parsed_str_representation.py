from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.parsed_representation import \
	ParsedRepresentation


class ParsedStrRepresentation(ParsedRepresentation):
	def __init__(self, words: ParsedText):
		super().__init__(words)

	def represent_predicate(self, predicate_idx: int) -> str:
		return super().represent_predicate(predicate_idx).text + f".{predicate_idx}"

	def represent_argument(self, predicate_idx: int, argument: ExtractedArgument) -> str:
		return super().represent_argument(predicate_idx, argument).text
