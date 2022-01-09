from typeguard import typechecked

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.parsed_representation import \
	ParsedRepresentation


class ParsedStrRepresentation(ParsedRepresentation):
	@typechecked
	def _represent_predicate(self, words: ParsedWords, predicate_idx: int) -> str:
		return super()._represent_predicate(words, predicate_idx).text + f".{predicate_idx}"

	@typechecked
	def _represent_argument(self, words: ParsedWords, predicate_idx: int, argument: ExtractedArgument) -> str:
		return super()._represent_argument(words, predicate_idx, argument).text
