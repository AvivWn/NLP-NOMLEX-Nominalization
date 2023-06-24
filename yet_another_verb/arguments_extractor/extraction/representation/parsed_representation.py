from typeguard import typechecked

from yet_another_verb.arguments_extractor.extraction.utils.argument_utils import get_argument_words
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.representation import \
	ExtractionRepresentation


class ParsedRepresentation(ExtractionRepresentation):
	@typechecked
	def _represent_predicate(self, words: ParsedWords, predicate_idx: int) -> ParsedWord:
		return words[predicate_idx]

	@typechecked
	def _represent_argument(self, words: ParsedWords, predicate_idx: int, argument: ExtractedArgument) -> ParsedSpan:
		return get_argument_words(words, argument)
