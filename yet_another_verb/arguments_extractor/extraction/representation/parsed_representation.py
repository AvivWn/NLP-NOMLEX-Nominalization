from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.representation import \
	ExtractionRepresentation


class ParsedRepresentation(ExtractionRepresentation):
	def __init__(self, words: ParsedText):
		super().__init__(words)

	def represent_predicate(self, predicate_idx: int) -> ParsedWord:
		return self.words[predicate_idx]

	def represent_argument(self, predicate_idx: int, argument: ExtractedArgument) -> ParsedSpan:
		min_idx = min(argument.arg_idxs)
		max_idx = max(argument.arg_idxs) + 1
		return self.words[min_idx:max_idx]
