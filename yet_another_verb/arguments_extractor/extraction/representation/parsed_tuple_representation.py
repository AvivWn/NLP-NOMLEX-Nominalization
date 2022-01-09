from typeguard import typechecked

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.parsed_representation import \
	ParsedRepresentation


class ParsedTupleRepresentation(ParsedRepresentation):
	@staticmethod
	def _arg_as_tuple(words: ParsedWords, arg: ParsedSpan, predicate: ParsedWord):
		arg_head_idx = arg.root.i if arg else -1
		arg_start_idx = arg[0].i if arg else -1
		arg_end_idx = arg[-1].i + 1 if arg else -1
		arg_str = arg.tokenized_text if arg else ""

		arg_tuple = \
			words.tokenized_text, \
			predicate.i, predicate.text, predicate.lemma.lower(), predicate.pos, \
			arg_head_idx, arg_start_idx, arg_end_idx, arg_str

		return arg_tuple

	@typechecked
	def _represent_predicate(self, words: ParsedWords, predicate_idx: int) -> ParsedWord:
		return super()._represent_predicate(words, predicate_idx)

	@typechecked
	def _represent_argument(self, words: ParsedWords, predicate_idx: int, argument: ExtractedArgument) -> tuple:
		arg_span = super()._represent_argument(words, predicate_idx, argument)
		return self._arg_as_tuple(words, arg_span, words[predicate_idx])
