from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.dependency_parser.parsed_word import ParsedWord
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.parsed_representation import \
	ParsedRepresentation


class TupleRepresentation(ParsedRepresentation):
	def __init__(self, words: ParsedText):
		super().__init__(words)

	def _arg_as_tuple(self, arg: ParsedSpan, predicate: ParsedWord):
		arg_head_idx = arg.root.i if arg else -1
		arg_start_idx = arg[0].i if arg else -1
		arg_end_idx = arg[-1].i + 1 if arg else -1
		arg_str = arg.tokenized_text if arg else ""

		arg_tuple = \
			self.words.tokenized_text, \
			predicate.i, predicate.text, predicate.lemma.lower(), predicate.pos, \
			arg_head_idx, arg_start_idx, arg_end_idx, arg_str

		return arg_tuple

	def represent_predicate(self, predicate_idx: int):
		return super().represent_predicate(predicate_idx)

	def represent_argument(self, predicate_idx: int, argument: ExtractedArgument):
		arg_span = super().represent_argument(predicate_idx, argument)
		return self._arg_as_tuple(arg_span, self.words[predicate_idx])
