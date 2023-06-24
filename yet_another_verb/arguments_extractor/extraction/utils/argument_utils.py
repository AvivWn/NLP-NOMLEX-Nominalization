from typing import Union, TYPE_CHECKING

from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.arguments_extractor.extraction.argument.extracted_argument import ExtractedArgument

if TYPE_CHECKING:
	from yet_another_verb.arguments_extractor.extraction.words import Words


def get_argument_words(words: 'Words', argument: ExtractedArgument) -> Union['Words', ParsedSpan]:
	return words[argument.start_idx: argument.end_idx + 1]


def get_argument_text(words: 'Words', argument: ExtractedArgument) -> str:
	arg_words = get_argument_words(words, argument)
	return " ".join([str(word) for word in arg_words])


def get_argument_head_idx(words: 'Words', argument: ExtractedArgument) -> int:
	if argument.head_idx is not None:
		return argument.head_idx

	if isinstance(words, ParsedText):
		return get_argument_words(words, argument).root.i

	raise NotImplementedError()
