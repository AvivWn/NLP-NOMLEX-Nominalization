from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.dependency_parsing.dependency_parser.parsed_span import ParsedSpan
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords


def get_argument_in_parsed_text(argument: ExtractedArgument, words: ParsedWords) -> ParsedSpan:
	return words[argument.start_idx: argument.end_idx + 1]
