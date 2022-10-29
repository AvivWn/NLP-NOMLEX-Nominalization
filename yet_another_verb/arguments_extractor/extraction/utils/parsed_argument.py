from yet_another_verb.arguments_extractor.extraction import ExtractedArgument
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords


def get_argument_in_parsed_text(argument: ExtractedArgument, words: ParsedWords):
	start_idx, end_idx = argument.tightest_range
	return words[start_idx: end_idx + 1]
