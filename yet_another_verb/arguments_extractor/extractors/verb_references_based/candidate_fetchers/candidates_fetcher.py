import abc

from yet_another_verb.arguments_extractor.extraction import ExtractedArguments
from yet_another_verb.arguments_extractor.extractors.verb_references_based.method_params import MethodParams
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedText


class CandidatesFetcher(abc.ABC):
	def fetch_arguments(self, word_idx: int, words: ParsedText, method_params: MethodParams) -> ExtractedArguments:
		pass
