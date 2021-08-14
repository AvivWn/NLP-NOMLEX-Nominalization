from typing import List

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.representation import \
	ExtractionRepresentation


class StrRepresentation(ExtractionRepresentation):
	def __init__(self, words: List[str]):
		super().__init__(words)

	def represent_predicate(self, predicate_idx: int) -> str:
		return self.words[predicate_idx] + f".{predicate_idx}"

	def represent_argument(self, predicate_idx: int, argument: ExtractedArgument) -> str:
		return " ".join([self.words[i] for i in argument.arg_idxs])
