from typing import List

from typeguard import typechecked

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.representation import \
	ExtractionRepresentation


class StrRepresentation(ExtractionRepresentation):
	@typechecked
	def _represent_predicate(self, words: List[str], predicate_idx: int) -> str:
		return words[predicate_idx] + f".{predicate_idx}"

	@typechecked
	def _represent_argument(self, words: List[str], predicate_idx: int, argument: ExtractedArgument) -> str:
		return " ".join([words[i] for i in argument.arg_idxs])
