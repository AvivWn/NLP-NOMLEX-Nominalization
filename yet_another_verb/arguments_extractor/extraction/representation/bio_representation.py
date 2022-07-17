from enum import Enum
from typing import List, Dict, Optional
from collections import ChainMap

from typeguard import typechecked

from yet_another_verb.arguments_extractor.extraction import Extraction, ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.representation.representation import \
	ExtractionRepresentation
from yet_another_verb.arguments_extractor.extraction.argument.argument_type import ArgumentTypes


class BIOTag(str, Enum):
	Begin = "B"
	In = "I"
	Out = "O"


class BIORepresentation(ExtractionRepresentation):
	def __init__(self, tag_predicate: bool, arg_types: Optional[ArgumentTypes] = None):
		super().__init__(arg_types)
		self.tag_predicate = tag_predicate

	@typechecked
	def _represent_predicate(self, words: list, predicate_idx: int) -> int:
		return predicate_idx

	def _represent_argument(self, words: list, predicate_idx: int, argument: ExtractedArgument) -> Dict[int, str]:
		tag_by_idx = {}
		start_idx, end_idx = argument.tightest_range

		for i in range(start_idx, end_idx + 1):
			tag_prefix = BIOTag.Begin if i == start_idx else BIOTag.In

			if start_idx != predicate_idx or self.tag_predicate:
				tag_by_idx[i] = f"{tag_prefix}-{argument.arg_tag}"

		return tag_by_idx

	def represent_single(self, extraction: Extraction) -> List[str]:
		indx_tags_by_arg = super().represent_single(extraction)
		tag_by_idx = dict(ChainMap(*indx_tags_by_arg.values()))  # combine
		return [tag_by_idx.get(i, BIOTag.Out) for i, word in enumerate(extraction.words)]
