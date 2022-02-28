from enum import Enum
from typing import List, Dict
from collections import ChainMap

from typeguard import typechecked

from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction
from yet_another_verb.dependency_parsing.dependency_parser.parsed_text import ParsedWords
from yet_another_verb.arguments_extractor.extraction.representation.representation import \
	ExtractionRepresentation, ArgumentTypes


class BIOTag(str, Enum):
	Begin = "B"
	In = "I"
	Out = "O"


class BIORepresentation(ExtractionRepresentation):
	@typechecked
	def _represent_predicate(self, words: list, predicate_idx: int) -> int:
		return predicate_idx

	def _represent_argument(self, words: list, predicate_idx: int, argument: ExtractedArgument) -> Dict[int, str]:
		tag_by_idx = {}
		start_idx, end_idx = argument.tightest_range

		for i in range(start_idx, end_idx + 1):
			tag_prefix = BIOTag.Begin if i == start_idx else BIOTag.In
			tag_by_idx[i] = f"{tag_prefix}-{argument.arg_tag}"

		return tag_by_idx

	def _represent_extraction(self, extraction: Extraction, arg_types: ArgumentTypes = None) -> List[str]:
		indx_tags_by_arg = super()._represent_extraction(extraction, arg_types)
		tag_by_idx = dict(ChainMap(*indx_tags_by_arg.values()))  # combine
		return [tag_by_idx.get(i, BIOTag.Out) for i, word in enumerate(extraction.words)]
