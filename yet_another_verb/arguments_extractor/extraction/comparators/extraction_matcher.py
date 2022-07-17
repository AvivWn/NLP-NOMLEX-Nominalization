import abc
from typing import Optional, Dict

from yet_another_verb.arguments_extractor.extraction import Extraction, Extractions, ExtractedArgument

ArgsMapping = Dict[ExtractedArgument, ExtractedArgument]


class ExtractionMatcher(abc.ABC):
	@abc.abstractmethod
	def _map_args(self, extraction: Extraction, reference: Extraction) -> Optional[ArgsMapping]:
		pass

	@staticmethod
	def _update_mapped_args(arg_mapping: ArgsMapping):
		for arg, ref_arg in arg_mapping.items():
			arg.arg_tag = ref_arg.arg_tag

	def isin(self, extraction: Extraction, references: Extractions):
		for reference in references:
			args_mapping = self._map_args(extraction, reference)

			if args_mapping is not None:
				self._update_mapped_args(args_mapping)
				return True

		return False

	def filter_by(self, extractions: Extractions, references: Extractions):
		filtered_extractions = []
		for extraction in extractions:
			if self.isin(extraction, references):
				filtered_extractions.append(extraction)

		return filtered_extractions
