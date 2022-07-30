from typing import Optional

from yet_another_verb.arguments_extractor.extraction.comparators.extraction_matcher import ExtractionMatcher, \
	ArgsMapping
from yet_another_verb.arguments_extractor.extraction.extraction import Extraction


class ArgTypeMatcher(ExtractionMatcher):
	def _map_args(self, extraction: Extraction, reference: Extraction) -> Optional[ArgsMapping]:
		args_mapping = {}
		for arg in extraction.args:
			if arg.arg_type not in reference.arg_types:
				return None

			args_mapping[arg] = reference.get_arg_by_type(arg.arg_type)

		return args_mapping
