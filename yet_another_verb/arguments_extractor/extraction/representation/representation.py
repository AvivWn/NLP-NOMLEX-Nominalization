import abc
from typing import List, Dict, Optional

from toolz.dicttoolz import merge_with

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction, Extractions
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.nomlex.constants.argument_type import ArgumentType

ExtractionsByIdx = Dict[int, Extractions]
ArgumentTypes = Optional[List[ArgumentType]]


class ExtractionRepresentation(abc.ABC):
	def __init__(self, words):
		self.words = words

	@abc.abstractmethod
	def represent_predicate(self, predicate_idx: int):
		raise NotImplementedError()

	@abc.abstractmethod
	def represent_argument(self, predicate_idx: int, argument: ExtractedArgument):
		raise NotImplementedError()

	def represent_extraction(self, extraction: Extraction, arg_types: ArgumentTypes = None):
		extraction_repr = {}

		for arg in extraction.args:
			if arg_types is None or arg.arg_type in arg_types:
				extraction_repr[arg.arg_type] = self.represent_argument(extraction.predicate_idx, arg)

		return extraction_repr

	def represent_list(self, extractions: Extractions, arg_types: ArgumentTypes = None) -> list:
		list_repr = [self.represent_extraction(e, arg_types) for e in extractions]
		return list_repr

	def represent_combined_list(self, extractions: Extractions, strict=False, arg_types: ArgumentTypes = None) -> list:
		list_repr = self.represent_list(extractions)

		if not strict:
			combined_repr = merge_with(list, *list_repr)
		else:
			# handle repeated args types
			combined_repr = merge_with(set, *list_repr)
			combined_repr = {k: v.pop() for k, v in combined_repr.items() if len(v) == 1}

			# handle repeated args candidates
			seperate_reversed_repr = [{v: k} for k, v in combined_repr.items()]
			combined_reversed_repr = merge_with(set, *seperate_reversed_repr)
			combined_repr = {v.pop(): k for k, v in combined_reversed_repr.items() if len(v) == 1}

		if arg_types:
			combined_repr = {k: v for k, v in combined_repr.items() if k in arg_types}

		return combined_repr

	def represent_dict(self, extractions_by_idx: ExtractionsByIdx, arg_types: ArgumentTypes = None) -> dict:
		dict_repr = {}

		for predicate_idx, extractions in extractions_by_idx.items():
			extractions_repr = self.represent_list(extractions, arg_types)
			dict_repr[self.represent_predicate(predicate_idx)] = extractions_repr

		return dict_repr

	def represent_combined_dict(
			self, extractions_by_idx: ExtractionsByIdx,
			strict=False, arg_types: ArgumentTypes = None
	) -> dict:
		dict_repr = {}

		for predicate_idx, extractions in extractions_by_idx.items():
			extractions_repr = self.represent_combined_list(extractions, strict, arg_types)
			dict_repr[self.represent_predicate(predicate_idx)] = extractions_repr

		return dict_repr
