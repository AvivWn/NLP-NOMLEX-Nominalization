import abc
from typing import List, Optional

from toolz.dicttoolz import merge_with

from yet_another_verb.arguments_extractor.extraction.extraction import Extraction, Extractions
from yet_another_verb.arguments_extractor.extraction.extracted_argument import ExtractedArgument
from yet_another_verb.arguments_extractor.extraction.multi_word_extraction import MultiWordExtraction
from yet_another_verb.nomlex.constants.argument_type import ArgumentType

ArgumentTypes = Optional[List[ArgumentType]]


class ExtractionRepresentation(abc.ABC):
	@abc.abstractmethod
	def _represent_predicate(self, words: list, predicate_idx: int):
		raise NotImplementedError()

	@abc.abstractmethod
	def _represent_argument(self, words: list, predicate_idx: int, argument: ExtractedArgument):
		raise NotImplementedError()

	def _represent_extraction(self, extraction: Extraction, arg_types: ArgumentTypes = None) -> dict:
		extraction_repr = {}

		for arg in extraction.args:
			if arg_types is None or arg.arg_type in arg_types:
				extraction_repr[arg.arg_tag] = self._represent_argument(extraction.words, extraction.predicate_idx, arg)

		return extraction_repr

	def represent_list(self, extractions: Extractions, arg_types: ArgumentTypes = None) -> list:
		list_repr = [self._represent_extraction(e, arg_types) for e in extractions]
		return list_repr

	def represent_combined_list(self, extractions: Extractions, strict=False, arg_types: ArgumentTypes = None) -> dict:
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

	def represent_dict(self, multi_word_ext: MultiWordExtraction, arg_types: ArgumentTypes = None) -> dict:
		dict_repr = {}

		for predicate_idx, extractions in multi_word_ext.extractions_per_idx.items():
			extractions_repr = self.represent_list(extractions, arg_types)
			dict_repr[self._represent_predicate(multi_word_ext.words, predicate_idx)] = extractions_repr

		return dict_repr

	def represent_combined_dict(
			self, multi_word_ext: MultiWordExtraction,
			strict=False, arg_types: ArgumentTypes = None
	) -> dict:
		dict_repr = {}

		for predicate_idx, extractions in multi_word_ext.extractions_per_idx.items():
			extractions_repr = self.represent_combined_list(extractions, strict, arg_types)
			dict_repr[self._represent_predicate(multi_word_ext.words, predicate_idx)] = extractions_repr

		return dict_repr
